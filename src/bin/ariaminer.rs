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
use ariaminer::official_proof::encode_base64;
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
#[command(name = "ariaminer", about = "ARIAMiner GPU — Blackwell-native Pearl (PRL) miner · 1% dev-fee (announced)")]
struct Args {
    #[arg(long)]
    pool: Option<String>,
    #[arg(long)]
    wallet: Option<String>,
    #[arg(long, default_value = "aria")]
    worker: String,
    #[arg(long, default_value = "x")]
    password: String,
    /// CPU grind threads. Default: all logical processors.
    #[arg(long)]
    threads: Option<usize>,
    /// Expose a JSON stats endpoint on 127.0.0.1:<port> (for HiveOS / monitoring).
    #[arg(long)]
    stats_port: Option<u16>,
    /// Wire dialect: "pearl" (AriaPool object-wire) or "luckypool". Default:
    /// auto-detected from the pool host ("luckypool" substring).
    #[arg(long)]
    dialect: Option<String>,
}

/// LuckyPool sends no `pearl.set_mining_params` — the Pearl mainnet defaults
/// apply (m=n=131072, k=4096, rank=256). m/n are advisory (the miner picks its
/// own batch via ARIA_BATCH_M/N — they are not part of the job_key); the
/// patterns are the canonical GPU MMA fragment (same lists as
/// `canonical_gpu_config`, which the GPU path re-derives anyway).
fn luckypool_default_params() -> MiningParams {
    MiningParams {
        m: 131_072,
        n: 131_072,
        k: 4096,
        rank: 256,
        rows_pattern: vec![0, 8, 32, 40, 64, 72, 96, 104],
        cols_pattern: vec![0, 1, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113],
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
                        if let Ok(proof_base64) = encode_base64(&proof) {
                            shares.fetch_add(1, Ordering::Relaxed);
                            credited.fetch_add(diff_arc.load(Ordering::Relaxed), Ordering::Relaxed);
                            let _ = submit_tx.try_send(Submission {
                                job_id: job.job_id.clone(),
                                proof_base64,
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
                        let gt0 = Instant::now();
                        let (found, hits) = ctx.grind2(setup_seed, next_seed, job_key, bound_le);
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
                                    let a = ProofGpuCtx::new(gm, k, 256);
                                    let b = ProofGpuCtx::new(gn, k, 256);
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
                                    if let Ok(proof_base64) = encode_base64(proof) {
                                        shares.fetch_add(1, Ordering::Relaxed);
                                        credited.fetch_add(diff_arc.load(Ordering::Relaxed), Ordering::Relaxed);
                                        let _ = submit_tx.try_send(Submission {
                                            job_id: job.job_id.clone(), proof_base64,
                                        });
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
    println!("  {WHT}{B}⛏  A R I A M I N E R{RST}   {GRY}GPU · v{ver}{RST}");
    println!("  {GRY}{line}{RST}");
    println!(
        "  {DIM}Pearl (PRL) · Blackwell IMMA · GEMM Int7×Int7 ·{RST} {YEL}1% dev-fee{RST}"
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
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    // GPU build inclus : on garde le multi-thread — chaque thread prépare son
    // attempt (signal+commitment+noise sur CPU) en parallèle et nourrit le GPU.
    // Le prologue CPU est le goulot, donc plus de threads = GPU mieux alimenté.
    let threads = args
        .threads
        .unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(1));

    let pool = args.pool.as_deref().unwrap_or("");
    let wallet = args.wallet.as_deref().unwrap_or("");

    let (host, port) = if let Some((h, p)) = pool.rsplit_once(':') {
        (h, p.parse::<u16>().unwrap_or(1200))
    } else {
        ("127.0.0.1", 1200)
    };

    print_banner(if pool.is_empty() { "WSS Proxy" } else { pool }, if wallet.is_empty() { "Proxy Injected" } else { wallet }, &args.worker, threads);

    let (job_tx, mut job_rx) = broadcast::channel::<JobEvent>(64);
    let (submit_tx, submit_rx) = mpsc::channel::<Submission>(256);

    let dialect = match args.dialect.as_deref() {
        Some("luckypool") => Dialect::LuckyPool,
        Some("pearl") => Dialect::Pearl,
        Some(other) => anyhow::bail!("--dialect must be \"pearl\" or \"luckypool\", got {other}"),
        None => Dialect::LuckyPool,
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
            if std::env::var("ARIA_FIXB").is_err() { std::env::set_var("ARIA_FIXB", "0"); }
        }
        // Bound = règle CONSENSUS post-softfork (jackpot LE ≤ target × h·w ×
        // (dpl/rank) × 128, dpl = k − k%rank) : à rank 128 c'est l'ancien ×h·w·k.
        // Mode LE (pas ARIA_BE). [[fold validé byte-exact]]
        tracing::info!(
            "dialecte LuckyPool : multistage TMA + rank {} + 131072² + bound rank-penalty ×h·w·(dpl/rank)·128",
            ariaminer::official_grind::rank_from_env()
        );
    }

    let scfg = StratumConfig {
        host: host.to_string(),
        port,
        wallet: wallet.to_string(),
        worker: args.worker.clone(),
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
                    "{CYA}{B}⚡ {:>7.2} TH/s{RST}  {GRY}│{RST}  {GRN}✓ {} shares{RST}  {GRY}│{RST}  {DIM}{:.0} setups/s{RST}  {GRY}│{RST}  {DIM}up {}{RST}",
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
                tracing::info!(m = p.m, n = p.n, k = p.k, rank = p.rank, "mining params");
                cur_params = Some(p);
            }
            Ok(JobEvent::SetDifficulty(d)) => {
                tracing::info!(difficulty = d, "difficulty");
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
                    tracing::warn!("job before params — waiting for set_mining_params");
                    continue;
                };
                // LuckyPool: no set_difficulty — the diff rides in the job_id
                // suffix (display/credited counters) and the authoritative bound
                // is the job's full 256-bit target (applied below).
                if dialect == Dialect::LuckyPool {
                    if let Some(d) = diff_from_job_id(&job.job_id) {
                        if d != cur_difficulty {
                            tracing::info!(difficulty = d, "difficulty (from job_id)");
                        }
                        cur_difficulty = d;
                        diff_arc.store(d, Ordering::Relaxed);
                    }
                }
                let gj = match build_grind_job(&params, &job, cur_difficulty) {
                    Ok(mut g) => {
                        if dialect == Dialect::LuckyPool {
                            if let Some(t) = job.full_target {
                                // Règle CONSENSUS rank-penalty (v1.3.0, mainnet 96251) :
                                // jackpot LE ≤ target × h·w × (dpl/rank) × 128, avec
                                // h·w = tuile 8×16 = 128 et dpl = k − k%rank (arithmétique
                                // entière IDENTIQUE à zk-pow sanity_checks.rs). À rank 128
                                // le facteur vaut l'ancien ×h·w·k exactement.
                                let rank = (params.rank as u64).max(1);
                                let dpl = params.k as u64 - params.k as u64 % rank;
                                let tile_size = (params.rows_pattern.len() as u64) * (params.cols_pattern.len() as u64);
                                let factor = tile_size * dpl;
                                g.official.bound_le = ariaminer::stratum_to_official::scaled_bound_le_from_target_be(&t, factor);
                            }
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
                        tracing::info!(job_id = %job.job_id, "→ starting official grind");
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
