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
#[cfg(feature = "gpu")]
use ariaminer::official_grind::{canonical_gpu_config, try_mine_one_bounded_gpu_resident_ctx};
use ariaminer::official_proof::encode_base64;
use ariaminer::protocol::{Job, MiningParams};
use ariaminer::stratum::{JobEvent, StratumConfig, Submission, run as stratum_run};
use ariaminer::stratum_to_official::{OfficialJob, build_official_job};
use clap::Parser;
use parking_lot::Mutex;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::{broadcast, mpsc};

#[derive(Parser, Debug)]
#[command(name = "ariaminer", about = "ARIAMiner — open-source 0% dev-fee CPU miner for Pearl (real PlainProof)")]
struct Args {
    #[arg(long)]
    pool: String,
    #[arg(long)]
    wallet: String,
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
}

/// Hashrate display convention: same TMACs/s ("TH/s") multiplier the pool uses,
/// so the miner/HiveOS number matches the pool dashboard for the same rig.
const HASHRATE_DISPLAY_MULT: f64 = 4.3e6;

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

#[allow(clippy::too_many_arguments)]
fn spawn_grind(
    job_slot: Arc<Mutex<Arc<GrindJob>>>,
    threads: usize,
    attempts: Arc<AtomicU64>,
    shares: Arc<AtomicU64>,
    credited: Arc<AtomicU64>,
    diff_arc: Arc<AtomicU64>,
    submit_tx: mpsc::Sender<Submission>,
) -> GrindGen {
    let stop = Arc::new(AtomicBool::new(false));
    let handles = (0..threads)
        .map(|tid| {
            let job_slot = Arc::clone(&job_slot);
            let stop = Arc::clone(&stop);
            let attempts = Arc::clone(&attempts);
            let shares = Arc::clone(&shares);
            let credited = Arc::clone(&credited);
            let diff_arc = Arc::clone(&diff_arc);
            let submit_tx = submit_tx.clone();
            thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(
                    0xA51A_0000 ^ (tid as u64) ^ (Instant::now().elapsed().as_nanos() as u64),
                );
                // One reusable workspace per thread — the hot loop allocates nothing.
                let mut ws = Workspace::new();
                // GPU : contexte résident persistant (alloué une fois, sur le 1er job).
                #[cfg(feature = "gpu")]
                let mut gpu_ctx: Option<ariaminer::gpu_ffi::ResidentCtx> = None;
                while !stop.load(Ordering::Relaxed) {
                    // Adopt the latest job at each setup boundary.
                    let job = job_slot.lock().clone();
                    let oj = &job.official;
                    // One faithful attempt: draw matrices, noise, sweep every
                    // PeriodicPattern tile against the share bound.
                    // CPU: the pool's tile config on the AVX micro-kernel.
                    #[cfg(not(feature = "gpu"))]
                    let hit = try_mine_one_bounded(
                        &mut ws, &mut rng, oj.m, oj.n, oj.k, &oj.header, &oj.config, &oj.bound_le,
                    );
                    // GPU: the GEMM IMMA kernel folds 8×16 MMA fragments, so we mine
                    // with `canonical_gpu_config` (the fragment tile shape). The share
                    // bound is config-independent (= MAX/difficulty), so it's unchanged;
                    // m/n are forced to multiples of 128 (kernel CTA tile). The node
                    // reconstructs this exact config from the submitted indices → verify OK.
                    #[cfg(feature = "gpu")]
                    let hit = {
                        let gconf = canonical_gpu_config(oj.k as u32);
                        let gm = oj.m.div_ceil(128) * 128;
                        let gn = oj.n.div_ceil(128) * 128;
                        // (Ré)alloue le contexte résident si la forme change (alloc 1× sinon).
                        if gpu_ctx.as_ref().map(|c| c.dims()) != Some((gm, gn, oj.k)) {
                            gpu_ctx = Some(ariaminer::gpu_ffi::ResidentCtx::new(gm, gn, oj.k, 64));
                        }
                        let ctx = gpu_ctx.as_ref().unwrap();
                        try_mine_one_bounded_gpu_resident_ctx(
                            ctx, &mut ws, &mut rng, &oj.header, &gconf, &oj.bound_le,
                        )
                    };
                    attempts.fetch_add(1, Ordering::Relaxed);
                    if let Some(proof) = hit {
                        match encode_base64(&proof) {
                            Ok(proof_base64) => {
                                shares.fetch_add(1, Ordering::Relaxed);
                                // Credit this share's difficulty (pool's basis for hashrate).
                                credited.fetch_add(diff_arc.load(Ordering::Relaxed), Ordering::Relaxed);
                                tracing::info!(job_id = %job.job_id, "✅ share (real PlainProof) — submitting");
                                let _ = submit_tx.try_send(Submission {
                                    job_id: job.job_id.clone(),
                                    proof_base64,
                                });
                            }
                            Err(e) => tracing::error!(error = %e, "encode PlainProof failed"),
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
            submit_tx.clone(),
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

    let (host, port) = args
        .pool
        .rsplit_once(':')
        .ok_or_else(|| anyhow::anyhow!("--pool must be host:port"))?;
    let port: u16 = port.parse()?;

    tracing::info!(pool = %args.pool, worker = %args.worker, threads, "ariaminer starting (real PlainProof)");

    let (job_tx, mut job_rx) = broadcast::channel::<JobEvent>(64);
    let (submit_tx, submit_rx) = mpsc::channel::<Submission>(256);

    let scfg = StratumConfig {
        host: host.to_string(),
        port,
        wallet: args.wallet.clone(),
        worker: args.worker.clone(),
        password: args.password.clone(),
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
    let hashrate_hs = Arc::new(AtomicU64::new(0));
    let started = Instant::now();

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

    // Rate reporter (setups/s + shares) + hashrate in the pool's TH/s convention.
    {
        let attempts = Arc::clone(&attempts);
        let shares = Arc::clone(&shares);
        let credited = Arc::clone(&credited);
        let hashrate_hs = Arc::clone(&hashrate_hs);
        tokio::spawn(async move {
            let mut last = 0u64;
            let mut last_cred = 0u64;
            let mut t = Instant::now();
            loop {
                tokio::time::sleep(Duration::from_secs(10)).await;
                let now = attempts.load(Ordering::Relaxed);
                let cred = credited.load(Ordering::Relaxed);
                let dt = t.elapsed().as_secs_f64().max(1e-3);
                // hashrate = Σ(credited difficulty)/window × display-mult — the SAME
                // basis the pool uses, so the miner's number matches the dashboard
                // (and converges cleanly with vardiff instead of spiking).
                let hr = (cred.saturating_sub(last_cred)) as f64 / dt * HASHRATE_DISPLAY_MULT;
                hashrate_hs.store(hr as u64, Ordering::Relaxed);
                tracing::info!(
                    "rate: {:.1} setups/s | total {} setups, {} shares",
                    (now - last) as f64 / dt,
                    now,
                    shares.load(Ordering::Relaxed)
                );
                last = now;
                last_cred = cred;
                t = Instant::now();
            }
        });
    }

    let mut cur_params: Option<MiningParams> = None;
    let mut cur_difficulty: u64 = 524_288;
    let mut grind_gen: Option<GrindGen> = None;
    let mut job_slot: Option<Arc<Mutex<Arc<GrindJob>>>> = None;

    loop {
        match job_rx.recv().await {
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
                let gj = match build_grind_job(&params, &job, cur_difficulty) {
                    Ok(g) => Arc::new(g),
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
                            submit_tx.clone(),
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
        }
    }

    if let Some(g) = grind_gen.take() {
        g.retire();
    }
    Ok(())
}
