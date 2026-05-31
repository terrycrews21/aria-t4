//! ARIAMiner CPU miner — wires the stratum client to the amortized grind
//! engine. Connects to a Pearl pool, takes live jobs, grinds across N threads at
//! full speed, reports hashrate, and submits share hits.
//!
//!   cargo run --release -- \
//!     --pool pearl.ariabrain.com:3334 --wallet <YOUR_PRL_ADDRESS> --worker my-cpu-01 [--threads N] [--batch 4096]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use ariaminer::cpu_engine::{JobSpec, grind_thread, share_target_from_difficulty};
use ariaminer::protocol::{Job, MiningParams, decode_nbits};
use ariaminer::stratum::{JobEvent, Submission, StratumConfig, run as stratum_run};
use base64::Engine;
use clap::Parser;
use parking_lot::Mutex;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::{broadcast, mpsc};

#[derive(Parser, Debug)]
#[command(name = "ariaminer", version, about = "ARIAMiner — open-source CPU miner for Pearl (PRL), amortized GEMM Int7×Int7 + BLAKE3 grind")]
struct Args {
    /// Pool endpoint host:port (e.g. pearl.ariabrain.com:3334)
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
    /// Amortization batch (rows/cols drawn per setup). 4096 is the sweet spot on
    /// most CPUs: noise gen (the dominant per-setup cost) amortizes over
    /// batch²/128 tiles, so a larger batch helps until it plateaus around 4096.
    #[arg(long, default_value_t = 4096)]
    batch: usize,
}

/// Workers for one job generation. Dropping/stopping retires them.
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

fn spawn_grind(
    job_slot: Arc<Mutex<Arc<JobSpec>>>,
    threads: usize,
    batch: usize,
    attempts: Arc<AtomicU64>,
    hits: Arc<AtomicU64>,
    submit_tx: mpsc::Sender<Submission>,
) -> GrindGen {
    let stop = Arc::new(AtomicBool::new(false));
    let handles = (0..threads)
        .map(|tid| {
            let job_slot = Arc::clone(&job_slot);
            let stop = Arc::clone(&stop);
            let attempts = Arc::clone(&attempts);
            let hits = Arc::clone(&hits);
            let submit_tx = submit_tx.clone();
            thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(
                    0xC0FFEE ^ (tid as u64) ^ (Instant::now().elapsed().as_nanos() as u64),
                );
                grind_thread(&job_slot, batch, batch, &mut rng, &stop, &attempts, |hit| {
                    hits.fetch_add(1, Ordering::Relaxed);
                    if hit.is_block {
                        tracing::warn!(hash = %hex::encode(hit.hash), "🟦🟦 BLOCK! hash < network target");
                    }
                    // Trust-mode pool accepts any base64 proof; ship the hash as a
                    // placeholder PlainProof until the full Merkle serialization lands.
                    let proof_base64 =
                        base64::engine::general_purpose::STANDARD.encode(hit.hash);
                    let _ = submit_tx.try_send(Submission {
                        job_id: hit.job_id,
                        proof_base64,
                    });
                });
            })
        })
        .collect();
    GrindGen { stop, handles }
}

fn build_job(params: &MiningParams, job: &Job, difficulty: u64) -> anyhow::Result<JobSpec> {
    // job_key = blake3(header_template). TODO: official is blake3(header||config).
    let header = hex::decode(job.header_template.trim_start_matches("0x"))?;
    let job_key = *blake3::hash(&header).as_bytes();
    let target = match job.full_target {
        Some(t) => t,
        None => decode_nbits(&job.target_nbits)?,
    };
    Ok(JobSpec {
        job_id: job.job_id.clone(),
        k: params.k as usize,
        r: params.rank as usize,
        h: params.rows_pattern.len().max(1),
        w: params.cols_pattern.len().max(1),
        job_key,
        target,
        share_target: share_target_from_difficulty(difficulty),
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    let threads = args
        .threads
        .unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(1));

    let (host, port) = args
        .pool
        .rsplit_once(':')
        .ok_or_else(|| anyhow::anyhow!("--pool must be host:port"))?;
    let port: u16 = port.parse()?;

    tracing::info!(pool = %args.pool, worker = %args.worker, threads, batch = args.batch, "ariaminer starting");

    let (job_tx, mut job_rx) = broadcast::channel::<JobEvent>(64);
    let (submit_tx, submit_rx) = mpsc::channel::<Submission>(256);

    // Stratum client task.
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
    let hits = Arc::new(AtomicU64::new(0));

    // Hashrate reporter.
    {
        let attempts = Arc::clone(&attempts);
        let hits = Arc::clone(&hits);
        tokio::spawn(async move {
            let mut last = 0u64;
            let mut t = Instant::now();
            loop {
                tokio::time::sleep(Duration::from_secs(10)).await;
                let now = attempts.load(Ordering::Relaxed);
                let dt = t.elapsed().as_secs_f64();
                let rate = (now - last) as f64 / dt;
                tracing::info!(
                    "hashrate: {:.0} attempts/s  | total {} attempts, {} hits",
                    rate,
                    now,
                    hits.load(Ordering::Relaxed)
                );
                last = now;
                t = Instant::now();
            }
        });
    }

    // Coordinator: rebuild grind workers whenever params+job define a new JobSpec.
    let mut cur_params: Option<MiningParams> = None;
    let mut cur_difficulty: u64 = 524288;
    let mut grind_gen: Option<GrindGen> = None;
    // Hot-swappable current job: workers spawn once and re-read this each setup,
    // so the pool's ~0.3s job cadence never restarts the grind.
    let mut job_slot: Option<Arc<Mutex<Arc<JobSpec>>>> = None;

    loop {
        match job_rx.recv().await {
            Ok(JobEvent::Params(p)) => {
                tracing::info!(m = p.m, n = p.n, k = p.k, rank = p.rank, h = p.rows_pattern.len(), w = p.cols_pattern.len(), "mining params");
                cur_params = Some(p);
            }
            Ok(JobEvent::SetDifficulty(d)) => {
                tracing::info!(difficulty = d, "difficulty");
                cur_difficulty = d;
                // Apply immediately to the running job. Vardiff retunes between
                // jobs (notify only on a new tip); without refreshing share_target
                // here the miner keeps the stale, easier target and the pool
                // rejects every share as "below pool target".
                if let Some(slot) = &job_slot {
                    let mut guard = slot.lock();
                    let mut spec = (**guard).clone();
                    spec.share_target = share_target_from_difficulty(d);
                    *guard = Arc::new(spec);
                }
            }
            Ok(JobEvent::NewJob(job)) => {
                let Some(params) = cur_params.clone() else {
                    tracing::warn!("job before params — waiting for set_mining_params");
                    continue;
                };
                let spec = match build_job(&params, &job, cur_difficulty) {
                    Ok(s) => Arc::new(s),
                    Err(e) => {
                        tracing::error!(error = %e, "build job failed");
                        continue;
                    }
                };
                match &job_slot {
                    // Already grinding: hot-swap the job, no thread restart.
                    Some(slot) => {
                        *slot.lock() = spec;
                    }
                    // First job: spawn the worker pool once.
                    None => {
                        tracing::info!(job_id = %job.job_id, h = spec.h, w = spec.w, k = spec.k, "→ starting grind");
                        let slot = Arc::new(Mutex::new(spec));
                        job_slot = Some(Arc::clone(&slot));
                        grind_gen = Some(spawn_grind(
                            slot,
                            threads,
                            args.batch,
                            Arc::clone(&attempts),
                            Arc::clone(&hits),
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
