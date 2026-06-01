//! ARIAMiner — open-source, 0% dev-fee CPU miner for Pearl (PRL).
//!
//! Grinds Pearl's canonical GEMM Int7×Int7 proof-of-work and submits real
//! `PlainProof` shares — every accepted share is a genuine proof that passes the
//! node-side `verify_plain_proof`, so a pool can assemble the block and
//! `submitblock` on a hit. Optimized with AVX-512 VNNI / AVX-VNNI (scalar
//! fallback on older CPUs).
//!
//!   ariaminer --pool <host:port> --wallet prl1... --worker my-rig [--threads N]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use ariaminer::official_grind::try_mine_one_bounded;
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

fn spawn_grind(
    job_slot: Arc<Mutex<Arc<GrindJob>>>,
    threads: usize,
    attempts: Arc<AtomicU64>,
    shares: Arc<AtomicU64>,
    submit_tx: mpsc::Sender<Submission>,
) -> GrindGen {
    let stop = Arc::new(AtomicBool::new(false));
    let handles = (0..threads)
        .map(|tid| {
            let job_slot = Arc::clone(&job_slot);
            let stop = Arc::clone(&stop);
            let attempts = Arc::clone(&attempts);
            let shares = Arc::clone(&shares);
            let submit_tx = submit_tx.clone();
            thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(
                    0xA51A_0000 ^ (tid as u64) ^ (Instant::now().elapsed().as_nanos() as u64),
                );
                while !stop.load(Ordering::Relaxed) {
                    // Adopt the latest job at each setup boundary.
                    let job = job_slot.lock().clone();
                    let oj = &job.official;
                    // One faithful attempt: draw matrices, noise, sweep every
                    // PeriodicPattern tile against the share bound.
                    let hit = try_mine_one_bounded(
                        &mut rng, oj.m, oj.n, oj.k, &oj.header, &oj.config, &oj.bound_le,
                    );
                    attempts.fetch_add(1, Ordering::Relaxed);
                    if let Some(proof) = hit {
                        match encode_base64(&proof) {
                            Ok(proof_base64) => {
                                shares.fetch_add(1, Ordering::Relaxed);
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
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

    // Rate reporter (setups/s + shares).
    {
        let attempts = Arc::clone(&attempts);
        let shares = Arc::clone(&shares);
        tokio::spawn(async move {
            let mut last = 0u64;
            let mut t = Instant::now();
            loop {
                tokio::time::sleep(Duration::from_secs(10)).await;
                let now = attempts.load(Ordering::Relaxed);
                let dt = t.elapsed().as_secs_f64();
                tracing::info!(
                    "rate: {:.1} setups/s | total {} setups, {} shares",
                    (now - last) as f64 / dt,
                    now,
                    shares.load(Ordering::Relaxed)
                );
                last = now;
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
                        tracing::info!(job_id = %job.job_id, "→ starting official grind");
                        let slot = Arc::new(Mutex::new(gj));
                        job_slot = Some(Arc::clone(&slot));
                        grind_gen = Some(spawn_grind(
                            slot,
                            threads,
                            Arc::clone(&attempts),
                            Arc::clone(&shares),
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
