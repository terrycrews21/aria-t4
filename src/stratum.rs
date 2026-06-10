//! TCP stratum client speaking the Pearl v1 dialect.
//!
//! Two channels are exposed to the rest of the miner:
//! - inbound `JobEvent` : `MiningParams`, `Job`, `SetDifficulty` updates pushed
//!   as they arrive on the wire.
//! - outbound `Submission` : the prover hands accepted shares back to the
//!   stratum task which serializes and submits them.

use crate::protocol::{Job, MiningParams, RpcRequest};
use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::net::tcp::OwnedWriteHalf;
use tokio::sync::{broadcast, mpsc};
use tokio::time::{Duration, Instant, sleep_until};

#[derive(Debug, Clone)]
pub enum JobEvent {
    SetDifficulty(u64),
    Params(MiningParams),
    NewJob(Job),
}

#[derive(Debug)]
pub struct Submission {
    pub job_id: String,
    pub proof_base64: String,
}

/// Wire dialect spoken by the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// Pearl stratum v1 (AriaPool/AlphaPool) : subscribe + authorize/submit en
    /// params-tableau, vardiff via `mining.set_difficulty`.
    Pearl,
    /// LuckyPool Pearl GPU (`pearl-eu2.luckypool.io:3360`) : pas de subscribe,
    /// params-OBJET (`authorize {wallet, worker, agent}`, `submit {job_id,
    /// plain_proof}`), pas de set_difficulty (le target voyage dans le job).
    LuckyPool,
}

pub struct StratumConfig {
    pub host: String,
    pub port: u16,
    pub wallet: String,
    pub worker: String,
    pub password: String,
    pub dialect: Dialect,
}

/// **Disclosed 1% dev-fee.** For ~1% of mining time the miner authorizes with
/// the dev wallet (short rounds, applied via reconnect since the pool credits
/// the *authorized* wallet). It is announced in the logs at startup — never
/// hidden, unlike alpha-miner. If the operator already mines to the dev wallet,
/// the fee is a no-op (you would only be paying yourself).
const DEVFEE_WALLET: &str = "prl1p6cxk57fv4yrxtzr97mpzpr9xqr37fenvhmt9twn5z4wtxc5d7k0slejqmu";
const DEVFEE_WORKER: &str = "dev";
const DEVFEE_ROUND_SECS: u64 = 60; // one dev-fee round
const USER_ROUND_SECS: u64 = 5_940; // 99 × 60s → 60s dev per 100 min = exactly 1.0%

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    User,
    Dev,
}

/// Why a connection lifecycle ended.
enum SessionEnd {
    /// Local submit channel closed → real process shutdown.
    Shutdown,
    /// Pool dropped us → reconnect, same phase.
    Disconnected,
    /// Dev-fee phase deadline reached → reconnect with the other wallet.
    Rotate,
}

/// Stratum client with **automatic reconnection** + disclosed 1% dev-fee. Runs
/// forever: if the pool drops the connection the miner waits with an exponential
/// backoff (1s → 30s cap) and reconnects on its own. Grind threads keep running
/// and pick the fresh job up as soon as the pool is back. Every ~100 min it
/// spends one 60s round authorized to the dev wallet (the 1% fee), then returns
/// to the operator's wallet. Only returns when the submit channel closes.
pub async fn run(
    cfg: StratumConfig,
    job_tx: broadcast::Sender<JobEvent>,
    mut submit_rx: mpsc::Receiver<Submission>,
) -> Result<()> {
    // Fee is active unless the operator already mines to the dev wallet.
    let devfee_on = cfg.wallet != DEVFEE_WALLET;
    if devfee_on {
        tracing::info!(
            "💎 dev-fee 1% (announced): one {DEVFEE_ROUND_SECS}s round every {}min mines to the dev wallet — thanks for supporting ARIAMiner",
            (USER_ROUND_SECS + DEVFEE_ROUND_SECS) / 60
        );
    }
    let mut phase = Phase::User;
    let mut deadline = Instant::now() + Duration::from_secs(USER_ROUND_SECS);
    let mut backoff = 1u64;
    loop {
        let (wallet, worker) = match phase {
            Phase::User => (cfg.wallet.as_str(), cfg.worker.as_str()),
            Phase::Dev => (DEVFEE_WALLET, DEVFEE_WORKER),
        };
        let dl = if devfee_on { Some(deadline) } else { None };
        match run_session(&cfg, wallet, worker, dl, &job_tx, &mut submit_rx).await {
            Ok(SessionEnd::Shutdown) => return Ok(()),
            Ok(SessionEnd::Rotate) => {
                phase = match phase {
                    Phase::User => Phase::Dev,
                    Phase::Dev => Phase::User,
                };
                let secs = match phase {
                    Phase::User => USER_ROUND_SECS,
                    Phase::Dev => DEVFEE_ROUND_SECS,
                };
                deadline = Instant::now() + Duration::from_secs(secs);
                if phase == Phase::Dev {
                    tracing::info!("💎 dev-fee round ({DEVFEE_ROUND_SECS}s)");
                }
                backoff = 1;
                continue; // reconnect immediately with the new wallet
            }
            Ok(SessionEnd::Disconnected) => {
                tracing::warn!("disconnected from pool — reconnecting");
                backoff = 1; // we were connected; reset the backoff
            }
            Err(e) => {
                tracing::warn!(error = %e, "stratum connection failed — retrying");
            }
        }
        tracing::info!(secs = backoff, "reconnecting in {backoff}s…");
        tokio::time::sleep(Duration::from_secs(backoff)).await;
        backoff = (backoff * 2).min(30);
    }
}

/// Completes at `deadline` if set, else never — drives the dev-fee rotation
/// from inside the session `select!` without disturbing mining.
async fn wait_deadline(deadline: Option<Instant>) {
    match deadline {
        Some(d) => sleep_until(d).await,
        None => std::future::pending::<()>().await,
    }
}

/// One connection lifecycle. `Ok(true)` = clean shutdown (submit channel closed),
/// `Ok(false)` = pool dropped us (caller should reconnect), `Err` = connect or
/// I/O error (caller should retry).
async fn run_session(
    cfg: &StratumConfig,
    wallet: &str,
    worker: &str,
    deadline: Option<Instant>,
    job_tx: &broadcast::Sender<JobEvent>,
    submit_rx: &mut mpsc::Receiver<Submission>,
) -> Result<SessionEnd> {
    let stream = TcpStream::connect((cfg.host.as_str(), cfg.port))
        .await
        .with_context(|| format!("connect {}:{}", cfg.host, cfg.port))?;
    stream.set_nodelay(true).ok();
    tracing::info!(host = %cfg.host, port = cfg.port, %worker, "stratum connected");

    let (rd, mut wr) = stream.into_split();
    let mut reader = BufReader::new(rd).lines();
    let mut next_id: u64 = 1;
    let login = format!("{}.{}", wallet, worker);

    match cfg.dialect {
        Dialect::Pearl => {
            // 1. mining.subscribe — params empty. The pool detects the Pearl wire and
            //    pushes `pearl.set_mining_params` automatically; an explicit
            //    `mining.subscribe.pearl` call is rejected as "unknown method".
            send(&mut wr, next_id, "mining.subscribe", json!([])).await?;
            next_id += 1;

            // 2. mining.authorize.
            send(
                &mut wr,
                next_id,
                "mining.authorize",
                json!([login, cfg.password]),
            )
            .await?;
            next_id += 1;
        }
        Dialect::LuckyPool => {
            // LuckyPool: no subscribe; a single object-based authorize. The pool
            // answers with `mining.notify` directly (no set_mining_params).
            send(
                &mut wr,
                next_id,
                "mining.authorize",
                json!({"wallet": login, "worker": worker, "agent": concat!("ariaminer/", env!("CARGO_PKG_VERSION"))}),
            )
            .await?;
            next_id += 1;
        }
    }

    loop {
        tokio::select! {
            line = reader.next_line() => {
                let line = match line.context("read stratum line")? {
                    Some(l) => l,
                    None => {
                        tracing::warn!("pool closed connection");
                        return Ok(SessionEnd::Disconnected);
                    }
                };
                if line.trim().is_empty() { continue; }
                let v: Value = serde_json::from_str(&line)
                    .with_context(|| format!("invalid JSON-RPC: {line}"))?;
                if let Some(method) = v.get("method").and_then(Value::as_str) {
                    if let Err(e) = handle_notification(method, &v, job_tx) {
                        tracing::warn!(method, error = %e, "notification parse failed (ignored)");
                    }
                } else {
                    handle_response(&v);
                }
            }
            sub = submit_rx.recv() => {
                let Some(sub) = sub else { return Ok(SessionEnd::Shutdown); };
                let params = match cfg.dialect {
                    Dialect::Pearl => json!([login, sub.job_id, sub.proof_base64]),
                    Dialect::LuckyPool => json!({"job_id": sub.job_id, "plain_proof": sub.proof_base64}),
                };
                send(&mut wr, next_id, "mining.submit", params).await?;
                next_id += 1;
            }
            _ = wait_deadline(deadline) => {
                return Ok(SessionEnd::Rotate);
            }
        }
    }
}

fn handle_notification(
    method: &str,
    msg: &Value,
    job_tx: &broadcast::Sender<JobEvent>,
) -> Result<()> {
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    match method {
        "mining.set_difficulty" => {
            let diff = params
                .as_array()
                .and_then(|a| a.first())
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow!("set_difficulty: invalid params"))?;
            tracing::info!(difficulty = diff, "set_difficulty");
            let _ = job_tx.send(JobEvent::SetDifficulty(diff));
        }
        "pearl.set_mining_params" => {
            let p = params
                .as_array()
                .and_then(|a| a.first())
                .ok_or_else(|| anyhow!("set_mining_params: empty array"))?;
            let mp: MiningParams = serde_json::from_value(p.clone())
                .context("decode MiningParams")?;
            tracing::info!(
                m = mp.m, n = mp.n, k = mp.k, rank = mp.rank,
                rows = mp.rows_pattern.len(), cols = mp.cols_pattern.len(),
                mma = %mp.mma_type,
                "mining_params"
            );
            let _ = job_tx.send(JobEvent::Params(mp));
        }
        "mining.notify" => {
            let job = Job::from_params(&params)?;
            tracing::info!(job_id = %job.job_id, clean = job.clean_jobs, "new job");
            let _ = job_tx.send(JobEvent::NewJob(job));
        }
        "pearl.challenge" => {
            // pearl.challenge is an alternative wire payload some pools send instead
            // of `mining.notify`. Same shape: forward as a NewJob if parseable.
            if let Ok(job) = Job::from_params(&params) {
                let _ = job_tx.send(JobEvent::NewJob(job));
            }
        }
        other => tracing::debug!(method = other, "stratum notification ignored"),
    }
    Ok(())
}

fn handle_response(msg: &Value) {
    let id = msg.get("id").and_then(Value::as_u64);
    let result = msg.get("result");
    let err = msg.get("error");
    if err.is_some() && !err.unwrap().is_null() {
        tracing::warn!(id = ?id, error = %err.unwrap(), "stratum error response");
    } else {
        tracing::debug!(id = ?id, result = ?result, "stratum response");
    }
}

async fn send(
    wr: &mut OwnedWriteHalf,
    id: u64,
    method: &str,
    params: Value,
) -> Result<()> {
    let req = RpcRequest {
        id: Some(id),
        method: method.to_string(),
        params,
    };
    let mut buf = serde_json::to_vec(&req)?;
    buf.push(b'\n');
    wr.write_all(&buf).await?;
    wr.flush().await?;
    Ok(())
}

// Reasonable defaults the Pearl wire uses (kept here so the rest of the crate
// doesn't redefine them).
pub use crate::protocol::RpcRequest as _StratumRpcRequest;
pub use crate::protocol::RpcResponse as _StratumRpcResponse;
