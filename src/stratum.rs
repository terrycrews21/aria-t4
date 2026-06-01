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

pub struct StratumConfig {
    pub host: String,
    pub port: u16,
    pub wallet: String,
    pub worker: String,
    pub password: String,
}

/// Stratum client with **automatic reconnection**. Runs forever: if the pool
/// drops the connection (restart, crash, network blip) the miner waits with an
/// exponential backoff (1s → 30s cap) and reconnects on its own — no manual
/// restart needed. Grind threads keep running and pick the fresh job up as soon
/// as the pool is back. Only returns when the local submit channel closes
/// (process shutdown).
pub async fn run(
    cfg: StratumConfig,
    job_tx: broadcast::Sender<JobEvent>,
    mut submit_rx: mpsc::Receiver<Submission>,
) -> Result<()> {
    let mut backoff = 1u64;
    loop {
        match run_session(&cfg, &job_tx, &mut submit_rx).await {
            Ok(true) => return Ok(()), // submit channel closed → real shutdown
            Ok(false) => {
                tracing::warn!("disconnected from pool — reconnecting");
                backoff = 1; // we were connected; reset the backoff
            }
            Err(e) => {
                tracing::warn!(error = %e, "stratum connection failed — retrying");
            }
        }
        tracing::info!(secs = backoff, "reconnecting in {backoff}s…");
        tokio::time::sleep(std::time::Duration::from_secs(backoff)).await;
        backoff = (backoff * 2).min(30);
    }
}

/// One connection lifecycle. `Ok(true)` = clean shutdown (submit channel closed),
/// `Ok(false)` = pool dropped us (caller should reconnect), `Err` = connect or
/// I/O error (caller should retry).
async fn run_session(
    cfg: &StratumConfig,
    job_tx: &broadcast::Sender<JobEvent>,
    submit_rx: &mut mpsc::Receiver<Submission>,
) -> Result<bool> {
    let stream = TcpStream::connect((cfg.host.as_str(), cfg.port))
        .await
        .with_context(|| format!("connect {}:{}", cfg.host, cfg.port))?;
    stream.set_nodelay(true).ok();
    tracing::info!(host = %cfg.host, port = cfg.port, "stratum connected");

    let (rd, mut wr) = stream.into_split();
    let mut reader = BufReader::new(rd).lines();
    let mut next_id: u64 = 1;
    let login = format!("{}.{}", cfg.wallet, cfg.worker);

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

    loop {
        tokio::select! {
            line = reader.next_line() => {
                let line = match line.context("read stratum line")? {
                    Some(l) => l,
                    None => {
                        tracing::warn!("pool closed connection");
                        return Ok(false);
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
                let Some(sub) = sub else { return Ok(true); };
                send(
                    &mut wr,
                    next_id,
                    "mining.submit",
                    json!([login, sub.job_id, sub.proof_base64]),
                ).await?;
                next_id += 1;
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
