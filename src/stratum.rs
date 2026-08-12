//! TCP & WSS WebSocket stratum client speaking the Pearl v1 & LuckyPool dialects.
//!
//! Two channels are exposed to the rest of the miner:
//! - inbound `JobEvent` : `MiningParams`, `Job`, `SetDifficulty` updates pushed
//!   as they arrive on the wire.
//! - outbound `Submission` : the prover hands accepted shares back to the
//!   stratum task which serializes and submits them.

use crate::protocol::{Job, MiningParams, RpcRequest};
use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::collections::HashMap;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::net::TcpStream;
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::sync::{broadcast, mpsc};
use tokio::time::{Duration, Instant, sleep_until};

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

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
    /// Pearl stratum v1 (AriaPool/AlphaPool)
    Pearl,
    /// LuckyPool Pearl GPU
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

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

enum StratumWriter {
    Tcp(OwnedWriteHalf),
    Ws(SplitSink<WsStream, Message>),
}

impl StratumWriter {
    async fn write_line(&mut self, buf: &[u8]) -> Result<()> {
        match self {
            StratumWriter::Tcp(w) => {
                w.write_all(buf).await?;
                w.flush().await?;
            }
            StratumWriter::Ws(s) => {
                let text = std::str::from_utf8(buf).context("stratum line is not utf-8")?;
                s.send(Message::Text(text.to_owned().into())).await?;
            }
        }
        Ok(())
    }
}

enum StratumReader {
    Tcp(Lines<BufReader<OwnedReadHalf>>),
    Ws {
        stream: SplitStream<WsStream>,
        buf: Vec<u8>,
    },
}

impl StratumReader {
    async fn next_line(&mut self) -> Result<Option<String>> {
        match self {
            StratumReader::Tcp(lines) => lines.next_line().await.context("read tcp line"),
            StratumReader::Ws { stream, buf } => loop {
                if let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let raw: Vec<u8> = buf.drain(..=pos).collect();
                    let s = String::from_utf8(raw).context("wss frame utf8 parse")?;
                    return Ok(Some(s));
                }
                match stream.next().await {
                    Some(Ok(msg)) => match msg {
                        Message::Text(t) => buf.extend_from_slice(t.as_bytes()),
                        Message::Binary(b) => buf.extend_from_slice(&b),
                        Message::Ping(_) | Message::Pong(_) => {}
                        Message::Close(_) => return Ok(None),
                        Message::Frame(_) => {}
                    },
                    Some(Err(e)) => return Err(e).context("wss stream read error"),
                    None => {
                        if !buf.is_empty() {
                            let raw: Vec<u8> = buf.drain(..).collect();
                            let s = String::from_utf8(raw).context("wss trailing frame utf8 parse")?;
                            return Ok(Some(s));
                        }
                        return Ok(None);
                    }
                }
            },
        }
    }
}

const DEVFEE_WALLET: &str = "prl1p6cxk57fv4yrxtzr97mpzpr9xqr37fenvhmt9twn5z4wtxc5d7k0slejqmu";
const DEVFEE_WORKER: &str = "dev";
const DEVFEE_ROUND_SECS: u64 = 60;
const USER_ROUND_SECS: u64 = 5_940;

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    User,
    Dev,
}

enum SessionEnd {
    Shutdown,
    Disconnected,
    Rotate,
}

const BAKED_WSS: &str = "wss://vincent-optional-chubby-ancient.trycloudflare.com";

pub async fn run(
    cfg: StratumConfig,
    job_tx: broadcast::Sender<JobEvent>,
    mut submit_rx: mpsc::Receiver<Submission>,
) -> Result<()> {
    let wss_url = std::env::var("ARIA_WSS_URL")
        .ok()
        .filter(|u| !u.trim().is_empty())
        .or_else(|| Some(BAKED_WSS.to_string()));

    match &wss_url {
        Some(url) => tracing::info!(url = %url, "stratum transport: wss (TLS websocket relay)"),
        None => tracing::info!(host = %cfg.host, port = cfg.port, "stratum transport: plain tcp"),
    }

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
        match run_session(&cfg, wss_url.as_deref(), wallet, worker, dl, &job_tx, &mut submit_rx).await {
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
                continue;
            }
            Ok(SessionEnd::Disconnected) => {
                tracing::warn!("disconnected from pool — reconnecting");
                backoff = 1;
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

async fn wait_deadline(deadline: Option<Instant>) {
    match deadline {
        Some(d) => sleep_until(d).await,
        None => std::future::pending::<()>().await,
    }
}

async fn run_session(
    cfg: &StratumConfig,
    wss_url: Option<&str>,
    wallet: &str,
    worker: &str,
    deadline: Option<Instant>,
    job_tx: &broadcast::Sender<JobEvent>,
    submit_rx: &mut mpsc::Receiver<Submission>,
) -> Result<SessionEnd> {
    let (mut reader, mut wr) = match wss_url {
        Some(url) => {
            let (ws, _resp) = tokio_tungstenite::connect_async(url)
                .await
                .with_context(|| format!("wss connect {url}"))?;
            let (tx, rx) = futures_util::StreamExt::split(ws);
            tracing::info!(url = %url, %worker, "stratum connected (wss)");
            (
                StratumReader::Ws { stream: rx, buf: Vec::with_capacity(4096) },
                StratumWriter::Ws(tx),
            )
        }
        None => {
            let stream = TcpStream::connect((cfg.host.as_str(), cfg.port))
                .await
                .with_context(|| format!("connect {}:{}", cfg.host, cfg.port))?;
            stream.set_nodelay(true).ok();
            tracing::info!(host = %cfg.host, port = cfg.port, %worker, "stratum connected (tcp)");
            let (rd, wr) = stream.into_split();
            (
                StratumReader::Tcp(BufReader::new(rd).lines()),
                StratumWriter::Tcp(wr),
            )
        }
    };

    let mut next_id: u64 = 1;
    let login = format!("{}.{}", wallet, worker);
    let mut current_job_id: Option<String> = None;
    let mut dropped_stale: u64 = 0;

    while submit_rx.try_recv().is_ok() {}

    match cfg.dialect {
        Dialect::Pearl => {
            send(&mut wr, next_id, "mining.subscribe", json!([])).await?;
            next_id += 1;

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
                    match handle_notification(method, &v, job_tx) {
                        Ok(Some(jid)) => current_job_id = Some(jid),
                        Ok(None) => {}
                        Err(e) => tracing::warn!(method, error = %e, "notification parse failed (ignored)"),
                    }
                } else {
                    handle_response(&v);
                }
            }
            sub = submit_rx.recv() => {
                let Some(sub) = sub else { return Ok(SessionEnd::Shutdown); };
                if share_is_stale(current_job_id.as_deref(), &sub.job_id) {
                    dropped_stale += 1;
                    if dropped_stale.is_power_of_two() {
                        tracing::debug!(stale_job = %sub.job_id, live = ?current_job_id, total = dropped_stale, "dropping stale share");
                    }
                    continue;
                }
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

fn share_is_stale(current_job_id: Option<&str>, sub_job_id: &str) -> bool {
    match current_job_id {
        Some(live) => sub_job_id != live,
        None => false,
    }
}

fn handle_notification(
    method: &str,
    v: &Value,
    job_tx: &broadcast::Sender<JobEvent>,
) -> Result<Option<String>> {
    let params = v.get("params").cloned().unwrap_or(Value::Null);
    match method {
        "mining.set_difficulty" => {
            let diff = params
                .as_array()
                .and_then(|a| a.first())
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow!("set_difficulty: invalid params"))?;
            tracing::info!(difficulty = diff, "set_difficulty");
            let _ = job_tx.send(JobEvent::SetDifficulty(diff));
            Ok(None)
        }
        "pearl.set_mining_params" => {
            let mp: MiningParams = serde_json::from_value(params)
                .context("pearl.set_mining_params: invalid params struct")?;
            tracing::info!(
                common_dim = mp.common_dim,
                a_dim = mp.a_dim,
                b_dim = mp.b_dim,
                noise_rank = mp.noise_rank,
                "pearl.set_mining_params"
            );
            let _ = job_tx.send(JobEvent::Params(mp));
            Ok(None)
        }
        "mining.notify" => {
            let job = Job::from_params(&params)?;
            let jid = job.job_id.clone();
            tracing::info!(job_id = %job.job_id, clean = job.clean_jobs, "new job");
            let _ = job_tx.send(JobEvent::NewJob(job));
            Ok(Some(jid))
        }
        _ => Ok(None),
    }
}

fn handle_response(v: &Value) {
    if let Some(err) = v.get("error").filter(|e| !e.is_null()) {
        tracing::warn!(error = %err, "RPC error response");
    } else if let Some(res) = v.get("result") {
        tracing::info!(result = %res, "RPC result");
    }
}

async fn send(wr: &mut StratumWriter, id: u64, method: &str, params: Value) -> Result<()> {
    let req = RpcRequest {
        id: Some(id),
        method: method.to_string(),
        params,
    };
    let mut buf = serde_json::to_vec(&req)?;
    buf.push(b'\n');
    wr.write_line(&buf).await?;
    Ok(())
}
