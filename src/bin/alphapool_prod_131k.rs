//! Aria GPU miner v0.4.2 — AlphaPool PROD, forme 131072 (comme alpha), preuve FULL GPU.
//! `cargo build --release --features gpu --bin alphapool_prod_131k`
//!
//! Différence avec v0.4.1 (alphapool_prod, 8192²) : on mine à m=n=131072. Un setup =
//! grille 1024×1024 CTAs (~680 ms) qui trouve des CENTAINES de hits partageant la MÊME
//! matrice A/B. On bâtit donc les 2 arbres Merkle UNE fois par setup puis on gather chaque
//! hit (modèle `fill_proof_kernel` d'alpha) → ~0.6 ms/preuve effectif, plus de drop.
use ariaminer::gpu_ffi::{Hit, ProofGpuCtx, ResidentCtx};
use ariaminer::official_grind::{alphapool_config_2x64, build_proofs_from_setup_gpu, compute_job_key_pub};
use ariaminer::official_proof::encode_base64;
use ariaminer::stratum_to_official::{header_from_template, share_bound_le};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use zk_pow::api::proof::{IncompleteBlockHeader, MiningConfiguration};

const POOL: &str = "eu1.alphapool.tech:5566";
const WALLET: &str = "prl1p6cxk57fv4yrxtzr97mpzpr9xqr37fenvhmt9twn5z4wtxc5d7k0slejqmu";
const WORKER: &str = "PC3";
const M: usize = 131072;
const N: usize = 131072;
const K: usize = 4096;
const MAX_HITS: usize = 256;          // ~134M tuiles / diff 524288 ≈ 256 hits/setup
const NUM_BUILDERS: usize = 2;        // 1 setup (~680ms) >> build de ses ~256 preuves (~180ms)

#[derive(Clone)]
struct JobData { job_id: String, header: IncompleteBlockHeader, config: MiningConfiguration }

/// Un SETUP entier à emballer : tous ses hits partagent la même matrice (arbre une fois).
struct ProofJob { seed: u64, hits: Vec<Hit>, job_key: [u8; 32], rank: usize, job_id: String }

type SharedWriter = Arc<Mutex<Option<TcpStream>>>;

fn lz_bits(h: &[u8]) -> u32 {
    let mut n = 0;
    for &b in h { if b == 0 { n += 8; } else { return n + b.leading_zeros(); } }
    n
}
fn challenge_hash(seed: &[u8; 32], nonce: u64) -> [u8; 32] {
    let mut m = Vec::with_capacity(40);
    m.extend_from_slice(seed); m.extend_from_slice(&nonce.to_le_bytes());
    *blake3::hash(&m).as_bytes()
}
fn solve_challenge(seed: [u8; 32], difficulty: u32) -> u64 {
    let found = Arc::new(AtomicBool::new(false));
    let answer = Arc::new(AtomicU64::new(0));
    let nt = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let mut hs = vec![];
    for t in 0..nt {
        let (f, a) = (found.clone(), answer.clone());
        hs.push(std::thread::spawn(move || {
            let mut nonce = t as u64;
            while !f.load(Ordering::Relaxed) {
                if lz_bits(&challenge_hash(&seed, nonce)) >= difficulty {
                    a.store(nonce, Ordering::Relaxed); f.store(true, Ordering::Relaxed); return;
                }
                nonce += nt as u64;
            }
        }));
    }
    for h in hs { let _ = h.join(); }
    answer.load(Ordering::Relaxed)
}
fn send(w: &SharedWriter, s: &str) {
    if let Some(g) = w.lock().unwrap().as_mut() {
        let _ = g.write_all(s.as_bytes()); let _ = g.write_all(b"\n"); let _ = g.flush();
    }
}

fn main() -> std::io::Result<()> {
    println!("▶ AlphaPool PROD — Aria GPU miner v0.4.2 (forme 131072, preuve full GPU), worker {WORKER}");
    let job: Arc<Mutex<Option<JobData>>> = Arc::new(Mutex::new(None));
    let difficulty = Arc::new(AtomicU64::new(524288));
    let params_k = Arc::new(AtomicU64::new(K as u64));
    let writer: SharedWriter = Arc::new(Mutex::new(None));
    let accepted = Arc::new(AtomicU64::new(0));
    let rejected = Arc::new(AtomicU64::new(0));
    let dropped = Arc::new(AtomicU64::new(0));

    // Canal setups → builders (borné). Un élément = tous les hits d'un setup.
    let (tx, rx) = sync_channel::<ProofJob>(4);
    let rx = Arc::new(Mutex::new(rx));

    for _ in 0..NUM_BUILDERS {
        let (rx, writer): (Arc<Mutex<Receiver<ProofJob>>>, SharedWriter) = (rx.clone(), writer.clone());
        std::thread::spawn(move || {
            // 2 ctx de preuve résidents (A et B), arbre bâti 1× par setup puis gather/hit.
            let ctx_a = ProofGpuCtx::new(M, K, 256);
            let ctx_b = ProofGpuCtx::new(N, K, 256);
            loop {
                let pj = { rx.lock().unwrap().recv() };
                let Ok(pj) = pj else { return };
                let proofs = build_proofs_from_setup_gpu(&ctx_a, &ctx_b, pj.seed, &pj.hits, &pj.job_key, M, N, K, pj.rank);
                for p in &proofs {
                    if let Ok(b64) = encode_base64(p) {
                        send(&writer, &format!(
                            r#"{{"id":99,"method":"mining.submit","params":["{WALLET}.{WORKER}","{}","{b64}"]}}"#, pj.job_id));
                    }
                }
            }
        });
    }

    // === Thread GPU : grind en continu (un setup géant à la fois) ===
    {
        let (job, difficulty, dropped, accepted) = (job.clone(), difficulty.clone(), dropped.clone(), accepted.clone());
        std::thread::spawn(move || {
            let ctx = ResidentCtx::new_2x64(M, N, K, MAX_HITS);
            let mut seed: u64 = 0x5080_A2FA_0420;
            let mut last_log = Instant::now();
            let mut setups = 0u64;
            loop {
                let j = { job.lock().unwrap().clone() };
                let Some(j) = j else { std::thread::sleep(Duration::from_millis(50)); continue };
                seed = seed.wrapping_add(0x9E3779B97F4A7C15);
                let job_key = compute_job_key_pub(&j.header, &j.config);
                let bound = share_bound_le(difficulty.load(Ordering::Relaxed));
                let (_found, hits) = ctx.grind(seed, &job_key, &bound);
                setups += 1;
                let valid: Vec<Hit> = hits.into_iter().filter(|h| h.rows.len() == 2 && h.cols.len() == 64).collect();
                if !valid.is_empty() {
                    let pj = ProofJob { seed, hits: valid, job_key, rank: j.config.rank as usize, job_id: j.job_id.clone() };
                    match tx.try_send(pj) {
                        Ok(()) => {}
                        Err(TrySendError::Full(_)) => { dropped.fetch_add(1, Ordering::Relaxed); }
                        Err(TrySendError::Disconnected(_)) => return,
                    }
                }
                if last_log.elapsed() >= Duration::from_secs(20) {
                    let secs = last_log.elapsed().as_secs_f64();
                    let th = setups as f64 / secs * (M * N * K) as f64 / 1e12;
                    println!("  ⛏  GPU {:.2} setups/s · ~{:.0} TH/s · acceptées {} · setups droppés {} · diff {}",
                        setups as f64 / secs, th, accepted.load(Ordering::Relaxed),
                        dropped.load(Ordering::Relaxed), difficulty.load(Ordering::Relaxed));
                    setups = 0; last_log = Instant::now();
                }
            }
        });
    }

    loop {
        match TcpStream::connect(POOL) {
            Ok(stream) => {
                println!("● connecté à {POOL}");
                *writer.lock().unwrap() = Some(stream.try_clone()?);
                *job.lock().unwrap() = None;
                run_session(stream, &writer, &job, &difficulty, &params_k, &accepted, &rejected);
                println!("○ session terminée (déconnexion)");
            }
            Err(e) => eprintln!("connexion échouée: {e}"),
        }
        *writer.lock().unwrap() = None;
        *job.lock().unwrap() = None;
        std::thread::sleep(Duration::from_secs(5));
        println!("↻ reconnexion…");
    }
}

#[allow(clippy::too_many_arguments)]
fn run_session(
    stream: TcpStream, writer: &SharedWriter, job: &Arc<Mutex<Option<JobData>>>,
    difficulty: &Arc<AtomicU64>, params_k: &Arc<AtomicU64>,
    accepted: &Arc<AtomicU64>, rejected: &Arc<AtomicU64>,
) {
    let mut rd = BufReader::new(stream);
    let mut id = 1u64;
    let mut hs_done = false;
    let mut line = String::new();
    loop {
        line.clear();
        match rd.read_line(&mut line) { Ok(0) | Err(_) => return, Ok(_) => {} }
        let msg: serde_json::Value = match serde_json::from_str(line.trim()) { Ok(v) => v, Err(_) => continue };
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        match method {
            "pearl.challenge" => {
                let p = &msg["params"];
                let seed_hex = p["seed"].as_str().unwrap_or("");
                let diff = p["difficulty"].as_u64().unwrap_or(32) as u32;
                let mut seed = [0u8; 32]; let _ = hex::decode_to_slice(seed_hex, &mut seed);
                let nonce = solve_challenge(seed, diff);
                send(writer, &format!(r#"{{"id":{id},"method":"pearl.challenge_response","params":{{"seed":"{seed_hex}","nonce":"{nonce:016x}"}}}}"#)); id += 1;
            }
            "mining.set_difficulty" => {
                if let Some(d) = msg["params"][0].as_u64() { difficulty.store(d, Ordering::Relaxed); println!("  ⚙ diff → {d}"); }
            }
            "pearl.set_mining_params" => {
                if let Some(k) = msg["params"][0]["k"].as_u64() { params_k.store(k, Ordering::Relaxed); }
            }
            "mining.notify" => {
                let p = &msg["params"];
                let job_id = p[0].as_str().unwrap_or("").to_string();
                let ht = p[2].as_str().unwrap_or("");
                if let Ok(header) = header_from_template(ht) {
                    let k = params_k.load(Ordering::Relaxed) as u32;
                    let config = alphapool_config_2x64(k);
                    *job.lock().unwrap() = Some(JobData { job_id, header, config });
                }
            }
            _ => {
                if msg.get("result").and_then(|r| r.as_bool()) == Some(true)
                    && msg.get("id").and_then(|i| i.as_u64()) == Some(99) {
                    let a = accepted.fetch_add(1, Ordering::Relaxed) + 1;
                    if a % 50 == 0 { println!("  ✅ {a} shares acceptées (cumul)"); }
                }
                if let Some(e) = msg.get("error") {
                    if !e.is_null() { let r = rejected.fetch_add(1, Ordering::Relaxed) + 1; println!("  ❌ rejet ({r}): {e}"); }
                }
            }
        }
        if !hs_done && msg.get("result").and_then(|r| r.as_bool()) == Some(true)
            && msg.get("id").and_then(|i| i.as_u64()) == Some(1) {
            hs_done = true;
            send(writer, &format!(r#"{{"id":{id},"method":"mining.configure","params":[["pearl/v1"],{{}}]}}"#)); id += 1;
            send(writer, &format!(r#"{{"id":{id},"method":"mining.subscribe","params":["aria-gpu-miner/0.4.2"]}}"#)); id += 1;
            send(writer, &format!(r#"{{"id":{id},"method":"mining.authorize","params":["{WALLET}.{WORKER}","x;d=524288"]}}"#)); id += 1;
        }
    }
}
