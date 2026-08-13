//! TEST CRÉDIT 2×64 — CPU, chemin prouvé `try_mine_one_bounded` + config DICTÉE
//! par la pool (rows=[0,32], cols=[0..63]). But : prouver que le tile 2×64 est
//! CRÉDITÉ sur le dashboard AlphaPool (≠ result:true). Si oui → on accélère sur GPU.
//! `cargo build --release --bin alphapool_cpu`  (PAS de feature gpu)
use ariaminer::official_grind::{try_mine_one_bounded, Workspace};
use ariaminer::official_proof::encode_base64;
use ariaminer::stratum_to_official::{header_from_template, share_bound_le};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern};

const POOL: &str = "eu1.alphapool.tech:5566";
const WALLET: &str = "prl1p6cxk57fv4yrxtzr97mpzpr9xqr37fenvhmt9twn5z4wtxc5d7k0slejqmu";
const WORKER: &str = "TESTCPU";
const M: usize = 1024;
const N: usize = 1024;
const K: usize = 4096;
const MINERS: usize = 12;

/// Config EXACTE dictée par AlphaPool dans pearl.set_mining_params (tile 2×64).
fn alphapool_config() -> MiningConfiguration {
    let cols: Vec<u32> = (0..64).collect();
    MiningConfiguration {
        common_dim: K as u32,
        rank: 128,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 32]).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&cols).unwrap(),
        moe: None,
    }
}

#[derive(Clone)]
struct JobData {
    job_id: String,
    header: IncompleteBlockHeader,
}

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
    println!("▶ TEST CRÉDIT 2×64 (CPU) — config pool [0,32]×[0..63], worker {WORKER}");
    let job: Arc<Mutex<Option<JobData>>> = Arc::new(Mutex::new(None));
    let difficulty = Arc::new(AtomicU64::new(524288));
    let writer: SharedWriter = Arc::new(Mutex::new(None));
    let accepted = Arc::new(AtomicU64::new(0));
    let rejected = Arc::new(AtomicU64::new(0));
    let config = alphapool_config();

    // === MINEURS CPU (2×64) ===
    for t in 0..MINERS {
        let (job, writer, difficulty, config) = (job.clone(), writer.clone(), difficulty.clone(), config.clone());
        std::thread::spawn(move || {
            let mut ws = Workspace::new();
            let mut rng = StdRng::seed_from_u64(0xA1FA_0000 + t as u64);
            loop {
                let j = { job.lock().unwrap().clone() };
                let Some(j) = j else { std::thread::sleep(Duration::from_millis(50)); continue };
                let bound = share_bound_le(difficulty.load(Ordering::Relaxed), &config);
                if let Some(proof) = try_mine_one_bounded(&mut ws, &mut rng, M, N, K, &j.header, &config, &bound) {
                    if let Ok(b64) = encode_base64(&proof) {
                        send(&writer, &format!(
                            r#"{{"id":99,"method":"mining.submit","params":["{WALLET}.{WORKER}","{}","{b64}"]}}"#, j.job_id));
                    }
                }
            }
        });
    }

    // === Boucle connexion + reconnexion ===
    loop {
        match TcpStream::connect(POOL) {
            Ok(stream) => {
                println!("● connecté");
                *writer.lock().unwrap() = Some(stream.try_clone()?);
                *job.lock().unwrap() = None;
                run_session(stream, &writer, &job, &difficulty, &accepted, &rejected);
                println!("○ déconnexion");
            }
            Err(e) => eprintln!("connexion échouée: {e}"),
        }
        *writer.lock().unwrap() = None;
        *job.lock().unwrap() = None;
        std::thread::sleep(Duration::from_secs(5));
    }
}

fn run_session(
    stream: TcpStream,
    writer: &SharedWriter,
    job: &Arc<Mutex<Option<JobData>>>,
    difficulty: &Arc<AtomicU64>,
    accepted: &Arc<AtomicU64>,
    rejected: &Arc<AtomicU64>,
) {
    let mut rd = BufReader::new(stream);
    let mut id = 1u64;
    let mut hs_done = false;
    let mut line = String::new();
    loop {
        line.clear();
        match rd.read_line(&mut line) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
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
            "mining.notify" => {
                let p = &msg["params"];
                let job_id = p[0].as_str().unwrap_or("").to_string();
                let ht = p[2].as_str().unwrap_or("");
                if let Ok(header) = header_from_template(ht) {
                    *job.lock().unwrap() = Some(JobData { job_id, header });
                }
            }
            _ => {
                if msg.get("result").and_then(|r| r.as_bool()) == Some(true)
                    && msg.get("id").and_then(|i| i.as_u64()) == Some(99) {
                    let a = accepted.fetch_add(1, Ordering::Relaxed) + 1;
                    println!("  ✅ submit accepté #{a} (result:true) — VÉRIFIER LE CRÉDIT DASHBOARD");
                }
                if let Some(e) = msg.get("error") {
                    if !e.is_null() {
                        let r = rejected.fetch_add(1, Ordering::Relaxed) + 1;
                        println!("  ❌ rejet ({r}): {e}");
                    }
                }
            }
        }
        if !hs_done && msg.get("result").and_then(|r| r.as_bool()) == Some(true)
            && msg.get("id").and_then(|i| i.as_u64()) == Some(1) {
            hs_done = true;
            send(writer, &format!(r#"{{"id":{id},"method":"mining.configure","params":[["pearl/v1"],{{}}]}}"#)); id += 1;
            send(writer, &format!(r#"{{"id":{id},"method":"mining.subscribe","params":["aria-gpu-miner/1.0"]}}"#)); id += 1;
            send(writer, &format!(r#"{{"id":{id},"method":"mining.authorize","params":["{WALLET}.{WORKER}","x;d=524288"]}}"#)); id += 1;
        }
    }
}
