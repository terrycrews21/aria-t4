//! Miner AlphaPool — test décisif : notre miner GPU 8×16 est-il accepté par AlphaPool ?
//! `cargo run --release --features gpu --bin alphapool_miner`
//! Handshake (challenge blake3 + auth) → notify → mine GPU 8×16 → mining.submit → accept/reject.
use ariaminer::gpu_ffi::ResidentCtx;
use ariaminer::official_grind::{canonical_gpu_config, try_mine_one_bounded_gpu_resident_ctx, Workspace};
use ariaminer::official_proof::encode_base64;
use ariaminer::stratum_to_official::{header_from_template, share_bound_le};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use zk_pow::api::proof::{IncompleteBlockHeader, MiningConfiguration};

const POOL: &str = "eu1.alphapool.tech:5566";
const WALLET: &str = "prl1p6cxk57fv4yrxtzr97mpzpr9xqr37fenvhmt9twn5z4wtxc5d7k0slejqmu";
const WORKER: &str = "GPU-PC1-ARIA";

#[derive(Clone)]
struct JobData {
    job_id: String,
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    bound_le: [u8; 32],
}

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
fn send(w: &Arc<Mutex<TcpStream>>, s: &str) {
    let mut g = w.lock().unwrap();
    let _ = g.write_all(s.as_bytes()); let _ = g.write_all(b"\n"); let _ = g.flush();
}

fn main() -> std::io::Result<()> {
    println!("▶ AlphaPool miner — test acceptation de notre shape 8×16");
    let stream = TcpStream::connect(POOL)?;
    let writer = Arc::new(Mutex::new(stream.try_clone()?));
    let mut rd = BufReader::new(stream);
    let job: Arc<Mutex<Option<JobData>>> = Arc::new(Mutex::new(None));
    let difficulty = Arc::new(AtomicU64::new(524288));
    let params_k = Arc::new(AtomicU64::new(4096));

    // Thread de minage GPU
    {
        let (job, writer, difficulty) = (job.clone(), writer.clone(), difficulty.clone());
        std::thread::spawn(move || {
            let (m, n, k) = (8192usize, 8192usize, 4096usize);
            let ctx = ResidentCtx::new(m, n, k, 64);
            let mut ws = Workspace::new();
            let mut rng = StdRng::seed_from_u64(0x5080_A1FA);
            let mut attempts = 0u64;
            loop {
                let j = { job.lock().unwrap().clone() };
                let Some(j) = j else { std::thread::sleep(Duration::from_millis(50)); continue };
                attempts += 1;
                // bound courant (vardiff peut avoir changé)
                let bound = share_bound_le(difficulty.load(Ordering::Relaxed), &j.config);
                if let Some(proof) = try_mine_one_bounded_gpu_resident_ctx(&ctx, &mut ws, &mut rng, &j.header, &j.config, &bound) {
                    match encode_base64(&proof) {
                        Ok(b64) => {
                            println!("  💎 share trouvée (attempt {attempts}) → submit job={}", j.job_id);
                            send(&writer, &format!(
                                r#"{{"id":99,"method":"mining.submit","params":["{WALLET}.{WORKER}","{}","{b64}"]}}"#, j.job_id));
                        }
                        Err(e) => eprintln!("encode err: {e}"),
                    }
                }
            }
        });
    }

    let mut id = 1u64;
    let mut hs_done = false;
    let t0 = Instant::now();
    let mut line = String::new();
    let mut accepted = 0u32;
    let mut rejected = 0u32;
    while t0.elapsed() < Duration::from_secs(90) {
        line.clear();
        if rd.read_line(&mut line)? == 0 { println!("(pool a fermé)"); break; }
        let msg: serde_json::Value = match serde_json::from_str(line.trim()) { Ok(v) => v, Err(_) => continue };
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        match method {
            "pearl.challenge" => {
                let p = &msg["params"];
                let seed_hex = p["seed"].as_str().unwrap_or("");
                let diff = p["difficulty"].as_u64().unwrap_or(32) as u32;
                let mut seed = [0u8; 32]; let _ = hex::decode_to_slice(seed_hex, &mut seed);
                let nonce = solve_challenge(seed, diff);
                println!("  ✅ challenge résolu");
                send(&writer, &format!(r#"{{"id":{id},"method":"pearl.challenge_response","params":{{"seed":"{seed_hex}","nonce":"{nonce:016x}"}}}}"#)); id += 1;
            }
            "mining.set_difficulty" => {
                if let Some(d) = msg["params"][0].as_u64() { difficulty.store(d, Ordering::Relaxed); println!("  set_difficulty {d}"); }
            }
            "pearl.set_mining_params" => {
                if let Some(k) = msg["params"][0]["k"].as_u64() { params_k.store(k, Ordering::Relaxed); }
                println!("  set_mining_params k={}", params_k.load(Ordering::Relaxed));
            }
            "mining.notify" => {
                let p = &msg["params"];
                let job_id = p[0].as_str().unwrap_or("").to_string();
                let ht = p[2].as_str().unwrap_or("");
                match header_from_template(ht) {
                    Ok(header) => {
                        let k = params_k.load(Ordering::Relaxed) as u32;
                        let config = canonical_gpu_config(k);
                        let bound = share_bound_le(difficulty.load(Ordering::Relaxed), &config);
                        *job.lock().unwrap() = Some(JobData { job_id: job_id.clone(), header, config, bound_le: bound });
                        println!("  📋 job {job_id} (header 76o OK, config 8×16, k={k})");
                    }
                    Err(e) => println!("  ⚠️ header parse err: {e}"),
                }
            }
            _ => {
                // réponses (accept/reject des submits)
                if let Some(r) = msg.get("result") {
                    if r.as_bool() == Some(true) && msg.get("id").and_then(|i| i.as_u64()) == Some(99) {
                        accepted += 1; println!("  ✅✅ SHARE ACCEPTÉE par AlphaPool ! (total {accepted})");
                    }
                }
                if let Some(e) = msg.get("error") {
                    if !e.is_null() { rejected += 1; println!("  ❌ REJET AlphaPool: {e}  (total {rejected})"); }
                }
            }
        }
        if !hs_done && msg.get("result").and_then(|r| r.as_bool()) == Some(true) && msg.get("id").and_then(|i| i.as_u64()) == Some(1) {
            hs_done = true;
            send(&writer, &format!(r#"{{"id":{id},"method":"mining.configure","params":[["pearl/v1"],{{}}]}}"#)); id += 1;
            send(&writer, &format!(r#"{{"id":{id},"method":"mining.subscribe","params":["aria-gpu-miner/1.0"]}}"#)); id += 1;
            send(&writer, &format!(r#"{{"id":{id},"method":"mining.authorize","params":["{WALLET}.{WORKER}","x;d=524288"]}}"#)); id += 1;
        }
    }
    println!("\n=== RÉSULTAT : {accepted} acceptées, {rejected} rejetées ===");
    if accepted > 0 { println!("🎉 NOTRE SHAPE 8×16 EST ACCEPTÉE PAR ALPHAPOOL — pas de rework 2×64 !"); }
    else if rejected > 0 { println!("→ Rejet : il faut le rework 2×64 (ou corriger le format)."); }
    else { println!("→ Aucune share soumise dans la fenêtre (diff trop haute ?) — relancer plus longtemps."); }
    Ok(())
}
