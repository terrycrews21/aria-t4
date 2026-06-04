//! Probe AlphaPool — prouve qu'on passe le handshake (pearl.challenge PoW + auth)
//! et qu'on reçoit des jobs. 1ère étape du port AlphaPool.
//! `cargo run --release --bin alphapool_probe`
//!
//! Protocole (reverse-eng tcpdump :5566 + binaire alpha) :
//!   pearl.challenge {seed,difficulty} → PoW : nonce u64 t.q. blake3(seed‖nonceLE)
//!     a ≥difficulty bits zéro en tête → pearl.challenge_response {seed,nonce}
//!   mining.configure [["pearl/v1"],{}] → mining.subscribe → mining.authorize
//!   pearl.set_mining_params (k/rank/m/n) ; mining.set_difficulty ; mining.notify
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const POOL: &str = "eu1.alphapool.tech:5566";
const WALLET: &str = "prl1p6cxk57fv4yrxtzr97mpzpr9xqr37fenvhmt9twn5z4wtxc5d7k0slejqmu";
const WORKER: &str = "GPU-PC1-ARIA";

fn leading_zero_bits(h: &[u8]) -> u32 {
    let mut n = 0;
    for &b in h {
        if b == 0 { n += 8; } else { return n + b.leading_zeros(); }
    }
    n
}

/// blake3(seed ‖ nonce.to_le_bytes()) — l'algo du challenge (craqué).
fn challenge_hash(seed: &[u8; 32], nonce: u64) -> [u8; 32] {
    let mut m = Vec::with_capacity(40);
    m.extend_from_slice(seed);
    m.extend_from_slice(&nonce.to_le_bytes());
    *blake3::hash(&m).as_bytes()
}

/// Brute-force parallèle (std threads) du nonce.
fn solve_challenge(seed: [u8; 32], difficulty: u32) -> u64 {
    let found = Arc::new(AtomicBool::new(false));
    let answer = Arc::new(AtomicU64::new(0));
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let mut handles = vec![];
    for t in 0..nthreads {
        let found = found.clone();
        let answer = answer.clone();
        handles.push(std::thread::spawn(move || {
            let mut nonce = t as u64;
            while !found.load(Ordering::Relaxed) {
                if leading_zero_bits(&challenge_hash(&seed, nonce)) >= difficulty {
                    answer.store(nonce, Ordering::Relaxed);
                    found.store(true, Ordering::Relaxed);
                    return;
                }
                nonce += nthreads as u64;
            }
        }));
    }
    for h in handles { let _ = h.join(); }
    answer.load(Ordering::Relaxed)
}

fn send(w: &mut TcpStream, s: &str) -> std::io::Result<()> {
    println!("→ {s}");
    w.write_all(s.as_bytes())?; w.write_all(b"\n")?; w.flush()
}

fn main() -> std::io::Result<()> {
    println!("▶ connexion AlphaPool {POOL}");
    let stream = TcpStream::connect(POOL)?;
    stream.set_read_timeout(Some(Duration::from_secs(40)))?;
    let mut wr = stream.try_clone()?;
    let mut rd = BufReader::new(stream);

    let mut id = 1u64;
    let mut handshake_done = false;
    let t0 = Instant::now();
    let mut line = String::new();
    while t0.elapsed() < Duration::from_secs(35) {
        line.clear();
        let n = rd.read_line(&mut line)?;
        if n == 0 { println!("(connexion fermée par la pool)"); break; }
        let msg: serde_json::Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v, Err(_) => { println!("← (non-JSON) {}", line.trim()); continue; }
        };
        println!("← {}", line.trim());
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        match method {
            "pearl.challenge" => {
                let p = &msg["params"];
                let seed_hex = p["seed"].as_str().unwrap_or("");
                let diff = p["difficulty"].as_u64().unwrap_or(32) as u32;
                let mut seed = [0u8; 32];
                hex::decode_to_slice(seed_hex, &mut seed).ok();
                println!("  ⚙ résolution challenge (blake3, {diff} bits zéro)...");
                let ts = Instant::now();
                let nonce = solve_challenge(seed, diff);
                println!("  ✅ nonce trouvé = {:016x} en {:.1}s (vérif {} bits)",
                    nonce, ts.elapsed().as_secs_f64(), leading_zero_bits(&challenge_hash(&seed, nonce)));
                send(&mut wr, &format!(
                    r#"{{"id":{id},"method":"pearl.challenge_response","params":{{"seed":"{seed_hex}","nonce":"{:016x}"}}}}"#, nonce))?;
                id += 1;
            }
            _ => {}
        }
        // dès qu'un id-réponse au challenge arrive (result:true) → handshake
        if !handshake_done && msg.get("result").and_then(|r| r.as_bool()) == Some(true) {
            handshake_done = true;
            send(&mut wr, &format!(r#"{{"id":{id},"method":"mining.configure","params":[["pearl/v1"],{{}}]}}"#))?; id += 1;
            send(&mut wr, &format!(r#"{{"id":{id},"method":"mining.subscribe","params":["aria-gpu-miner/1.0"]}}"#))?; id += 1;
            send(&mut wr, &format!(r#"{{"id":{id},"method":"mining.authorize","params":["{WALLET}.{WORKER}","x;d=524288"]}}"#))?; id += 1;
        }
    }
    println!("\n=== Fin probe (35s). Si on a reçu set_mining_params + notify = handshake OK. ===");
    Ok(())
}
