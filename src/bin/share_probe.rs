
//! Self-contained live share probe: connects to the real Pearl pool, grinds
//! ONE real share on the CPU production path at the pool's ACTUAL difficulty,
//! submits it, and reports the pool's response. No GPU, no long-running loop —
//! single-shot, bounded, exits immediately after one submit or a timeout.

use ariaminer::official_grind::{Workspace, try_mine_one_bounded};
use ariaminer::official_proof::encode_base64_gzip;
use ariaminer::protocol::{Job, MiningParams};
use ariaminer::stratum_to_official::{config_from_params, share_bound_le, header_from_template};
use rand::SeedableRng;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

const WALLET: &str = "prl1pu3mc6ex4n4nznknctdafleq3asq4fr0njpwz4vqnt6e4xlnv72hq5s528j";
const WORKER: &str = "share_probe";

fn herominers_default_params() -> MiningParams {
    let (rows_pattern, cols_pattern) = (
        vec![0, 8, 32, 40, 64, 72, 96, 104],
        vec![0, 1, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113],
    );
    MiningParams { m: 16_384, n: 65_536, k: 8192, rank: 128, rows_pattern, cols_pattern, mma_type: "Int7xInt7ToInt32".into() }
}

fn main() -> anyhow::Result<()> {
    let host = "br.pearl.gfwroute.com:1200";
    println!("[probe] connecting {host}");
    let stream = TcpStream::connect(host)?;
    stream.set_read_timeout(Some(Duration::from_millis(500)))?;
    let mut wr = stream.try_clone()?;
    let mut rd = BufReader::new(stream);

    let login = format!("{}.{}", WALLET, WORKER);
    wr.write_all(format!("{{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"ariaminer/0.1.0\",\"{login}\"]}}\n").as_bytes())?;
    wr.write_all(format!("{{\"id\":2,\"method\":\"mining.authorize\",\"params\":{{\"agent\":\"ariaminer/0.1.0\",\"password\":\"x\",\"type\":\"v2\",\"wallet\":\"{login}\",\"worker\":\"\"}}}}\n").as_bytes())?;
    println!("[probe] sent subscribe+authorize, waiting for job...");

    let start = Instant::now();
    let mut job_json: Option<serde_json::Value> = None;
    let mut line = String::new();
    while start.elapsed() < Duration::from_secs(20) && job_json.is_none() {
        line.clear();
        match rd.read_line(&mut line) {
            Ok(0) => { println!("[probe] connection closed early"); return Ok(()); }
            Ok(_) => {
                if line.trim().is_empty() { continue; }
                println!("[probe] recv: {}", line.trim());
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    if v.get("method").and_then(|m| m.as_str()) == Some("mining.notify") {
                        job_json = Some(v);
                    }
                }
            }
            Err(_) => {}
        }
    }
    let v = job_json.expect("no job received within 20s");
    let job = Job::from_params(v.get("params").expect("params"))?;
    println!("[probe] job_id={} header={}...", job.job_id, &job.header_template[..16]);

    let suffix: u64 = job.job_id.rsplit_once('_').and_then(|(_, d)| d.parse().ok()).expect("job_id suffix");
    let difficulty = suffix.saturating_mul(1u64 << 32);
    println!("[probe] pool difficulty={difficulty} (2^{:.1})", (difficulty as f64).log2());

    let params = herominers_default_params();
    let config = config_from_params(&params)?;
    let bound_le = share_bound_le(difficulty, &config);
    let header = header_from_template(&job.header_template)?;

    let (m, n, k) = (256usize, 256usize, params.k as usize);
    let mut ws = Workspace::new();
    let mut rng = rand::rngs::StdRng::from_entropy();

    println!("[probe] grinding on REAL pool bound (CPU path)...");
    let grind_start = Instant::now();
    let mut proof = None;
    let max_attempts: u64 = 5_000_000;
    for attempt in 0..max_attempts {
        if let Some(p) = try_mine_one_bounded(&mut ws, &mut rng, m, n, k, &header, &config, &bound_le) {
            println!("[probe] HIT at attempt {attempt} after {:?}", grind_start.elapsed());
            proof = Some(p);
            break;
        }
        if attempt % 200_000 == 0 && attempt > 0 {
            println!("[probe] {attempt} attempts, {:?} elapsed, no hit yet", grind_start.elapsed());
        }
        if grind_start.elapsed() > Duration::from_secs(60) { break; }
    }
    let proof = match proof {
        Some(p) => p,
        None => { println!("[probe] no hit within budget — pool difficulty too high for CPU-only probe"); return Ok(()); }
    };

    let plain_proof = encode_base64_gzip(&proof)?;
    println!("[probe] submitting share, proof_b64_len={}", plain_proof.len());
    let submit = format!(
        "{{\"id\":3,\"method\":\"mining.submit\",\"params\":{{\"hs\":1.0,\"job_id\":\"{}\",\"plain_proof\":\"{plain_proof}\"}}}}\n",
        job.job_id
    );
    wr.write_all(submit.as_bytes())?;

    let sub_start = Instant::now();
    while sub_start.elapsed() < Duration::from_secs(10) {
        line.clear();
        match rd.read_line(&mut line) {
            Ok(0) => { println!("[probe] connection closed after submit"); break; }
            Ok(_) => { if !line.trim().is_empty() { println!("[probe] POOL_RESPONSE: {}", line.trim()); } }
            Err(_) => {}
        }
    }
    println!("[probe] SHARE_PROBE_DONE");
    Ok(())
}
