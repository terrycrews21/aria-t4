
//! Self-contained live GPU share probe: connects to the real Pearl pool,
//! grinds ONE real share on the resident GPU path (feature `gpu`), at the
//! pool's ACTUAL difficulty, throttled to a hard duty-cycle cap so sustained
//! GPU utilization stays bounded, submits it, reports the pool's response,
//! then exits. Single-shot, bounded — no long-running loop.
//!
//!   ARIA_GPU_DUTY=40  ./share_probe_gpu    # cap ~40% GPU util

use ariaminer::gpu_ffi::{ResidentCtx, ProofGpuCtx};
use ariaminer::official_grind::{compute_job_key_pub, build_proofs_from_setup_gpu_fixb};
use ariaminer::official_proof::encode_base64_gzip;
use ariaminer::protocol::{Job, MiningParams};
use ariaminer::stratum_to_official::build_official_job;
use rand::RngCore;
use rand::SeedableRng;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

const WALLET: &str = "prl1pu3mc6ex4n4nznknctdafleq3asq4fr0njpwz4vqnt6e4xlnv72hq5s528j";
const WORKER: &str = "share_probe_gpu";

fn herominers_default_params() -> MiningParams {
    let (rows_pattern, cols_pattern) = (
        vec![0, 8, 32, 40, 64, 72, 96, 104],
        vec![0, 1, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113],
    );
    MiningParams { m: 16_384, n: 65_536, k: 8192, rank: 128, rows_pattern, cols_pattern, mma_type: "Int7xInt7ToInt32".into() }
}

fn main() -> anyhow::Result<()> {
    let duty_pct: u64 = std::env::var("ARIA_GPU_DUTY").ok().and_then(|s| s.parse().ok()).unwrap_or(40).clamp(1, 100);
    println!("[probeG] GPU duty-cycle cap = {duty_pct}%");

    let host = "br.pearl.gfwroute.com:1200";
    println!("[probeG] connecting {host}");
    let stream = TcpStream::connect(host)?;
    stream.set_read_timeout(Some(Duration::from_millis(500)))?;
    let mut wr = stream.try_clone()?;
    let mut rd = BufReader::new(stream);

    let login = format!("{}.{}", WALLET, WORKER);
    wr.write_all(format!("{{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"ariaminer/0.1.0\",\"{login}\"]}}\n").as_bytes())?;
    wr.write_all(format!("{{\"id\":2,\"method\":\"mining.authorize\",\"params\":{{\"agent\":\"ariaminer/0.1.0\",\"password\":\"x\",\"type\":\"v2\",\"wallet\":\"{login}\",\"worker\":\"\"}}}}\n").as_bytes())?;
    println!("[probeG] sent subscribe+authorize, waiting for job...");

    let start = Instant::now();
    let mut job_json: Option<serde_json::Value> = None;
    let mut line = String::new();
    while start.elapsed() < Duration::from_secs(20) && job_json.is_none() {
        line.clear();
        match rd.read_line(&mut line) {
            Ok(0) => { println!("[probeG] connection closed early"); return Ok(()); }
            Ok(_) => {
                if line.trim().is_empty() { continue; }
                println!("[probeG] recv: {}", line.trim());
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
    println!("[probeG] job_id={} header={}...", job.job_id, &job.header_template[..16]);

    let suffix: u64 = job.job_id.rsplit_once('_').and_then(|(_, d)| d.parse().ok()).expect("job_id suffix");
    let difficulty = suffix.saturating_mul(1u64 << 32);
    println!("[probeG] pool difficulty={difficulty} (2^{:.1})", (difficulty as f64).log2());

    let params = herominers_default_params();
    let oj = build_official_job(&job, &params, difficulty)?;
    println!("[probeG] official job: m={} n={} k={} bound_le={}", oj.m, oj.n, oj.k, hex::encode(oj.bound_le));

    let mut header = oj.header;
    let mut bound_le = oj.bound_le;
    let mut cur_job_id = job.job_id.clone();
    let mut job_key = compute_job_key_pub(&oj.header, &oj.config);
    let mut drain_buf = String::new();
    // fast non-grind socket polling during the duty window
    rd.get_ref().set_read_timeout(Some(Duration::from_millis(1)))?;
    let tile_h = oj.config.rows_pattern.to_list().len();
    let tile_w = oj.config.cols_pattern.to_list().len();
    let rank = oj.config.rank as usize;

    println!("[probeG] allocating ResidentCtx({},{},{})", oj.m, oj.n, oj.k);
    let gpu_ctx = ResidentCtx::new(oj.m, oj.n, oj.k, 64);
    let mut rng = rand::rngs::StdRng::from_entropy();
    let mut next_seed = rng.next_u64();

    let max_secs: u64 = std::env::var("ARIA_PROBE_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(120);
    let grind_start = Instant::now();
    let mut attempts: u64 = 0;
    let mut proof_opt = None;
    let mut b_seed_job: Option<u64> = None;

    println!("[probeG] grinding on REAL pool bound (GPU resident path, duty={duty_pct}%)...");
    while grind_start.elapsed() < Duration::from_secs(max_secs) {
        let setup_seed = next_seed;
        next_seed = rng.next_u64();
        let b_seed = *b_seed_job.get_or_insert(setup_seed);

        let gt0 = Instant::now();
        let (found, hits) = gpu_ctx.grind2(setup_seed, next_seed, &job_key, &bound_le);
        if duty_pct < 100 {
            let spent = gt0.elapsed().as_secs_f64();
            if spent > 0.0 {
                let sleep_s = spent * (100.0 - duty_pct as f64) / duty_pct as f64;
                std::thread::sleep(Duration::from_secs_f64(sleep_s));
            }
        }
        // roll with pool job updates (pool emits a fresh notify every ~11s);
        // a share stamped on an ancient job_id would be rejected as stale.
        loop {
            match rd.read_line(&mut drain_buf) {
                Ok(0) => { println!("[probeG] connection closed by peer during grind"); break; }
                Ok(_) => {
                    if drain_buf.ends_with('\n') {
                        let msg = drain_buf.trim_end().to_string();
                        drain_buf.clear();
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&msg) {
                            if v.get("method").and_then(|m| m.as_str()) == Some("mining.notify") {
                                let nj = Job::from_params(v.get("params").expect("params")).expect("notify parse");
                                if nj.job_id != cur_job_id {
                                    let nsuffix: u64 = nj.job_id.rsplit_once('_').and_then(|(_, d)| d.parse().ok()).expect("job_id suffix");
                                    let ndiff = nsuffix.saturating_mul(1u64 << 32);
                                    let noj = build_official_job(&nj, &params, ndiff)?;
                                    job_key = compute_job_key_pub(&noj.header, &noj.config);
                                    bound_le = noj.bound_le;
                                    header = noj.header;
                                    cur_job_id = nj.job_id.clone();
                                    b_seed_job = None;
                                    println!("[probeG] rolled to new job {}", cur_job_id);
                                }
                            }
                        }
                    }
                    break;
                }
                Err(_) => break,
            }
        }
        attempts += 1;
        if attempts % 20 == 0 {
            println!("[probeG] {attempts} setups, {:?} elapsed, no hit yet", grind_start.elapsed());
        }
        if found > 0 {
            if let Some(hit) = hits.into_iter().find(|h| h.rows.len() == tile_h && h.cols.len() == tile_w) {
                println!("[probeG] HIT at setup {attempts} after {:?}", grind_start.elapsed());
                let ca = ProofGpuCtx::new(oj.m, oj.k, 256);
                let cb = ProofGpuCtx::new(oj.n, oj.k, 256);
                let proofs = build_proofs_from_setup_gpu_fixb(
                    &ca, &cb, setup_seed, b_seed, std::slice::from_ref(&hit),
                    &job_key, oj.m, oj.n, oj.k, rank, tile_h, tile_w,
                );
                if let Some(p) = proofs.into_iter().next() {
                    proof_opt = Some(p);
                    break;
                }
            }
        }
    }

    let proof = match proof_opt {
        Some(p) => p,
        None => { println!("[probeG] no hit within ARIA_PROBE_SECS budget"); return Ok(()); }
    };

    let (private_params, public_params) = ariaminer::official_proof::parse_plain_proof(header, &proof)?;
    let compiled = zk_pow::api::proof_utils::CompiledPublicParams::from(&public_params);
    let noise = zk_pow::circuit::pearl_noise::compute_noise(&compiled);
    let jackpot = zk_pow::circuit::chip::compute_jackpot(&compiled, &private_params.s_a, &private_params.s_b, &noise);
    let hash_jackpot = zk_pow::api::proof_utils::compute_jackpot_hash(&jackpot, compiled.a_noise_seed());
    let hash_u256 = primitive_types::U256::from_little_endian(&hash_jackpot);
    let bound_u256 = primitive_types::U256::from_little_endian(&bound_le);
    println!("[probeG] local verify: hash<=bound = {}", hash_u256 <= bound_u256);

    let plain_proof = encode_base64_gzip(&proof)?;
    println!("[probeG] submitting share, proof_b64_len={}", plain_proof.len());
    let submit = format!(
        "{{\"id\":3,\"method\":\"mining.submit\",\"params\":{{\"hs\":1.0,\"job_id\":\"{}\",\"plain_proof\":\"{plain_proof}\"}}}}\n",
        cur_job_id
    );
    wr.write_all(submit.as_bytes())?;

    let sub_start = Instant::now();
    while sub_start.elapsed() < Duration::from_secs(10) {
        line.clear();
        match rd.read_line(&mut line) {
            Ok(0) => { println!("[probeG] connection closed after submit"); break; }
            Ok(_) => { if !line.trim().is_empty() { println!("[probeG] POOL_RESPONSE: {}", line.trim()); } }
            Err(_) => {}
        }
    }
    println!("[probeG] SHARE_PROBE_GPU_DONE");
    Ok(())
}
