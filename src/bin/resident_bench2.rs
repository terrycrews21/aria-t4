//! Bench DOUBLE-BUFFER batché — mesure le débit avec overlap prologue‖grind.
//! `cargo run --release --features gpu --bin resident_bench2 -- [m] [n] [k] [batch] [reps]`
use ariaminer::gpu_ffi::ResidentCtx2;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(8192);
    let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(8192);
    let k: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4032);
    let batch: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
    let reps: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(6);

    let job_key = [0x5au8; 32];
    let bound = [0u8; 32]; // 0 hit → débit pur

    println!("▶ resident_bench2 (double-buffer) m={m} n={n} k={k} batch={batch} reps={reps}");
    let ctx = ResidentCtx2::new(m, n, k);
    let _ = ctx.grind_batch(1, &job_key, &bound, batch); // warm-up

    let t0 = Instant::now();
    let mut setups = 0usize;
    for r in 0..reps {
        let _ = ctx.grind_batch(0xC0FFEE00 + (r * batch) as u64, &job_key, &bound, batch);
        setups += batch;
    }
    let dt = t0.elapsed().as_secs_f64();
    let sps = setups as f64 / dt;
    let macs = m as f64 * n as f64 * k as f64;
    let th = macs * sps / 1e12;
    println!("  {setups} setups en {:.2}s → {:.1} setups/s", dt, sps);
    println!("  débit = {:.0} TOPS = {:.1} TH/s", 2.0 * th, th);
    println!("  réf : single-buffer 535 setups/s=145 TH/s · alpha 174.78 · GEMM seul 217");
}
