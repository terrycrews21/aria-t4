//! Bench du pipeline résident (contexte persistant) — mesure le débit réel.
//! `cargo run --release --features gpu --bin resident_bench -- [m] [n] [k] [iters]`
//! Bound impossible (0) → aucun hit → boucle pure gen+commit+noise+grind.
//! hashrate TH/s = m·n·k · setups/s / 1e12  (= TMAC/s, convention Pearl).
use ariaminer::gpu_ffi::ResidentCtx;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(8192);
    let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(8192);
    let k: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4032);
    let iters: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);

    let job_key = [0x5au8; 32];
    let bound = [0u8; 32]; // impossible -> 0 hit, mesure le débit pur

    println!("▶ resident_bench m={m} n={n} k={k} iters={iters}");
    let ctx = ResidentCtx::new(m, n, k, 64);

    // warm-up (compile JIT, caches)
    let _ = ctx.grind(1, &job_key, &bound);

    let t0 = Instant::now();
    let mut total_found = 0i64;
    for s in 0..iters {
        let (found, _) = ctx.grind(0xC0FFEE_0000 + s as u64, &job_key, &bound);
        total_found += found.max(0) as i64;
    }
    let dt = t0.elapsed().as_secs_f64();

    let setups_per_s = iters as f64 / dt;
    let macs = m as f64 * n as f64 * k as f64;
    let th_s = macs * setups_per_s / 1e12;
    let tops = 2.0 * th_s;
    println!(
        "  {iters} setups en {:.2}s → {:.1} setups/s",
        dt, setups_per_s
    );
    println!(
        "  débit = {:.0} TOPS = {:.1} TH/s  (hits parasites={total_found})",
        tops, th_s
    );
    println!(
        "  réf : alpha-miner 174.78 TH/s · GEMM seul 217 TH/s (gpu_sat). Prologue inclus ici."
    );
}
