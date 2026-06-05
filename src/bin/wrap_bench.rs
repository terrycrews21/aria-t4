//! wrap_bench — isole le COÛT DU WRAPPER live par setup (v0.5.0).
//! Compare, à grille identique 8192², la boucle tight `ctx.grind` (= resident_bench,
//! 159 TH/s) vs la boucle réelle `try_mine_one_bounded_gpu_resident_ctx` (le chemin
//! du miner live). L'écart = ce que le wrapper coûte en CPU et qui affame le GPU.
use ariaminer::gpu_ffi::ResidentCtx;
use ariaminer::official_grind::{canonical_gpu_config, try_mine_one_bounded_gpu_resident_ctx, Workspace};
use rand::{rngs::StdRng, SeedableRng};
use std::time::Instant;
use zk_pow::api::proof::IncompleteBlockHeader;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(8192);
    let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(8192);
    let k: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let iters: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(300);

    let header = IncompleteBlockHeader {
        version: 0, prev_block: [1u8; 32], merkle_root: [2u8; 32],
        timestamp: 0x6666_6666, nbits: 0x207f_ffff,
    };
    let config = canonical_gpu_config(k as u32);
    let job_key = [0x5au8; 32];
    let bound_le = [0u8; 32]; // 0 hit → mesure le débit pur
    let macs = m as f64 * n as f64 * k as f64;

    let ctx = ResidentCtx::new(m, n, k, 64);
    let mut ws = Workspace::new();
    let mut rng = StdRng::seed_from_u64(0x5080_AB1A_C0FFEE);

    println!("▶ wrap_bench m={m} n={n} k={k} iters={iters}\n");

    // A) boucle TIGHT (= bench) : juste ctx.grind
    let _ = ctx.grind(1, &job_key, &bound_le);
    let t0 = Instant::now();
    for s in 0..iters { let _ = ctx.grind(0xC0FFEE + s as u64, &job_key, &bound_le); }
    let dt_a = t0.elapsed().as_secs_f64();
    let sps_a = iters as f64 / dt_a;
    println!("  A) tight ctx.grind         = {:>6.1} TH/s ({sps_a:.0} setups/s)", macs * sps_a / 1e12);

    // B) boucle WRAPPER réelle du miner
    let _ = try_mine_one_bounded_gpu_resident_ctx(&ctx, &mut ws, &mut rng, &header, &config, &bound_le);
    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = try_mine_one_bounded_gpu_resident_ctx(&ctx, &mut ws, &mut rng, &header, &config, &bound_le);
    }
    let dt_b = t1.elapsed().as_secs_f64();
    let sps_b = iters as f64 / dt_b;
    println!("  B) wrapper live complet    = {:>6.1} TH/s ({sps_b:.0} setups/s)", macs * sps_b / 1e12);

    let wrap_ms = (1.0 / sps_b - 1.0 / sps_a) * 1000.0;
    println!("\n  ⇒ surcoût wrapper = {wrap_ms:.3} ms/setup (CPU sérialisé qui affame le GPU)");
    println!("    (A inclut déjà ~1.72ms GPU ; si wrapper≈0 → 1 thread suffit pour 159)");
}
