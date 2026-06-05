//! overlap_bench — balaie la taille de grille G du grind PERSISTANT pour trouver
//! le sweet-spot d'overlap prologue(N+1) ‖ grind(N) (v0.5.0).
//!
//! G=0 → grille pleine (= ancien Ctx2, overlap nul, ~140). G>0 → grind persistant
//! à G blocs (occupation bridée) → laisse des SM libres pour le prologue.
//! Cible : dépasser 157 (1 contexte série) en cachant le prologue (~193 plafond).
//!
//! `cargo run --release --features gpu --bin overlap_bench -- [m] [n] [k] [num] [G...]`
use ariaminer::gpu_ffi::{ResidentCtx2, gpu_sm_count};
use std::time::Instant;

fn bench(ctx: &ResidentCtx2, g: usize, m: usize, n: usize, k: usize, num: usize) -> f64 {
    ctx.set_grind_blocks(g);
    let job_key = [0x5au8; 32];
    let bound = [0u8; 32]; // 0 hit
    let _ = ctx.grind_batch(1, &job_key, &bound, 8); // warm-up
    let t0 = Instant::now();
    let _ = ctx.grind_batch(0xC0FFEE, &job_key, &bound, num);
    let dt = t0.elapsed().as_secs_f64();
    let macs = m as f64 * n as f64 * k as f64;
    macs * (num as f64 / dt) / 1e12
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(8192);
    let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(8192);
    let k: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let num: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);

    let sm = gpu_sm_count();
    let full_tiles = m.div_ceil(128) * n.div_ceil(128);
    println!("▶ overlap_bench m={m} n={n} k={k} num={num} | SM={sm} | tuiles pleines={full_tiles}");
    println!("  réf : 1 contexte série = 157 TH/s · plafond (prologue caché) ~193 · GEMM pur 217\n");

    // Sweep G : 0 (grille pleine) puis multiples du nombre de SM (headroom variable).
    let sweep: Vec<usize> = if a.len() > 5 {
        a[5..].iter().filter_map(|s| s.parse().ok()).collect()
    } else if sm > 0 {
        vec![0, sm * 4, sm * 3, sm * 2, (sm * 3) / 2, sm, (sm * 3) / 4, sm / 2]
    } else {
        vec![0, 336, 252, 168, 126, 84, 63, 42]
    };

    let ctx = ResidentCtx2::new(m, n, k);
    let base = bench(&ctx, 0, m, n, k, num);
    println!("  G=0 (grille pleine, baseline Ctx2) = {base:.1} TH/s");
    let mut best = (0usize, base);
    for &g in sweep.iter().filter(|&&g| g != 0) {
        let th = bench(&ctx, g, m, n, k, num);
        let tag = if th > best.1 { " ◀ best" } else { "" };
        let pct = (g as f64 / sm.max(1) as f64) * 100.0;
        println!("  G={g:>5} ({pct:>4.0}% SM) = {th:>6.1} TH/s{tag}");
        if th > best.1 { best = (g, th); }
    }
    println!("\n  ⇒ meilleur : G={} → {:.1} TH/s (vs 157 série, vs ~193 plafond)", best.0, best.1);
}
