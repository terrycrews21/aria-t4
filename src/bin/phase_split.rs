//! phase_split — décompose le temps GPU d'un setup : prologue (gen+commit+noise)
//! vs grind (GEMM). But : valider la thèse v0.5.0 « le prologue coûte ~X% NON
//! recouvert par le grind ; le cacher sous le grind = le cap vers 175-217 ».
//!
//! `cargo run --release --features gpu --bin phase_split -- [m] [n] [k] [iters]`
//! Bound impossible (0) → 0 hit → mesure le débit pur + le split par cudaEvents.
use ariaminer::gpu_ffi::ResidentCtx;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(8192);
    let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(8192);
    let k: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let iters: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);

    let job_key = [0x5au8; 32];
    let bound = [0u8; 32]; // impossible -> 0 hit

    println!("▶ phase_split m={m} n={n} k={k} iters={iters}");
    let ctx = ResidentCtx::new(m, n, k, 64);
    ctx.set_timing(true);
    let _ = ctx.grind(1, &job_key, &bound); // warm-up (JIT, caches)

    let t0 = Instant::now();
    let (mut sum_pro, mut sum_grind) = (0f64, 0f64);
    let (mut sum_genc, mut sum_noise) = (0f64, 0f64);
    for s in 0..iters {
        let _ = ctx.grind(0xC0FFEE_0000 + s as u64, &job_key, &bound);
        let (p, g) = ctx.last_times();
        let (gc, no, _gr) = ctx.last_times4();
        sum_pro += p as f64;
        sum_grind += g as f64;
        sum_genc += gc as f64;
        sum_noise += no as f64;
    }
    let dt = t0.elapsed().as_secs_f64();
    let genc_ms = sum_genc / iters as f64;
    let noise_ms = sum_noise / iters as f64;

    let setups_per_s = iters as f64 / dt;
    let macs = m as f64 * n as f64 * k as f64;
    let th_s = macs * setups_per_s / 1e12;
    let pro_ms = sum_pro / iters as f64;
    let grind_ms = sum_grind / iters as f64;
    let wall_ms = dt / iters as f64 * 1000.0;
    let gpu_busy = pro_ms + grind_ms; // temps GPU série mesuré (prologue puis grind)
    let frac_pro = pro_ms / gpu_busy * 100.0;

    println!("  débit ACTUEL = {th_s:.1} TH/s ({setups_per_s:.0} setups/s, wall {wall_ms:.2} ms/setup)");
    println!("  ── split GPU par setup (cudaEvents, série) ──");
    println!("     · gen+commit+stir          = {genc_ms:.3} ms  ({:.1}%)", genc_ms / gpu_busy * 100.0);
    println!("     · noise (perm+add)         = {noise_ms:.3} ms  ({:.1}%)", noise_ms / gpu_busy * 100.0);
    println!("     prologue (gen+commit+noise) = {pro_ms:.3} ms  ({frac_pro:.1}%)");
    println!("     grind (GEMM+fold+powcheck)  = {grind_ms:.3} ms  ({:.1}%)", 100.0 - frac_pro);
    println!("     somme GPU série             = {gpu_busy:.3} ms   (overhead hôte = {:.3} ms)", wall_ms - gpu_busy);
    // Plafond si le prologue était PARFAITEMENT caché sous le grind (= max au lieu de somme).
    let hidden_ms = grind_ms.max(pro_ms);
    let ceil_th = macs / (hidden_ms / 1000.0) / 1e12;
    println!("  ── thèse v0.5.0 ──");
    println!("     si prologue 100% recouvert → ~{ceil_th:.0} TH/s (grind seul = {:.0} TH/s)",
             macs / (grind_ms / 1000.0) / 1e12);
    println!("     réf : alpha 174.78 · GEMM seul gpu_sat 217");
}
