//! v0.6.3-beta — VALIDATION BYTE-EXACT du path PORTABLE cp.async vs le path TMA.
//! Deux passes sur les MÊMES seeds : passe 1 = kernel TMA (défaut 5080), passe 2 =
//! kernel cp.async forcé (ARIA_FORCE_CPASYNC). Exigé : found + hits IDENTIQUES par setup.
//! Comme le path TMA == zk-pow officiel (prouvé v0.6.0), cp.async == TMA ⇒ cp.async == officiel
//! ⇒ le mineur portable Ampere/Ada produira des jackpots byte-exacts.
//! Lancer : ARIA_RANK=256 ./cpasync_check   (forme via ARIA_BATCH_M/N, déf 8192²)
use ariaminer::gpu_ffi::ResidentCtx;
use ariaminer::official_grind::{canonical_gpu_config, compute_job_key_pub};
use zk_pow::api::proof::IncompleteBlockHeader;

fn main() {
    unsafe {
        std::env::set_var("ARIA_FIXB", "1");
        std::env::set_var("ARIA_TMA_MS", "1");
    }
    let dim = |v: &str, d: usize| std::env::var(v).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let (m, n, k) = (dim("ARIA_BATCH_M", 8192), dim("ARIA_BATCH_N", 8192), 4096usize);
    let nsetups = dim("ARIA_NSETUPS", 30);
    println!("  forme {m}×{n}×{k}, {nsetups} setups — TMA vs cp.async");
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [1u8; 32],
        merkle_root: [2u8; 32],
        timestamp: 0x6666_6666,
        nbits: 0x207f_ffff,
    };
    let config = canonical_gpu_config(k as u32);
    let bound = zk_pow::api::sanity_checks::extract_difficulty_bound(header.nbits, &config);
    let mut bound_le = [0u8; 32];
    bound.to_little_endian(&mut bound_le);
    let job_key = compute_job_key_pub(&header, &config);
    let seeds: Vec<u64> = (0..nsetups as u64)
        .map(|i| 0xC0FF_EE00u64 ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let max_hits = 64usize;
    let canon = |hits: &[ariaminer::gpu_ffi::Hit]| -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut v: Vec<_> = hits.iter().map(|h| (h.rows.clone(), h.cols.clone())).collect();
        v.sort();
        v
    };

    println!("▶ passe 1 : kernel TMA (Blackwell natif)…");
    unsafe { std::env::remove_var("ARIA_FORCE_CPASYNC"); }
    let ctx1 = ResidentCtx::new(m, n, k, max_hits);
    let reference: Vec<(i32, Vec<(Vec<usize>, Vec<usize>)>)> = seeds
        .iter()
        .map(|&s| { let (f, h) = ctx1.grind(s, &job_key, &bound_le); (f, canon(&h)) })
        .collect();
    drop(ctx1);

    println!("▶ passe 2 : kernel cp.async PORTABLE (forcé)…");
    unsafe { std::env::set_var("ARIA_FORCE_CPASYNC", "1"); }
    let ctx2 = ResidentCtx::new(m, n, k, max_hits);
    let mut nerr = 0usize;
    for (i, &s) in seeds.iter().enumerate() {
        let (found, hits) = ctx2.grind(s, &job_key, &bound_le);
        let (rfound, rhits) = &reference[i];
        if found != *rfound {
            println!("❌ setup {i} seed={s:#x} : found {found} ≠ TMA {rfound}");
            nerr += 1; continue;
        }
        if (found as usize) <= max_hits && canon(&hits) != *rhits {
            println!("❌ setup {i} seed={s:#x} : hits différents (found={found})");
            nerr += 1;
        }
    }
    if nerr == 0 {
        println!("\n🎉 cp.async == TMA byte-exact — {nsetups} setups : found + hits IDENTIQUES.");
        println!("   ⇒ le mineur PORTABLE (Ampere/Ada) produira des jackpots consensus-valides.");
    } else {
        println!("\n❌ {nerr} divergence(s) — le path cp.async N'est PAS byte-exact, à corriger.");
        std::process::exit(2);
    }
}
