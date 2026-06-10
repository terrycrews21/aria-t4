//! v0.6.2 — VALIDATION A/B de l'OVERLAP PROLOGUE.
//! Deux passes sur les MÊMES seeds : chemin classique (`grind`) vs pipeliné (`grind2`,
//! prologue N+1 préfetché pendant le grind N). Exigé : `found` identique par setup,
//! et ensembles de hits identiques (triés) quand found ≤ max_hits.
//! Le kernel grind est inchangé — ce harnais prouve que le pipeline fournit les MÊMES
//! entrées par setup (A/noise/pow_key), donc les mêmes sorties.
//! Lancer : ARIA_RANK=256 ./overlap_check   (forme via ARIA_BATCH_M/N, déf 8192²)
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
    println!("  forme {m}×{n}×{k}, {nsetups} setups");
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [1u8; 32],
        merkle_root: [2u8; 32],
        timestamp: 0x6666_6666,
        nbits: 0x207f_ffff, // facile → hits réguliers
    };
    let config = canonical_gpu_config(k as u32);
    let bound = zk_pow::api::sanity_checks::extract_difficulty_bound(header.nbits, &config);
    let mut bound_le = [0u8; 32];
    bound.to_little_endian(&mut bound_le);
    let job_key = compute_job_key_pub(&header, &config);
    let seeds: Vec<u64> = (0..nsetups as u64)
        .map(|i| 0xA11A_5EEDu64 ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let max_hits = 64usize;

    // clé de comparaison : hits triés (rows/cols déjà triés+dédupliqués par le wrapper)
    let canon = |hits: &[ariaminer::gpu_ffi::Hit]| -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut v: Vec<_> = hits.iter().map(|h| (h.rows.clone(), h.cols.clone())).collect();
        v.sort();
        v
    };

    println!("▶ passe 1 : chemin CLASSIQUE (grind)…");
    let ctx1 = ResidentCtx::new(m, n, k, max_hits);
    let reference: Vec<(i32, Vec<(Vec<usize>, Vec<usize>)>)> = seeds
        .iter()
        .map(|&s| {
            let (found, hits) = ctx1.grind(s, &job_key, &bound_le);
            (found, canon(&hits))
        })
        .collect();
    drop(ctx1);

    println!("▶ passe 2 : chemin PIPELINÉ (grind2, prologue préfetché)…");
    let ctx2 = ResidentCtx::new(m, n, k, max_hits);
    let mut nerr = 0usize;
    for (i, &s) in seeds.iter().enumerate() {
        let nxt = seeds.get(i + 1).copied().unwrap_or(0);
        let (found, hits) = ctx2.grind2(s, nxt, &job_key, &bound_le);
        let (rfound, rhits) = &reference[i];
        if found != *rfound {
            println!("❌ setup {i} seed={s:#x} : found {found} ≠ ref {rfound}");
            nerr += 1;
            continue;
        }
        if (found as usize) <= max_hits && canon(&hits) != *rhits {
            println!("❌ setup {i} seed={s:#x} : ensembles de hits différents (found={found})");
            nerr += 1;
        }
    }
    if nerr == 0 {
        println!("\n🎉 OVERLAP A/B OK — {nsetups} setups : found + hits IDENTIQUES au chemin classique");
    } else {
        println!("\n❌ OVERLAP A/B : {nerr} divergence(s)");
        std::process::exit(2);
    }
}
