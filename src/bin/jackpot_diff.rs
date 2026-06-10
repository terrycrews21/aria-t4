//! DIAGNOSTIC FOLD : compare le transcript[16] + jackpot hash que le kernel
//! multistage calcule vs le compute_jackpot OFFICIEL, pour UN hit déterministe.
//! Forme minimale m=128,n=256 = 1 seul CTA = 1 tuile = 1 hit (slot 0 = ce hit).
//! Le kernel printe son transcript via -DARIA_DUMP_TRANSCRIPT (build.rs).
//! Lancer : ARIA_TMA_MS=1 ARIA_RANK=256 ./jackpot_diff
use ariaminer::gpu_ffi::ResidentCtx;
use ariaminer::official_grind::{build_proof_from_hit, canonical_gpu_config, grind_only_ctx, Workspace};
use rand::rngs::StdRng;
use rand::SeedableRng;
use zk_pow::api::proof::IncompleteBlockHeader;

fn main() {
    // PAS de fix-B : A=signal(seed,0), B=signal(seed,1), même seed → build_proof_from_hit simple.
    let (m, n, k) = (128usize, 256usize, 4096usize);
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [1u8; 32],
        merkle_root: [2u8; 32],
        timestamp: 0x6666_6666,
        nbits: 0x207f_ffff, // facile → hit en quelques setups (1 tuile/setup)
    };
    let config = canonical_gpu_config(k as u32);
    let rank = config.rank as usize;
    println!("forme {m}×{n}×{k} rank={rank} cols={:?}", config.cols_pattern.to_list());
    let bound = zk_pow::api::sanity_checks::extract_difficulty_bound(header.nbits, &config);
    let mut bound_le = [0u8; 32];
    bound.to_little_endian(&mut bound_le);

    let ctx = ResidentCtx::new(m, n, k, 64);
    let mut rng = StdRng::seed_from_u64(0x1ACC_0123);
    println!("▶ grind 1 hit (le kernel printe KERNEL transcript/cv)…");
    let (seed, hit, jk) = loop {
        if let Some(t) = grind_only_ctx(&ctx, &mut rng, &header, &config, &bound_le) { break t; }
    };
    println!("  seed={seed:#x} rows={:?} cols={:?}", { let mut r=hit.rows.clone(); r.sort_unstable(); r.dedup(); r },
             { let mut c=hit.cols.clone(); c.sort_unstable(); c.dedup(); c });

    let mut ws = Workspace::new();
    let proof = build_proof_from_hit(seed, &hit, &jk, m, n, k, rank, &mut ws);

    // Recompute OFFICIEL : parse → noise → compute_jackpot (= transcript) → hash.
    use zk_pow::api::proof_utils::{compute_jackpot_hash, CompiledPublicParams};
    use zk_pow::circuit::chip::compute_jackpot;
    use zk_pow::circuit::pearl_noise::compute_noise;
    match zk_pow::ffi::plain_proof::parse_plain_proof(header, &proof) {
        Err(e) => println!("❌ parse échoue : {e:?}"),
        Ok((privp, public)) => {
            let compiled = CompiledPublicParams::from(&public);
            let noise = compute_noise(&compiled);
            let transcript = compute_jackpot(&compiled, &privp.s_a, &privp.s_b, &noise);
            let h = compute_jackpot_hash(&transcript, compiled.a_noise_seed());
            print!("OFFICIEL    transcript=");
            for w in &transcript { print!("{:08x} ", w); }
            print!("\nOFFICIEL    cv=");
            for i in 0..8 { print!("{:08x} ", u32::from_le_bytes([h[4*i],h[4*i+1],h[4*i+2],h[4*i+3]])); }
            println!("\n→ compare aux lignes KERNEL ci-dessus (transcript = fold avant blake3).");
        }
    }
}
