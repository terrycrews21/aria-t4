//! proof_check FIX-B (10/06) — valide le consensus du mode "fixer B, streamer A".
//! grind S0 (remplit B) → grind S1 (A varie, B gardé=S0) → preuve CROISÉE (a=int7(S1,0),
//! b=int7(S0,1)) → parse + verify_plain_proof OFFICIELS. Si ça passe : fix-B est consensus-légal.
//! Build : cargo build --release --features gpu --bin proof_check_fixb
use ariaminer::gpu_ffi::ResidentCtx;
use ariaminer::official_grind::{build_proof_from_hit_fixb, canonical_gpu_config, grind_only_ctx, Workspace};
use rand::rngs::StdRng;
use rand::SeedableRng;
use zk_pow::api::proof::IncompleteBlockHeader;

fn main() {
    unsafe { std::env::set_var("ARIA_FIXB", "1"); } // active fix-B dans resident_run (kernel C++)
    // Forme paramétrable (ARIA_BATCH_M/N) pour valider aussi la forme officielle 131072².
    let dim = |v: &str, d: usize| std::env::var(v).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let (m, n, k) = (dim("ARIA_BATCH_M", 8192), dim("ARIA_BATCH_N", 8192), 4096usize);
    println!("  forme {m}×{n}×{k}");
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [1u8; 32],
        merkle_root: [2u8; 32],
        timestamp: 0x6666_6666,
        nbits: 0x207f_ffff, // difficulté facile → hit immédiat
    };
    let config = canonical_gpu_config(k as u32); // 8×16 (AriaPool)
    let rank = config.rank as usize;
    let bound = zk_pow::api::sanity_checks::extract_difficulty_bound(header.nbits, &config);
    let mut bound_le = [0u8; 32];
    bound.to_little_endian(&mut bound_le);

    let ctx = ResidentCtx::new(m, n, k, 64);
    let mut rng = StdRng::seed_from_u64(0x2000_5080_0F1B);

    println!("▶ grind #1 (remplit B avec S0) …");
    let (s0, _h0, _jk0) = loop {
        if let Some(t) = grind_only_ctx(&ctx, &mut rng, &header, &config, &bound_le) { break t; }
    };
    println!("▶ grind #2 (A varie = S1, B gardé = S0) …");
    let (s1, hit1, jk1) = loop {
        if let Some(t) = grind_only_ctx(&ctx, &mut rng, &header, &config, &bound_le) { break t; }
    };
    println!("  S0 (b_seed) = {s0:#x}   S1 (a_seed) = {s1:#x}");
    assert_ne!(s0, s1, "S0 et S1 doivent différer pour un vrai test fix-B");

    // PREUVE CROISÉE : a depuis S1, b depuis S0 (= ce que le kernel fix-B a réellement grindé)
    let mut ws = Workspace::new();
    let proof = build_proof_from_hit_fixb(s1, s0, &hit1, &jk1, m, n, k, rank, &mut ws);

    if let Err(e) = zk_pow::ffi::plain_proof::parse_plain_proof(header, &proof) {
        println!("❌ parse_plain_proof ÉCHOUE : {e:?}");
        std::process::exit(2);
    }
    println!("  ✅ parse_plain_proof OK (roots == hash_a/hash_b croisés)");
    match zk_pow::api::verify::verify_plain_proof(&header, &proof) {
        Ok(_) => println!("\n🎉 FIX-B CONSENSUS-VALIDE — verify_plain_proof OK sur preuve croisée (A=S1 / B=S0 figé)"),
        Err(e) => {
            println!("\n❌ verify_plain_proof ÉCHOUE : {e:?}  → fix-B PAS valide en l'état");
            std::process::exit(3);
        }
    }
}
