//! ÉTAPE 2 — test bit-exact du kernel GPU 2×64 (à exécuter sur le GPU de PC3).
//! Mine une share 2×64 via le kernel GPU, puis vérifie qu'elle passe les vérifieurs
//! officiels (`parse_plain_proof` + `verify_plain_proof`). Si OK → le jackpot GPU
//! sur tuile 2 lignes {r,r+32} × 64 cols est exact.
//! Build : cargo build --release --features gpu --bin test_2x64
use ariaminer::gpu_ffi::ResidentCtx;
use ariaminer::official_grind::{alphapool_config_2x64, try_mine_one_bounded_gpu_resident_ctx, Workspace};
use rand::rngs::StdRng;
use rand::SeedableRng;
use zk_pow::api::proof::IncompleteBlockHeader;

fn main() {
    let (m, n, k) = (256usize, 128usize, 4096usize);
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [1u8; 32],
        merkle_root: [2u8; 32],
        timestamp: 0x6666_6666,
        nbits: 0x207f_ffff, // difficulté facile → hit immédiat
    };
    let config = alphapool_config_2x64(k as u32);
    let bound = zk_pow::api::sanity_checks::extract_difficulty_bound(header.nbits, &config);
    let mut bound_le = [0u8; 32];
    bound.to_little_endian(&mut bound_le);

    println!("▶ mine une share 2×64 sur le GPU — chemin RÉSIDENT (m={m} n={n} k={k}, config pool [0,32]×[0..63]) …");
    let ctx = ResidentCtx::new_2x64(m, n, k, 64);
    let mut ws = Workspace::new();
    let mut rng = StdRng::seed_from_u64(0x2064_5080_FACE);
    let proof = loop {
        if let Some(p) = try_mine_one_bounded_gpu_resident_ctx(&ctx, &mut ws, &mut rng, &header, &config, &bound_le) {
            break p;
        }
    };
    println!("  preuve 2×64 obtenue, vérification officielle …");

    if let Err(e) = zk_pow::ffi::plain_proof::parse_plain_proof(header, &proof) {
        println!("  ❌ parse_plain_proof ÉCHOUE : {e:?}");
        std::process::exit(1);
    }
    println!("  ✅ parse_plain_proof OK (roots == hash_a/hash_b)");

    match zk_pow::api::verify::verify_plain_proof(&header, &proof) {
        Ok(_) => println!("\n🎉 ÉTAPE 2 RÉUSSIE — jackpot GPU 2×64 BIT-EXACT (verify_plain_proof OK)"),
        Err(e) => {
            println!("\n❌ verify_plain_proof ÉCHOUE : {e:?}");
            std::process::exit(2);
        }
    }
}
