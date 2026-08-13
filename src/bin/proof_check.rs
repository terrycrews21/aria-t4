//! Aria GPU miner v2.0 — test de la preuve Merkle FULL GPU (à exécuter sur le GPU de PC3).
//!
//! Mine un hit 2×64 via le pipeline résident, puis construit la `PlainProof` de DEUX façons :
//!   1. CPU (`build_proof_from_hit`)  — la référence déjà validée.
//!   2. GPU (`build_proof_from_hit_gpu`) — la nouvelle voie full-GPU (hash_leaves + parent_level).
//! Vérifie :
//!   - les deux preuves sont OCTET-POUR-OCTET identiques (bincode)  → arbre GPU bit-exact ;
//!   - la preuve GPU passe `parse_plain_proof` + `verify_plain_proof` officiels.
//! Build : cargo build --release --features gpu --bin proof_check
use ariaminer::gpu_ffi::{ProofGpuCtx, ResidentCtx};
use ariaminer::official_grind::{
    alphapool_config_2x64, build_proof_from_hit, build_proof_from_hit_gpu, grind_only_ctx, Workspace,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use zk_pow::api::proof::IncompleteBlockHeader;

fn main() {
    // Taille PRODUCTION : m=n=8192, k=4096 → arbres de 32768 feuilles (33 Mo chacun), comme en live.
    let (m, n, k) = (8192usize, 8192usize, 4096usize);
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [1u8; 32],
        merkle_root: [2u8; 32],
        timestamp: 0x6666_6666,
        nbits: 0x207f_ffff, // difficulté facile → hit immédiat
    };
    let config = alphapool_config_2x64(k as u32);
    let rank = config.rank as usize;
    let bound = zk_pow::api::sanity_checks::extract_difficulty_bound(header.nbits, &config);
    let mut bound_le = [0u8; 32];
    bound.to_little_endian(&mut bound_le);

    println!("▶ grind un hit 2×64 (m={m} n={n} k={k}, config pool [0,32]×[0..63]) …");
    let ctx = ResidentCtx::new_2x64(m, n, k, 64);
    let mut rng = StdRng::seed_from_u64(0x2000_5080_FACE);
    let (setup_seed, hit, job_key) = loop {
        if let Some(t) = grind_only_ctx(&ctx, &mut rng, &header, &config, &bound_le) {
            break t;
        }
    };
    println!(
        "  hit : rows={:?} cols={} colonnes  (setup_seed={:#x})",
        hit.rows,
        hit.cols.len(),
        setup_seed
    );

    // 1) preuve CPU (référence) — chrono (regen 67M int7 + 2 arbres blake3 33 Mo)
    let mut ws = Workspace::new();
    let t0 = std::time::Instant::now();
    let proof_cpu = build_proof_from_hit(setup_seed, &hit, &job_key, m, n, k, rank, &mut ws);
    let dt_cpu = t0.elapsed();

    // 2) preuve GPU (full-GPU) — chrono. 1er appel = warm-up (alloc), on mesure le 2e.
    let pctx = ProofGpuCtx::new(m.max(n), k, 256);
    let _ = build_proof_from_hit_gpu(&pctx, setup_seed, &hit, &job_key, m, n, k, rank);
    let t1 = std::time::Instant::now();
    let proof_gpu = build_proof_from_hit_gpu(&pctx, setup_seed, &hit, &job_key, m, n, k, rank);
    let dt_gpu = t1.elapsed();
    println!("  ⏱  build preuve : CPU {:.1} ms · GPU {:.1} ms", dt_cpu.as_secs_f64()*1e3, dt_gpu.as_secs_f64()*1e3);

    // --- comparaison bit-exact CPU vs GPU ---
    let bc = bincode::serialize(&proof_cpu).unwrap();
    let bg = bincode::serialize(&proof_gpu).unwrap();
    if bc == bg {
        println!("  ✅ preuve GPU == preuve CPU OCTET-POUR-OCTET ({} octets)", bg.len());
    } else {
        println!("  ❌ DIVERGENCE GPU vs CPU : {} vs {} octets", bg.len(), bc.len());
        // diagnostic root par matrice
        println!("    root A  cpu={} gpu={}", hex(&proof_cpu.a.proof.root), hex(&proof_gpu.a.proof.root));
        println!("    root Bt cpu={} gpu={}", hex(&proof_cpu.bt.proof.root), hex(&proof_gpu.bt.proof.root));
        println!("    siblings A  cpu={} gpu={}", proof_cpu.a.proof.siblings.len(), proof_gpu.a.proof.siblings.len());
        println!("    siblings Bt cpu={} gpu={}", proof_cpu.bt.proof.siblings.len(), proof_gpu.bt.proof.siblings.len());
        std::process::exit(1);
    }

    // --- vérification officielle de la preuve GPU ---
    if let Err(e) = ariaminer::official_proof::parse_plain_proof(header, &proof_gpu) {
        println!("  ❌ parse_plain_proof (GPU) ÉCHOUE : {e:?}");
        std::process::exit(2);
    }
    println!("  ✅ parse_plain_proof (GPU) OK (roots == hash_a/hash_b)");

    match zk_pow::api::verify::verify_plain_proof(&header, &proof_gpu, None, zk_pow::api::proof::SeedDerivation::Salted) {
        Ok(_) => println!("\n🎉 PREUVE FULL GPU BIT-EXACTE — verify_plain_proof OK (Aria GPU miner v2.0)"),
        Err(e) => {
            println!("\n❌ verify_plain_proof (GPU) ÉCHOUE : {e:?}");
            std::process::exit(3);
        }
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
