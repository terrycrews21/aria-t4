//! Aria GPU miner v0.4.2 — validation de la forme 131072 (comme alpha) avec preuve full GPU
//! "arbre une fois par setup". À exécuter sur un GPU LIBRE (PC3).
//!
//! 1. grind un setup à m=n=131072 (grille 1024×1024 CTAs, ~768 ms) → liste de hits.
//! 2. bâtit les 2 arbres (A, B = 536 Mo chacun) UNE fois, puis gather chaque hit.
//! 3. vérifie que les preuves passent parse_plain_proof + verify_plain_proof (le juge officiel).
//!    (La bit-exactitude vs CPU est déjà prouvée à 8192 ; à 131072 la regen CPU = ~4 Go RAM,
//!     on s'appuie donc sur le vérifieur du node, qui recalcule les racines depuis la preuve.)
//! Build : cargo build --release --features gpu --bin proof_check_131k
use ariaminer::gpu_ffi::{ProofGpuCtx, ResidentCtx};
use ariaminer::official_grind::{alphapool_config_2x64, build_proofs_from_setup_gpu};
use zk_pow::api::proof::IncompleteBlockHeader;

fn main() {
    let (m, n, k) = (131072usize, 131072usize, 4096usize);
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [1u8; 32],
        merkle_root: [2u8; 32],
        timestamp: 0x6666_6666,
        nbits: 0x207f_ffff, // diff facile → beaucoup de hits immédiats
    };
    let config = alphapool_config_2x64(k as u32);
    let rank = config.rank as usize;
    let bound = zk_pow::api::sanity_checks::extract_difficulty_bound(header.nbits, &config);
    let mut bound_le = [0u8; 32];
    bound.to_little_endian(&mut bound_le);
    let job_key = ariaminer::official_grind::compute_job_key_pub(&header, &config);

    println!("▶ alloc grind 131072² (A,B = 536 Mo chacun) + 2 ctx de preuve …");
    let ctx = ResidentCtx::new_2x64(m, n, k, 256);
    let ctx_a = ProofGpuCtx::new(m, k, 256);
    let ctx_b = ProofGpuCtx::new(n, k, 256);

    println!("▶ grind 1 setup (grille 1024×1024 CTAs, ~768 ms) …");
    let setup_seed = 0x0420_5080_FACEu64;
    let t0 = std::time::Instant::now();
    let (found, hits) = ctx.grind(setup_seed, &job_key, &bound_le);
    println!("  setup en {:.0} ms · found={found} · hits retournés={}", t0.elapsed().as_secs_f64() * 1e3, hits.len());
    let valid: Vec<_> = hits.into_iter().filter(|h| h.rows.len() == 2 && h.cols.len() == 64).collect();
    println!("  hits 2×64 valides={}", valid.len());
    if valid.is_empty() {
        println!("  ❌ aucun hit 2×64 — diff trop dure ?");
        std::process::exit(1);
    }
    // borne le nombre de preuves construites pour le test (les premières suffisent)
    let test_hits: Vec<_> = valid.into_iter().take(4).collect();
    println!("  exemple hit[0] : rows={:?} cols[0..3]={:?}…", test_hits[0].rows, &test_hits[0].cols[..3.min(test_hits[0].cols.len())]);

    println!("▶ build preuves (2 arbres bâtis 1× puis gather/hit) …");
    let t1 = std::time::Instant::now();
    let proofs = build_proofs_from_setup_gpu(&ctx_a, &ctx_b, setup_seed, &test_hits, &job_key, m, n, k, rank);
    println!("  {} preuves en {:.0} ms (arbres + gathers)", proofs.len(), t1.elapsed().as_secs_f64() * 1e3);

    let mut ok = 0;
    for (i, p) in proofs.iter().enumerate() {
        if let Err(e) = zk_pow::ffi::plain_proof::parse_plain_proof(header, p) {
            println!("  ❌ preuve {i} : parse_plain_proof échoue : {e:?}");
            std::process::exit(2);
        }
        match zk_pow::api::verify::verify_plain_proof(&header, p) {
            Ok(_) => ok += 1,
            Err(e) => {
                println!("  ❌ preuve {i} : verify_plain_proof échoue : {e:?}");
                std::process::exit(3);
            }
        }
    }
    println!("\n🎉 v0.4.2 FORME 131072 OK — {ok}/{} preuves passent verify_plain_proof (full GPU, arbre une fois)", proofs.len());
}
