//! Live GPU mining proof-of-life (feature `gpu`).
//! Boucle continue : grind GPU → sur hit, VÉRIFIE la preuve contre le verify
//! officiel (`parse_plain_proof` + `verify_plain_proof`) puis logue. Affiche le
//! débit (attempts/s, candidats, shares vérifiées). C'est la preuve, en direct,
//! que le 5080 produit des PlainProof Pearl valides en continu.
//!
//!   cargo run --release --features gpu --bin gpu_live -- [m] [n] [k] [nbits_hex]
//!
//! nbits par défaut = 0x207fffff (regtest facile → bound saturé, chaque attempt
//! donne des candidats : flux rapide pour suivre le live). Mettre un nbits plus
//! dur (ex 0x1f00ffff) pour voir le grind chercher.
use ariaminer::official_grind::{Workspace, canonical_gpu_config, try_mine_one_bounded_gpu};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;
use zk_pow::api::proof::IncompleteBlockHeader;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let k: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4032);
    let nbits = a
        .get(4)
        .map(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).expect("nbits hex"))
        .unwrap_or(0x207f_ffff);

    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [1u8; 32],
        merkle_root: [2u8; 32],
        timestamp: 0x6666_6666,
        nbits,
    };
    let config = canonical_gpu_config(k as u32);
    let bound = zk_pow::api::sanity_checks::extract_difficulty_bound(nbits, &config);
    let mut bound_le = [0u8; 32];
    bound.to_little_endian(&mut bound_le);

    println!(
        "▶ gpu_live  m={m} n={n} k={k}  nbits={nbits:#010x}  tile=8x16 (canonical)\n\
         ▶ vérification officielle (parse + verify_plain_proof) sur CHAQUE share. Ctrl-C pour stopper.\n"
    );

    let mut ws = Workspace::new();
    let mut rng = StdRng::seed_from_u64(0x5080_AB1A_C0FFEE);
    let t0 = Instant::now();
    let mut attempts: u64 = 0;
    let mut shares: u64 = 0;
    let mut verified: u64 = 0;
    let mut last_log = Instant::now();

    loop {
        attempts += 1;
        if let Some(proof) =
            try_mine_one_bounded_gpu(&mut ws, &mut rng, m, n, k, &header, &config, &bound_le)
        {
            shares += 1;
            // Vérification officielle node-side, en direct.
            let parsed = ariaminer::official_proof::parse_plain_proof(header, &proof).is_ok();
            let ok = parsed && zk_pow::api::verify::verify_plain_proof(&header, &proof, None, zk_pow::api::proof::SeedDerivation::Salted).is_ok();
            if ok {
                verified += 1;
            }
            let rows0 = proof.a.row_indices.first().copied().unwrap_or(0);
            let cols0 = proof.bt.row_indices.first().copied().unwrap_or(0);
            println!(
                "  [{:>7.1}s] share #{shares}  tile@(r0={rows0},c0={cols0})  verify={}  (attempt {attempts})",
                t0.elapsed().as_secs_f64(),
                if ok { "✅ VALID" } else { "❌ REJET" }
            );
        }
        if last_log.elapsed().as_secs_f64() >= 2.0 {
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "  ── {:>7.1}s | attempts={attempts} ({:.1}/s) | shares={shares} | vérifiées={verified}/{shares}",
                dt,
                attempts as f64 / dt
            );
            last_log = Instant::now();
        }
    }
}
