//! Capture les patterns (rows/cols normalisés) que le kernel MULTISTAGE TMA bN=256
//! produit réellement, pour les faire correspondre à canonical_gpu_config (→ job_key match).
//! Lancer : ARIA_TMA_MS=1 ARIA_RANK=256 ./tma_coords_dump
use ariaminer::gpu_ffi::ResidentCtx;
use ariaminer::official_grind::{canonical_gpu_config, compute_job_key_pub};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use zk_pow::api::proof::IncompleteBlockHeader;

fn norm(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v.dedup();
    let base = v[0];
    v.iter().map(|x| x - base).collect()
}

fn main() {
    unsafe { std::env::set_var("ARIA_FIXB", "1") };
    let (m, n, k) = (8192usize, 8192usize, 4096usize);
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
    let jk = compute_job_key_pub(&header, &config);

    let ctx = ResidentCtx::new(m, n, k, 64);
    let mut rng = StdRng::seed_from_u64(0x70A_C0DE);
    println!("config canonical : rows={:?}", config.rows_pattern.to_list());
    println!("                   cols={:?}", config.cols_pattern.to_list());
    println!("▶ grind multistage (ARIA_TMA_MS={:?}) …", std::env::var("ARIA_TMA_MS").ok());
    for attempt in 0..200 {
        let (found, hits) = ctx.grind(rng.next_u64(), &jk, &bound_le);
        if found == 0 { continue; }
        for h in &hits {
            let nr = norm(h.rows.clone());
            let nc = norm(h.cols.clone());
            println!("HIT (attempt {attempt}): {} rows × {} cols", nr.len(), nc.len());
            println!("  rows normalisés = {nr:?}");
            println!("  cols normalisés = {nc:?}");
            println!("  rows bruts (triés) = {:?}", { let mut r=h.rows.clone(); r.sort_unstable(); r.dedup(); r });
            println!("  cols bruts (triés) = {:?}", { let mut c=h.cols.clone(); c.sort_unstable(); c.dedup(); c });
        }
        return;
    }
    println!("aucun hit en 200 tentatives");
}
