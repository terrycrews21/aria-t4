//! Production-shape repro: grind on the GPU at a REAL (tight) bound, then recompute
//! the jackpot from the packaged proof exactly as the verifier does.
//!
//! The GPU only reports a hit when ITS jackpot <= bound. If the recomputed jackpot
//! from the packaged proof is far above that bound, the data the GPU folded differs
//! from the data the proof opens (noise / signal / seed mismatch).
//!
//! Build: cargo build --release --features gpu --bin prod_repro
//! Run:   REPRO_DIFF=5000 REPRO_M=8192 prod_repro
use ariaminer::gpu_ffi::{ProofGpuCtx, ResidentCtx};
use ariaminer::official_grind::{
    build_proofs_from_setup_gpu_fixb, canonical_gpu_config, compute_job_key_pub,
};
use ariaminer::stratum_to_official::{scaled_bound_le_from_target_be, share_bound_le};
use zk_pow::api::proof::IncompleteBlockHeader;
use zk_pow::api::proof_utils::{compute_jackpot_hash, CompiledPublicParams};

/// log2 of a little-endian 256-bit integer.
fn log2_le(v: &[u8; 32]) -> f64 {
    let mut acc = 0.0f64;
    for i in (0..32).rev() {
        if v[i] != 0 {
            acc = (v[i] as f64) * 256f64.powi(i as i32);
            if i > 0 {
                acc += (v[i - 1] as f64) * 256f64.powi(i as i32 - 1);
            }
            break;
        }
    }
    if acc == 0.0 { f64::NEG_INFINITY } else { acc.log2() }
}

/// `a <= b` for little-endian 256-bit integers.
fn le_leq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    true
}

fn main() {
    let difficulty: u64 = std::env::var("REPRO_DIFF").ok().and_then(|s| s.parse().ok()).unwrap_or(5_000);
    let m: usize = std::env::var("REPRO_M").ok().and_then(|s| s.parse().ok()).unwrap_or(8192);
    let (n, k) = (m, 4096usize);
    let config = canonical_gpu_config(k as u32);
    let rank = config.rank as usize;
    let tile_h = config.rows_pattern.to_list().len();
    let tile_w = config.cols_pattern.to_list().len();
    let dpl = k - k % rank;
    let factor = (tile_h * tile_w * dpl) as u64;

    // Real LuckyPool notify target (big-endian 256-bit), captured from the wire.
    let thex = std::env::var("REPRO_TARGET")
        .unwrap_or_else(|_| "000000000000218dcdb37c99ae924f227d028a1dfb9389b52007dd441355475a".to_string());
    let tb = (0..32).map(|i| u8::from_str_radix(&thex[i * 2..i * 2 + 2], 16).unwrap()).collect::<Vec<u8>>();
    let mut target_be = [0u8; 32];
    target_be.copy_from_slice(&tb);
    let _ = share_bound_le(difficulty);
    let bound_le = scaled_bound_le_from_target_be(&target_be, factor);

    println!("== production repro ==");
    println!("  m=n={m} k={k} rank={rank} tile={tile_h}x{tile_w} dpl={dpl}");
    println!("  difficulty={difficulty} factor={factor}");
    println!("  mining bound = 2^{:.2}", log2_le(&bound_le));

    // Replay a REAL pool job when REPRO_HEADER is given (80-byte hex from the wire).
    let header = match std::env::var("REPRO_HEADER") {
        Ok(h) => {
            let bytes = (0..h.len() / 2)
                .map(|i| u8::from_str_radix(&h[i * 2..i * 2 + 2], 16).unwrap())
                .collect::<Vec<u8>>();
            let hd = IncompleteBlockHeader::from_bytes(&bytes).expect("header parse");
            println!("  replaying real pool header ({} bytes) nbits={:#x}", bytes.len(), hd.nbits);
            hd
        }
        Err(_) => IncompleteBlockHeader {
            version: 0,
            prev_block: [1u8; 32],
            merkle_root: [2u8; 32],
            timestamp: 0x6666_6666,
            nbits: 0x1d00_ffff,
        },
    };
    let rotate: u32 = std::env::var("REPRO_ROTATE").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let mut header = header;
    let mut job_key = compute_job_key_pub(&header, &config);

    println!("\n> allocating grind + proof contexts …");
    let ctx = ResidentCtx::new(m, n, k, 64);
    let ctx_a = ProofGpuCtx::new(m, k, 256);
    let ctx_b = ProofGpuCtx::new(n, k, 256);

    // Determinism probe: grind the SAME seed twice and compare the commitment root.
    // Differing roots => race inside the commit kernel; identical-but-wrong => a
    // deterministic defect (e.g. reading memory it does not own).
    {
        let probe_seed: u64 = 0x1234_5678_9abc_def0;
        let mut h1 = vec![0i8; 32];
        let mut h2 = vec![0i8; 32];
        ctx.set_b_seed(probe_seed);
        let _ = ctx.grind(probe_seed, &job_key, &bound_le);
        ctx.dump_row(5, 0, &mut h1);
        let _ = ctx.grind(probe_seed, &job_key, &bound_le);
        ctx.dump_row(5, 0, &mut h2);
        let hx = |v: &[i8]| v.iter().map(|b| format!("{:02x}", *b as u8)).collect::<String>();
        println!("  [determinism] hash_a run1 = {}", hx(&h1));
        println!("  [determinism] hash_a run2 = {}", hx(&h2));
        println!("  [determinism] identical across identical grinds: {}", h1 == h2);
        // Same seed, FIXED tile: the proof context regenerates A/B independently of the
        // grind. If its strips + root match across machines while the grind's root does
        // not, the defect is isolated to the grind's commit kernel.
        let rows_pat: Vec<usize> = config.rows_pattern.to_list().iter().map(|v| *v as usize).collect();
        let cols_pat: Vec<usize> = config.cols_pattern.to_list().iter().map(|v| *v as usize).collect();
        // Hit carries the deduplicated indices: tile_h rows and tile_w columns.
        let phit = ariaminer::gpu_ffi::Hit { rows: rows_pat.clone(), cols: cols_pat.clone() };
        let pps = build_proofs_from_setup_gpu_fixb(
            &ctx_a, &ctx_b, probe_seed, probe_seed, std::slice::from_ref(&phit),
            &job_key, m, n, k, rank, tile_h, tile_w,
        );
        println!("  [determinism] probe proofs built: {}", pps.len());
        for pp in &pps {
            if let Ok((pr, pu)) = zk_pow::ffi::plain_proof::parse_plain_proof(header, pp) {
                println!("  [determinism] proof-side hash_a = {}", hex::encode(pu.hash_a));
                println!("  [determinism] proof-side hash_b = {}", hex::encode(pu.hash_b));
                println!("  [determinism] signal strip[0][0..8] = {:?}", &pr.s_a[0][..8]);
            } else {
                println!("  [determinism] probe proof did not parse");
            }
        }
    }

    let fixb = std::env::var("ARIA_FIXB").is_ok();
    let mut b_seed_job: Option<u64> = None;
    println!("  fix-B = {fixb}");
    let want_hits: u32 = std::env::var("REPRO_HITS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let (mut n_inside, mut n_above) = (0u32, 0u32);
    let mut seed: u64 = std::env::var("REPRO_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0x0420_5080_FACE);
    let forced_b: Option<u64> = std::env::var("REPRO_BSEED").ok().and_then(|s| s.parse().ok());
    for attempt in 1..=2000u32 {
        // EXACT production contract (bin/ariaminer.rs): frozen B seed under fix-B.
        let b_seed = match forced_b {
            Some(b) => b,
            None => if fixb { *b_seed_job.get_or_insert(seed) } else { seed },
        };
        ctx.set_b_seed(b_seed);
        let next_seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let (found, hits) = ctx.grind2(seed, next_seed, &job_key, &bound_le);
        if found > 0 {
            if let Some(hit) = hits.into_iter().find(|h| h.rows.len() == tile_h && h.cols.len() == tile_w) {
                println!("\n> GPU hit on attempt {attempt} (seed={seed:#x})");
                println!("  rows = {:?}", hit.rows);
                println!("  cols[0..4] = {:?}", &hit.cols[..4.min(hit.cols.len())]);
                let proofs = build_proofs_from_setup_gpu_fixb(
                    &ctx_a, &ctx_b, seed, b_seed, std::slice::from_ref(&hit),
                    &job_key, m, n, k, rank, tile_h, tile_w,
                );
                println!("  packaged {} proof(s)", proofs.len());
                for (i, p) in proofs.iter().enumerate() {
                    match zk_pow::ffi::plain_proof::parse_plain_proof(header, p) {
                        Err(e) => println!("  proof {i}: parse FAILED: {e:?}"),
                        Ok((priv_p, publ)) => {
                            let compiled = CompiledPublicParams::from(&publ);
                            let noise = zk_pow::circuit::pearl_noise::compute_noise(&compiled);
                            let jp = zk_pow::circuit::chip::compute_jackpot(
                                &compiled, &priv_p.s_a, &priv_p.s_b, &noise,
                            );
                            let hv = compute_jackpot_hash(&jp, compiled.commitment_hash.1);
                            let verdict = if le_leq(&hv, &bound_le) {
                                "INSIDE bound — GPU and proof agree"
                            } else {
                                "ABOVE bound — GPU folded different data than the proof opens"
                            };
                            if le_leq(&hv, &bound_le) { n_inside += 1; } else { n_above += 1; }
                            {
                                // LOCALISE the divergence: device row = signal + device noise.
                                // The proof strips are the pure signal, so (device - strip) is
                                // exactly the noise the GPU folded. Compare it with the noise
                                // the verifier derives from the same commitment.
                                let mut devrow = vec![0i8; k];
                                ctx.dump_row(0, hit.rows[0], &mut devrow);
                                let strip = &priv_p.s_a[0];
                                let nz = &noise.a[0];
                                let (mut sig_bad, mut noise_bad, mut first) = (0usize, 0usize, usize::MAX);
                                for j in 0..k {
                                    let expect = (strip[j] as i32 + nz[j] as i32).clamp(-128, 127) as i8;
                                    if devrow[j] != expect {
                                        noise_bad += 1;
                                        if first == usize::MAX { first = j; }
                                    }
                                    if devrow[j] == strip[j] && nz[j] != 0 { sig_bad += 1; }
                                }
                                println!("    [probe] row {} : device!=signal+noise at {}/{} cols (first={}), device==signal-only at {} cols",
                                    hit.rows[0], noise_bad, k, if first == usize::MAX { -1 } else { first as i64 }, sig_bad);
                                println!("    [probe] device[0..8] = {:?}", &devrow[..8]);
                                println!("    [probe] strip [0..8] = {:?}", &strip[..8]);
                                println!("    [probe] noise [0..8] = {:?}", &nz[..8]);
                                // Is the device noise the SAME multiset, just ordered
                                // differently? That would pin the bug on `perm_k_`
                                // (the per-k permutation) rather than on noise values.
                                let dev_noise: Vec<i32> = (0..k)
                                    .map(|j| devrow[j] as i32 - strip[j] as i32)
                                    .collect();
                                let circ_noise: Vec<i32> = (0..k).map(|j| nz[j] as i32).collect();
                                let (mut ds, mut cs) = (dev_noise.clone(), circ_noise.clone());
                                ds.sort_unstable();
                                cs.sort_unstable();
                                println!("    [probe] same multiset (device vs circuit noise): {}", ds == cs);
                                // Rotation?
                                let mut rot = None;
                                for r in 0..k {
                                    if (0..k).all(|j| dev_noise[j] == circ_noise[(j + r) % k]) { rot = Some(r); break; }
                                }
                                println!("    [probe] rotation offset: {:?}", rot);
                                // Block-permutation within rank-sized blocks?
                                let nblk = k / rank;
                                let mut blk_map: Vec<i64> = Vec::new();
                                for b in 0..nblk.min(8) {
                                    let seg = &dev_noise[b * rank..(b + 1) * rank];
                                    let mut found: i64 = -1;
                                    for c in 0..nblk {
                                        if seg == &circ_noise[c * rank..(c + 1) * rank] { found = c as i64; break; }
                                    }
                                    blk_map.push(found);
                                }
                                println!("    [probe] rank-block map (device blk -> circuit blk, first 8): {:?}", blk_map);
                            }
                            if !le_leq(&hv, &bound_le) {
                                // The GPU says this tile is under the bound but the opened
                                // strips say otherwise. If the emission is merely OFFSET,
                                // some neighbouring tile reproduces the winning jackpot.
                                // Sweep row/col shifts and report any candidate that lands.
                                let shifts: [i64; 9] = [-64, -48, -32, -16, 0, 16, 32, 48, 64];
                                let mut found_shift = false;
                                for &dr in shifts.iter() {
                                    for &dc in shifts.iter() {
                                        if dr == 0 && dc == 0 { continue; }
                                        let mut h2 = hit.clone();
                                        let mut ok = true;
                                        for r in h2.rows.iter_mut() {
                                            let v = *r as i64 + dr;
                                            if v < 0 || v >= m as i64 { ok = false; break; }
                                            *r = v as i32;
                                        }
                                        if !ok { continue; }
                                        for c in h2.cols.iter_mut() {
                                            let v = *c as i64 + dc;
                                            if v < 0 || v >= n as i64 { ok = false; break; }
                                            *c = v as i32;
                                        }
                                        if !ok { continue; }
                                        let ps = build_proofs_from_setup_gpu_fixb(
                                            &ctx_a, &ctx_b, seed, b_seed, std::slice::from_ref(&h2),
                                            &job_key, m, n, k, rank, tile_h, tile_w,
                                        );
                                        for p2 in &ps {
                                            if let Ok((pp, pu)) = zk_pow::ffi::plain_proof::parse_plain_proof(header, p2) {
                                                let cp = CompiledPublicParams::from(&pu);
                                                let nz = zk_pow::circuit::pearl_noise::compute_noise(&cp);
                                                let j2 = zk_pow::circuit::chip::compute_jackpot(&cp, &pp.s_a, &pp.s_b, &nz);
                                                let h2v = compute_jackpot_hash(&j2, cp.commitment_hash.1);
                                                if le_leq(&h2v, &bound_le) {
                                                    println!("    >>> SHIFT MATCH dr={dr} dc={dc}: jackpot 2^{:.2} INSIDE", log2_le(&h2v));
                                                    found_shift = true;
                                                }
                                            }
                                        }
                                    }
                                }
                                if !found_shift {
                                    println!("    (no neighbouring tile reproduces the hit — not a pure index offset)");
                                }
                            }
                            println!("  proof {i}: recomputed jackpot = 2^{:.2}  (bound 2^{:.2})  => {verdict}",
                                log2_le(&hv), log2_le(&bound_le));
                        }
                    }
                }
                if n_inside + n_above >= want_hits {
                    println!("\n== summary: {n_inside} inside / {n_above} ABOVE out of {} ==", n_inside + n_above);
                    return;
                }
            }
        }
        if attempt % 50 == 0 {
            println!("  … {attempt} attempts, no hit yet");
        }
        // Pool behaviour: a fresh notify every ~30 s -> new job_key, and the miner
        // releases the frozen fix-B seed exactly here.
        if rotate > 0 && attempt % rotate == 0 {
            header.timestamp = header.timestamp.wrapping_add(1);
            job_key = compute_job_key_pub(&header, &config);
            b_seed_job = None;
            println!("  [job rotate at attempt {attempt}]");
        }
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    }
    println!("\n== summary: {n_inside} inside / {n_above} ABOVE ==");
}
