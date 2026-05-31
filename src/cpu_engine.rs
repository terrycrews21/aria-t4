//! Amortized CPU mining engine — the production grind wired behind the stratum
//! client. Mirrors the official `try_mine_one` structure: draw a batch of
//! A/B rows once, hash+noise once, then sweep many (row-tile × col-tile)
//! jackpots reusing the noised matrices. Validated at ~460 GMAC/s on a Ryzen 9950X3D.
//!
//! Unlike `pearl_compute::compute_jackpot_flat` (fixed shape per call), this
//! works on a *flat batch* and takes the tile row/col indices at runtime so it
//! can honor the job's `rows_pattern` / `cols_pattern`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::pearl_compute::{JACKPOT_SIZE, LROT_PER_TILE};
use blake3::Hasher;
use parking_lot::Mutex;
use rand::RngCore;
use rand::rngs::StdRng;

/// The whole per-tile jackpot loop, parameterised by the int8 dot `$dot`. The
/// SIMD feature dispatch is hoisted ONCE to the caller (`jackpot_tile_flat`):
/// the inner dot is over only r=128 elements (~2 `vpdpbusd`), so a per-dot
/// `is_x86_feature_detected` + non-inlinable call would erase the VNNI gain.
/// Here the dot folds inline into the same-feature tile fn.
macro_rules! jackpot_tile_body {
    ($a_eff:expr, $b_eff:expr, $a_idx:expr, $b_idx:expr, $k:expr, $r:expr, $jackpot:expr, $dot:expr) => {{
        let (a_eff, b_eff, a_idx, b_idx, k, r, jackpot) =
            ($a_eff, $b_eff, $a_idx, $b_idx, $k, $r, $jackpot);
        let h = a_idx.len();
        let w = b_idx.len();
        jackpot.iter_mut().for_each(|x| *x = 0);
        let mut msg = [0u32; JACKPOT_SIZE];
        let mut ll = r;
        let mut blk = 0usize;
        while ll <= k {
            let l0 = ll - r;
            for (u, &ai) in a_idx.iter().enumerate() {
                let a = &a_eff[ai * k + l0..ai * k + ll];
                for (v, &bi) in b_idx.iter().enumerate() {
                    let b = &b_eff[bi * k + l0..bi * k + ll];
                    let d: i32 = $dot(a, b);
                    jackpot[u * w + v] = jackpot[u * w + v].wrapping_add(d);
                }
            }
            let mut xored = 0u32;
            for &x in jackpot[..h * w].iter() {
                xored ^= x as u32;
            }
            msg[blk % JACKPOT_SIZE] = msg[blk % JACKPOT_SIZE].rotate_left(LROT_PER_TILE) ^ xored;
            ll += r;
            blk += 1;
        }
        msg
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
unsafe fn jackpot_tile_avx512(
    a_eff: &[i8], b_eff: &[i8], a_idx: &[usize], b_idx: &[usize],
    k: usize, r: usize, jackpot: &mut [i32],
) -> [u32; JACKPOT_SIZE] {
    jackpot_tile_body!(a_eff, b_eff, a_idx, b_idx, k, r, jackpot,
        |a: &[i8], b: &[i8]| unsafe { crate::vnni::dot_i8_avx512vnni(a, b) })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn jackpot_tile_avxvnni(
    a_eff: &[i8], b_eff: &[i8], a_idx: &[usize], b_idx: &[usize],
    k: usize, r: usize, jackpot: &mut [i32],
) -> [u32; JACKPOT_SIZE] {
    jackpot_tile_body!(a_eff, b_eff, a_idx, b_idx, k, r, jackpot,
        |a: &[i8], b: &[i8]| unsafe { crate::vnni::dot_i8_avxvnni(a, b) })
}

fn jackpot_tile_scalar(
    a_eff: &[i8], b_eff: &[i8], a_idx: &[usize], b_idx: &[usize],
    k: usize, r: usize, jackpot: &mut [i32],
) -> [u32; JACKPOT_SIZE] {
    jackpot_tile_body!(a_eff, b_eff, a_idx, b_idx, k, r, jackpot,
        |a: &[i8], b: &[i8]| crate::vnni::dot_i8_scalar(a, b))
}

/// One jackpot over a tile selected from the flat noised batch.
/// `a_eff`/`b_eff` are `rows*k` flat i8. `a_idx` (len h) and `b_idx` (len w)
/// pick the tile's rows. Bit-identical to `compute_jackpot_flat`. Dispatches
/// the SIMD path once per tile (not per dot).
#[inline]
pub fn jackpot_tile_flat(
    a_eff: &[i8],
    b_eff: &[i8],
    a_idx: &[usize],
    b_idx: &[usize],
    k: usize,
    r: usize,
    jackpot: &mut [i32],
) -> [u32; JACKPOT_SIZE] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512f")
        {
            return unsafe { jackpot_tile_avx512(a_eff, b_eff, a_idx, b_idx, k, r, jackpot) };
        }
        if is_x86_feature_detected!("avxvnni") {
            return unsafe { jackpot_tile_avxvnni(a_eff, b_eff, a_idx, b_idx, k, r, jackpot) };
        }
    }
    jackpot_tile_scalar(a_eff, b_eff, a_idx, b_idx, k, r, jackpot)
}

/// Static job description the engine grinds against.
#[derive(Clone)]
pub struct JobSpec {
    /// Pool job id (echoed back in the submit).
    pub job_id: String,
    pub k: usize,
    pub r: usize,
    /// tile height = `rows_pattern.len()`.
    pub h: usize,
    /// tile width = `cols_pattern.len()`.
    pub w: usize,
    /// `blake3(header || config)` — seeds the commitment chain.
    pub job_key: [u8; 32],
    /// Network/block target (32-byte BE). `hash < target` ⇒ a real block.
    pub target: [u8; 32],
    /// Pool share target (easier, from `mining.set_difficulty`). `hash < share_target`
    /// ⇒ a share worth submitting.
    pub share_target: [u8; 32],
}

/// Pool share target from stratum difficulty: `floor((2^256 - 1) / difficulty)`,
/// big-endian. Long division of 0xFF…FF (32 bytes) by `difficulty`.
pub fn share_target_from_difficulty(difficulty: u64) -> [u8; 32] {
    let d = difficulty.max(1) as u128;
    let mut t = [0u8; 32];
    let mut rem: u128 = 0;
    for slot in t.iter_mut() {
        rem = (rem << 8) | 0xFF;
        *slot = (rem / d) as u8;
        rem %= d;
    }
    t
}

/// A winning tile. `is_block` flags the rare case where the hash also beats the
/// network target (a real block, not just a pool share). `job_id` is the job
/// live at the moment of the hit (workers adopt new jobs mid-run).
pub struct Hit {
    pub hash: [u8; 32],
    pub is_block: bool,
    pub job_id: String,
}

/// Run the amortized grind for `job` until `stop` is set, on a single thread.
/// `attempts_ctr` is incremented live. `on_hit` fires for each share found.
///
/// `m_batch`/`n_batch` control the amortization window (rows/cols drawn per
/// setup). Batch 1024 was the sweet spot on large-V-cache CPUs (4 MB matrices ⊂ V-cache).
pub fn grind_thread(
    job_slot: &Mutex<Arc<JobSpec>>,
    m_batch: usize,
    n_batch: usize,
    rng: &mut StdRng,
    stop: &AtomicBool,
    attempts_ctr: &AtomicU64,
    mut on_hit: impl FnMut(Hit),
) {
    // k/r are fixed for the chain (Pearl: k=4096, r=128); size buffers once.
    let (k, r) = {
        let j = job_slot.lock();
        (j.k, j.r)
    };
    // Row-panel size for 2D cache-blocking (see the sweep below). Default 192
    // row-groups = 384 A-rows ≈ 1.5 MB at k=4096 — measured sweet spot on a desktop CPU
    // (Ryzen 9950X3D, 96 MB V-cache L3): sweep gave 64→8.16M, 128→8.52M,
    // 192→8.87M (peak), plateau to 512, 1024→8.47M. Tunable per-machine via
    // ARIA_PANEL_RG (i9 285K with no V-cache may prefer a smaller panel).
    let panel_rg: usize = std::env::var("ARIA_PANEL_RG")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(192);
    let mut a = vec![0i8; m_batch * k];
    let mut b = vec![0i8; n_batch * k];
    // Column-interleaved repack of B for the register-blocked micro-kernel
    // (filled per setup only when the fast path applies). See `microkernel`.
    let mut b_packed: Vec<i8> = Vec::new();
    let mut jackpot: Vec<i32> = Vec::new();
    let mut local: u64 = 0;

    while !stop.load(Ordering::Relaxed) {
        // Adopt the latest job at each setup boundary — no worker restart when the
        // pool pushes a new job (which it does every ~0.3s); the in-flight tile
        // sweep finishes against a coherent job, the next setup picks up the new one.
        let job = job_slot.lock().clone();
        let h = job.h.max(1);
        let w = job.w.max(1);
        // 8×64 fast path needs 512 i32 scratch (4 tiles × 128 cells).
        let jp_need = (h * w).max(512);
        if jackpot.len() < jp_need {
            jackpot.resize(jp_need, 0);
        }

        // 1. draw A / B as int7.
        for buf in [&mut a, &mut b] {
            let bytes: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len()) };
            rng.fill_bytes(bytes);
            for c in buf.iter_mut() {
                let raw = (*c as u8) & 0x7f;
                *c = if raw & 0x40 != 0 { (raw as i8) | -0x40i8 } else { raw as i8 };
            }
        }

        // 2. commitment hash + noise seeds (once per setup).
        let ab: &[u8] = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len()) };
        let bb: &[u8] = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len()) };
        let hash_a = {
            let mut h = Hasher::new();
            h.update(&job.job_key);
            h.update(ab);
            *h.finalize().as_bytes()
        };
        let hash_b = {
            let mut h = Hasher::new();
            h.update(&job.job_key);
            h.update(bb);
            *h.finalize().as_bytes()
        };
        let b_seed = {
            let mut h = Hasher::new();
            h.update(&job.job_key);
            h.update(&hash_b);
            *h.finalize().as_bytes()
        };
        let a_seed = {
            let mut h = Hasher::new();
            h.update(&b_seed);
            h.update(&hash_a);
            *h.finalize().as_bytes()
        };

        // 3+4. noise for the whole batch + pre-noise add, FUSED (once).
        // The SIMD (VBMI vpermi2b) matvec writes the noise straight into the
        // drawn strips — the intermediate noise matrices are never allocated,
        // written or read back (saves an 8 MiB alloc + write + read per setup,
        // the dominant memory-traffic cost at small batch). Bit-identical to the
        // old compute_noise + element-wise wrapping_add.
        let a_rows_idx: Vec<usize> = (0..m_batch).collect();
        let b_cols_idx: Vec<usize> = (0..n_batch).collect();
        crate::fast_noise::add_noise_into_fast(
            k,
            r,
            (b_seed, a_seed),
            &a_rows_idx,
            &b_cols_idx,
            &mut a,
            &mut b,
        );

        // 4b. Repack B column-interleaved for the register-blocked kernel when
        // the shape/CPU allow it (h=2, w=64 mainnet on AVX-512 VNNI). The pack
        // is O(n·k) once per setup, amortized over row_groups*col_groups tiles.
        // AVX-512 (16-lane groups) vs AVX-VNNI (8-lane groups).
        // Mutually exclusive; AVX-512 wins when present. The non-micro fallback
        // (flat) still covers any other shape/CPU.
        let use_micro512 = crate::microkernel::micro_kernel_applicable(h, w, n_batch);
        let use_micro_vnni =
            !use_micro512 && crate::microkernel::micro_kernel_avxvnni_applicable(h, w, n_batch);
        if use_micro512 {
            crate::microkernel::pack_b_16(&b, n_batch, k, &mut b_packed);
        } else if use_micro_vnni {
            crate::microkernel::pack_b_8(&b, n_batch, k, &mut b_packed);
        }

        // 5. sweep tiles: contiguous row/col groups (h rows, w cols each).
        // 2D CACHE-BLOCKING. The old order (col-groups outermost) kept B's 256 KB
        // slice L2-resident but re-streamed ALL of A once per col-group: A (m·k =
        // 16 MB @batch4096) read `col_groups` (=64) times ⇒ ~1 GB of A traffic per
        // setup vs 16 MB for B. A dominated the memory bus 64:1 — the real wall at
        // 16t (bandwidth-bound).
        // Fix: tile the ROW axis into panels of `panel_rg` row-groups (≈512 KB of
        // A) and sweep ALL col-groups against the resident panel, so A is read
        // from DRAM/L3 once per panel (`row_groups/panel_rg` ≈ 32 passes total,
        // not 64 per-col-group re-reads), while B's col-group stays L2-resident
        // across the panel's row-blocks. Tiles are independent ⇒ reorder is free;
        // kernel + output stay bit-identical (no hash changes ⇒ shares valid).
        let row_groups = m_batch / h;
        let col_groups = n_batch / w;
        let groups_per_tile16 = w / 16; // packed 16-col groups per tile (AVX-512)
        let groups_per_tile8 = w / 8; //   packed 8-col groups per tile (AVX-VNNI)
        let mut a_idx = vec![0usize; h];
        let mut b_idx = vec![0usize; w];
        let mut panel_start = 0usize;
        while panel_start < row_groups {
            let panel_end = (panel_start + panel_rg).min(row_groups);
            for cg in 0..col_groups {
                for (v, slot) in b_idx.iter_mut().enumerate() {
                    *slot = cg * w + v;
                }
                // 8-row fast path: 4 tiles per call, B reused across 8 A-rows (vs
                // 2) — quadruples B reuse; with the panel, A also stays resident.
                let mut rg = panel_start;
                // 8-row fast path: 4 tiles per call, B reused across 8 A-rows.
                // micro path guarantees h==2, so a0 = rg*2 spans 8 contiguous rows.
                // Inline closure to hash + test the 4 tile messages (shared by both
                // the AVX-512 and AVX-VNNI kernels).
                #[cfg(target_arch = "x86_64")]
                {
                    let mut emit4 = |msgs: &[[u32; JACKPOT_SIZE]; 4], local: &mut u64| {
                        for msg in msgs.iter() {
                            let mut mb = [0u8; JACKPOT_SIZE * 4];
                            for (i, word) in msg.iter().enumerate() {
                                mb[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
                            }
                            let hash: [u8; 32] = *blake3::keyed_hash(&a_seed, &mb).as_bytes();
                            *local += 1;
                            if hash.as_slice() < job.share_target.as_slice() {
                                let is_block = hash.as_slice() < job.target.as_slice();
                                on_hit(Hit { hash, is_block, job_id: job.job_id.clone() });
                            }
                        }
                    };
                    if use_micro512 {
                        while rg + 4 <= panel_end {
                            let a0 = rg * h;
                            let msgs = unsafe {
                                crate::microkernel::jackpot_tile_8x64_avx512(
                                    &a, &b_packed, a0, cg * groups_per_tile16, k, r, &mut jackpot,
                                )
                            };
                            emit4(&msgs, &mut local);
                            rg += 4;
                            if stop.load(Ordering::Relaxed) {
                                break;
                            }
                        }
                    } else if use_micro_vnni {
                        while rg + 4 <= panel_end {
                            let a0 = rg * h;
                            let msgs = unsafe {
                                crate::microkernel::jackpot_tile_8x64_avxvnni(
                                    &a, &b_packed, a0, cg * groups_per_tile8, k, r, &mut jackpot,
                                )
                            };
                            emit4(&msgs, &mut local);
                            rg += 4;
                            if stop.load(Ordering::Relaxed) {
                                break;
                            }
                        }
                    }
                }
                // Remaining row-groups (and the whole non-micro path): 2-row tiles.
                while rg < panel_end {
                    for (u, slot) in a_idx.iter_mut().enumerate() {
                        *slot = rg * h + u;
                    }
                    let msg = if use_micro512 {
                        #[cfg(target_arch = "x86_64")]
                        {
                            unsafe {
                                crate::microkernel::jackpot_tile_2x64_avx512(
                                    &a, &b_packed, a_idx[0], a_idx[1],
                                    cg * groups_per_tile16, k, r, &mut jackpot,
                                )
                            }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            jackpot_tile_flat(&a, &b, &a_idx, &b_idx, k, r, &mut jackpot)
                        }
                    } else {
                        // AVX-VNNI panel remainder (<4 row-groups) + any non-micro
                        // CPU/shape: flat path (AVX-VNNI dot on i9, scalar elsewhere).
                        jackpot_tile_flat(&a, &b, &a_idx, &b_idx, k, r, &mut jackpot)
                    };
                    let mut mb = [0u8; JACKPOT_SIZE * 4];
                    for (i, word) in msg.iter().enumerate() {
                        mb[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
                    }
                    let hash: [u8; 32] = *blake3::keyed_hash(&a_seed, &mb).as_bytes();
                    local += 1;
                    if hash.as_slice() < job.share_target.as_slice() {
                        let is_block = hash.as_slice() < job.target.as_slice();
                        on_hit(Hit { hash, is_block, job_id: job.job_id.clone() });
                    }
                    rg += 1;
                    if rg & 0x7f == 0 && stop.load(Ordering::Relaxed) {
                        break;
                    }
                }
                // Publish progress + honor stop between col-groups.
                attempts_ctr.fetch_add(local, Ordering::Relaxed);
                local = 0;
                if stop.load(Ordering::Relaxed) {
                    break;
                }
            }
            if stop.load(Ordering::Relaxed) {
                break;
            }
            panel_start += panel_rg;
        }
        attempts_ctr.fetch_add(local, Ordering::Relaxed);
        local = 0;
    }
}
