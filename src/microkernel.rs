//! Register-blocked int8 micro-kernel for the Pearl mainnet tile (h=2, w=64).
//!
//! Why this exists: the per-`(u,v)` dot path (`cpu_engine::jackpot_tile_flat`)
//! issues one tiny `vpdpbusd` chain (r=128 ⇒ 2 instr) + a horizontal reduce per
//! cell. The reduce dominates and the single dependency chain stalls on the
//! ~5-cycle `vpdpbusd` latency → we measured only +5% over autovec.
//!
//! The fix is a register-blocked kernel that avoids both: it keeps **16
//! independent zmm accumulators** so the latency is fully hidden, puts each
//! output **column in a lane** (no per-cell horizontal reduce), feeds A by
//! broadcast and streams B contiguously.
//!
//! We mirror that for our shape. Pearl mainnet is **2 rows × 64 cols** = 4 col
//! groups of 16, so the row/col axes only give 2×4 = 8 product chains. To reach
//! a deeper pipeline we additionally **2-way unroll the k loop**: each product
//! keeps a `lo`/`hi` partial accumulator over even/odd k-steps, summed before
//! store. That is **16 independent product accumulators**, plus **4 shared bias
//! accumulators** (`Σb` per group, for the signed→unsigned `vpdpbusd`
//! correction `Σa·b = Σ(a+128)·b − 128·Σb`) = **20 independent chains** ⇒ the
//! `vpdpbusd` latency is fully hidden even at the per-chain level.
//!
//! B must be packed column-interleaved so 16 columns land in the 16 lanes:
//! `b_packed[group][kk][16 cols][4 k]`. Packing is O(n·k) once per setup,
//! amortized over thousands of tiles (~0.1% overhead).
//!
//! Output is **bit-identical** to `pearl_compute::compute_jackpot_pearl`
//! (wrapping i32 throughout), so shares stay valid against the Pearl reference.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::pearl_compute::{JACKPOT_SIZE, LROT_PER_TILE};

/// True when the fast micro-kernel is usable for this shape on this CPU.
#[inline]
pub fn micro_kernel_applicable(h: usize, w: usize, n_batch: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return h == 2
            && w == 64
            && n_batch % 16 == 0
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512f");
    }
    #[allow(unreachable_code)]
    {
        let _ = (h, w, n_batch);
        false
    }
}

/// Pack `b` (`n_cols` rows of `k` i8, row-major: `b[col*k + l]`) into the
/// 16-wide column-interleaved layout the AVX-512 kernel consumes:
/// `out[((g*(k/4) + kk)*16 + j)*4 + e] = b[(g*16+j)*k + kk*4 + e]`.
/// `n_cols` must be a multiple of 16 and `k` a multiple of 4. `out` is resized
/// to `n_cols*k` and fully overwritten.
pub fn pack_b_16(b: &[i8], n_cols: usize, k: usize, out: &mut Vec<i8>) {
    debug_assert_eq!(n_cols % 16, 0);
    debug_assert_eq!(k % 4, 0);
    debug_assert_eq!(b.len(), n_cols * k);
    if out.len() != n_cols * k {
        out.resize(n_cols * k, 0);
    }
    let ng = n_cols / 16;
    let kk_n = k / 4;
    for g in 0..ng {
        for kk in 0..kk_n {
            let dst_base = (g * kk_n + kk) * 64;
            for j in 0..16 {
                let src = (g * 16 + j) * k + kk * 4;
                let d = dst_base + j * 4;
                out[d..d + 4].copy_from_slice(&b[src..src + 4]);
            }
        }
    }
}

/// Correct one group's product accumulator (`acc = Σ(a+128)·b`, `accb = Σb`)
/// into `Σa·b` and add (wrapping) into the 16 jackpot cells `dst`.
/// `corrected = acc - 128·accb`, lane-wise wrapping i32.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn store_group(
    dst: &mut [i32],
    acc: std::arch::x86_64::__m512i,
    accb: std::arch::x86_64::__m512i,
) {
    use std::arch::x86_64::*;
    let corrected = _mm512_sub_epi32(acc, _mm512_slli_epi32(accb, 7));
    // SIMD accumulate straight into the 16 jackpot cells: load + add + store,
    // no vector→scalar roundtrip. `_mm512_add_epi32` wraps mod 2^32 exactly like
    // the old per-lane `wrapping_add`, so the result is bit-identical. `dst` is
    // always a contiguous 16-i32 (64-byte) jackpot slice.
    let cur = _mm512_loadu_si512(dst.as_ptr() as *const __m512i);
    _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, _mm512_add_epi32(cur, corrected));
}

/// XOR-reduce a tile's 128 contiguous jackpot cells to one `u32`. XOR is
/// associative/commutative ⇒ the SIMD grouping (8× `_mm512_xor` + horizontal
/// reduce) is bit-identical to the old 128-iteration scalar XOR, for ~1/16th the
/// ops — keeps the hot fold off the scalar units at the power ceiling.
///
/// # Safety
/// `p` must point to ≥128 readable contiguous `i32`. AVX-512F required.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn xor_reduce_128(p: *const i32) -> u32 {
    use std::arch::x86_64::*;
    let mut v = _mm512_loadu_si512(p as *const __m512i);
    v = _mm512_xor_si512(v, _mm512_loadu_si512(p.add(16) as *const __m512i));
    v = _mm512_xor_si512(v, _mm512_loadu_si512(p.add(32) as *const __m512i));
    v = _mm512_xor_si512(v, _mm512_loadu_si512(p.add(48) as *const __m512i));
    v = _mm512_xor_si512(v, _mm512_loadu_si512(p.add(64) as *const __m512i));
    v = _mm512_xor_si512(v, _mm512_loadu_si512(p.add(80) as *const __m512i));
    v = _mm512_xor_si512(v, _mm512_loadu_si512(p.add(96) as *const __m512i));
    v = _mm512_xor_si512(v, _mm512_loadu_si512(p.add(112) as *const __m512i));
    // Horizontal XOR of the 16 lanes (no _mm512_reduce_xor intrinsic in std):
    // fold 512→256→128 with vector XORs, then 4 scalar lanes.
    let lo = _mm512_castsi512_si256(v);
    let hi = _mm512_extracti64x4_epi64(v, 1);
    let x256 = _mm256_xor_si256(lo, hi);
    let x128 = _mm_xor_si128(_mm256_castsi256_si128(x256), _mm256_extracti128_si256(x256, 1));
    let mut lanes = [0i32; 4];
    _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, x128);
    (lanes[0] ^ lanes[1] ^ lanes[2] ^ lanes[3]) as u32
}

/// Full per-tile jackpot for the **h=2, w=64** Pearl shape, register-blocked
/// with a 2-way k-unroll (16 product + 4 bias accumulators).
///
/// `a_eff`: flat `m_batch*k` i8 effective A strips; rows `a0`,`a1` are this
/// tile's two A rows. `b_packed`: output of [`pack_b_16`] over the whole batch;
/// `grp_base` is the first of this tile's 4 column-groups (`= col_tile * 4`).
/// Walks k in `r`-blocks, folding the tile (XOR + rotate_left) after each block
/// exactly like the reference. Returns the 16-word jackpot message.
///
/// # Safety
/// Requires AVX-512 + VNNI (gated by [`micro_kernel_applicable`]). `jackpot`
/// must hold at least 128 elements; `a0`,`a1` < `m_batch`; the 4 groups
/// `grp_base..grp_base+4` must be in range of `b_packed`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
pub unsafe fn jackpot_tile_2x64_avx512(
    a_eff: &[i8],
    b_packed: &[i8],
    a0: usize,
    a1: usize,
    grp_base: usize,
    k: usize,
    r: usize,
    jackpot: &mut [i32],
) -> [u32; JACKPOT_SIZE] {
    use std::arch::x86_64::*;

    let kk_per_r = r / 4;
    let grp_stride = (k / 4) * 64; // i8 per packed column-group
    let bias = _mm512_set1_epi8(-128i8); // XOR 0x80 ⇒ i8 + 128 as u8
    let ones = _mm512_set1_epi8(1i8);

    for x in jackpot[..128].iter_mut() {
        *x = 0;
    }
    let mut msg = [0u32; JACKPOT_SIZE];

    let pa0 = a_eff.as_ptr().add(a0 * k);
    let pa1 = a_eff.as_ptr().add(a1 * k);
    let bp = b_packed.as_ptr();
    let g0 = (grp_base) * grp_stride;
    let g1 = (grp_base + 1) * grp_stride;
    let g2 = (grp_base + 2) * grp_stride;
    let g3 = (grp_base + 3) * grp_stride;

    let mut blk = 0usize;
    let mut ll = r;
    while ll <= k {
        let l0 = ll - r;
        let off0 = g0 + (l0 / 4) * 64;
        let off1 = g1 + (l0 / 4) * 64;
        let off2 = g2 + (l0 / 4) * 64;
        let off3 = g3 + (l0 / 4) * 64;

        // 16 product accumulators: 2 rows × 4 groups × {lo,hi} k-halves.
        let mut p00l = _mm512_setzero_si512();
        let mut p01l = _mm512_setzero_si512();
        let mut p02l = _mm512_setzero_si512();
        let mut p03l = _mm512_setzero_si512();
        let mut p10l = _mm512_setzero_si512();
        let mut p11l = _mm512_setzero_si512();
        let mut p12l = _mm512_setzero_si512();
        let mut p13l = _mm512_setzero_si512();
        let mut p00h = _mm512_setzero_si512();
        let mut p01h = _mm512_setzero_si512();
        let mut p02h = _mm512_setzero_si512();
        let mut p03h = _mm512_setzero_si512();
        let mut p10h = _mm512_setzero_si512();
        let mut p11h = _mm512_setzero_si512();
        let mut p12h = _mm512_setzero_si512();
        let mut p13h = _mm512_setzero_si512();
        // 4 bias accumulators (Σb per group, shared across both rows).
        let mut s0 = _mm512_setzero_si512();
        let mut s1 = _mm512_setzero_si512();
        let mut s2 = _mm512_setzero_si512();
        let mut s3 = _mm512_setzero_si512();

        // Walk k in pairs of dwords: even step → lo accumulators, odd → hi.
        let mut kk = 0usize;
        while kk + 1 < kk_per_r {
            // --- even step (lo) ---
            let o = kk * 64;
            let bv0 = _mm512_loadu_si512(bp.add(off0 + o) as *const __m512i);
            let bv1 = _mm512_loadu_si512(bp.add(off1 + o) as *const __m512i);
            let bv2 = _mm512_loadu_si512(bp.add(off2 + o) as *const __m512i);
            let bv3 = _mm512_loadu_si512(bp.add(off3 + o) as *const __m512i);
            let ad0 = (pa0.add(l0 + kk * 4) as *const i32).read_unaligned();
            let ad1 = (pa1.add(l0 + kk * 4) as *const i32).read_unaligned();
            let au0 = _mm512_xor_si512(_mm512_set1_epi32(ad0), bias);
            let au1 = _mm512_xor_si512(_mm512_set1_epi32(ad1), bias);
            p00l = _mm512_dpbusd_epi32(p00l, au0, bv0);
            p01l = _mm512_dpbusd_epi32(p01l, au0, bv1);
            p02l = _mm512_dpbusd_epi32(p02l, au0, bv2);
            p03l = _mm512_dpbusd_epi32(p03l, au0, bv3);
            p10l = _mm512_dpbusd_epi32(p10l, au1, bv0);
            p11l = _mm512_dpbusd_epi32(p11l, au1, bv1);
            p12l = _mm512_dpbusd_epi32(p12l, au1, bv2);
            p13l = _mm512_dpbusd_epi32(p13l, au1, bv3);
            s0 = _mm512_dpbusd_epi32(s0, ones, bv0);
            s1 = _mm512_dpbusd_epi32(s1, ones, bv1);
            s2 = _mm512_dpbusd_epi32(s2, ones, bv2);
            s3 = _mm512_dpbusd_epi32(s3, ones, bv3);

            // --- odd step (hi) ---
            let o2 = (kk + 1) * 64;
            let cv0 = _mm512_loadu_si512(bp.add(off0 + o2) as *const __m512i);
            let cv1 = _mm512_loadu_si512(bp.add(off1 + o2) as *const __m512i);
            let cv2 = _mm512_loadu_si512(bp.add(off2 + o2) as *const __m512i);
            let cv3 = _mm512_loadu_si512(bp.add(off3 + o2) as *const __m512i);
            let bd0 = (pa0.add(l0 + (kk + 1) * 4) as *const i32).read_unaligned();
            let bd1 = (pa1.add(l0 + (kk + 1) * 4) as *const i32).read_unaligned();
            let bu0 = _mm512_xor_si512(_mm512_set1_epi32(bd0), bias);
            let bu1 = _mm512_xor_si512(_mm512_set1_epi32(bd1), bias);
            p00h = _mm512_dpbusd_epi32(p00h, bu0, cv0);
            p01h = _mm512_dpbusd_epi32(p01h, bu0, cv1);
            p02h = _mm512_dpbusd_epi32(p02h, bu0, cv2);
            p03h = _mm512_dpbusd_epi32(p03h, bu0, cv3);
            p10h = _mm512_dpbusd_epi32(p10h, bu1, cv0);
            p11h = _mm512_dpbusd_epi32(p11h, bu1, cv1);
            p12h = _mm512_dpbusd_epi32(p12h, bu1, cv2);
            p13h = _mm512_dpbusd_epi32(p13h, bu1, cv3);
            s0 = _mm512_dpbusd_epi32(s0, ones, cv0);
            s1 = _mm512_dpbusd_epi32(s1, ones, cv1);
            s2 = _mm512_dpbusd_epi32(s2, ones, cv2);
            s3 = _mm512_dpbusd_epi32(s3, ones, cv3);

            kk += 2;
        }
        // Odd remainder (kk_per_r odd): fold into the lo accumulators.
        if kk < kk_per_r {
            let o = kk * 64;
            let bv0 = _mm512_loadu_si512(bp.add(off0 + o) as *const __m512i);
            let bv1 = _mm512_loadu_si512(bp.add(off1 + o) as *const __m512i);
            let bv2 = _mm512_loadu_si512(bp.add(off2 + o) as *const __m512i);
            let bv3 = _mm512_loadu_si512(bp.add(off3 + o) as *const __m512i);
            let ad0 = (pa0.add(l0 + kk * 4) as *const i32).read_unaligned();
            let ad1 = (pa1.add(l0 + kk * 4) as *const i32).read_unaligned();
            let au0 = _mm512_xor_si512(_mm512_set1_epi32(ad0), bias);
            let au1 = _mm512_xor_si512(_mm512_set1_epi32(ad1), bias);
            p00l = _mm512_dpbusd_epi32(p00l, au0, bv0);
            p01l = _mm512_dpbusd_epi32(p01l, au0, bv1);
            p02l = _mm512_dpbusd_epi32(p02l, au0, bv2);
            p03l = _mm512_dpbusd_epi32(p03l, au0, bv3);
            p10l = _mm512_dpbusd_epi32(p10l, au1, bv0);
            p11l = _mm512_dpbusd_epi32(p11l, au1, bv1);
            p12l = _mm512_dpbusd_epi32(p12l, au1, bv2);
            p13l = _mm512_dpbusd_epi32(p13l, au1, bv3);
            s0 = _mm512_dpbusd_epi32(s0, ones, bv0);
            s1 = _mm512_dpbusd_epi32(s1, ones, bv1);
            s2 = _mm512_dpbusd_epi32(s2, ones, bv2);
            s3 = _mm512_dpbusd_epi32(s3, ones, bv3);
        }

        // Reduce lo+hi per product, then correct with the shared bias and add.
        // Row 0 groups 0..4 ⇒ cols 0..64 ; row 1 ⇒ cols 64..128.
        store_group(&mut jackpot[0..16], _mm512_add_epi32(p00l, p00h), s0);
        store_group(&mut jackpot[16..32], _mm512_add_epi32(p01l, p01h), s1);
        store_group(&mut jackpot[32..48], _mm512_add_epi32(p02l, p02h), s2);
        store_group(&mut jackpot[48..64], _mm512_add_epi32(p03l, p03h), s3);
        store_group(&mut jackpot[64..80], _mm512_add_epi32(p10l, p10h), s0);
        store_group(&mut jackpot[80..96], _mm512_add_epi32(p11l, p11h), s1);
        store_group(&mut jackpot[96..112], _mm512_add_epi32(p12l, p12h), s2);
        store_group(&mut jackpot[112..128], _mm512_add_epi32(p13l, p13h), s3);

        let xored = xor_reduce_128(jackpot.as_ptr());
        msg[blk % JACKPOT_SIZE] = msg[blk % JACKPOT_SIZE].rotate_left(LROT_PER_TILE) ^ xored;

        ll += r;
        blk += 1;
    }
    msg
}

/// **8-row variant** — process **4 tiles (8 contiguous A rows) at once** against
/// the same 64-col group, so each B load is reused across **8 rows** instead of
/// 2. This is the 8×N register-blocking trick: it quadruples arithmetic
/// intensity (B re-streamed 4× less), the real lever once a high-thread sweep is
/// memory-bandwidth bound. 16 product accumulators (8 rows × 2 groups, walked
/// 2 groups at a time) + 2 shared bias accumulators ⇒ latency stays hidden.
///
/// `a0` is the first of 8 contiguous rows (`a0..a0+8`); tile `t` owns rows
/// `a0+2t`,`a0+2t+1`. Returns the 4 tile messages, byte-for-byte equal to four
/// independent [`jackpot_tile_2x64_avx512`] calls (and hence to zk-pow).
///
/// # Safety
/// AVX-512 + VNNI (gated by [`micro_kernel_applicable`]). `jackpot` must hold
/// ≥512 i32 scratch; `a0+8 <= m_batch`; the 4 groups `grp_base..grp_base+4`
/// must be in range of `b_packed`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
pub unsafe fn jackpot_tile_8x64_avx512(
    a_eff: &[i8],
    b_packed: &[i8],
    a0: usize,
    grp_base: usize,
    k: usize,
    r: usize,
    jackpot: &mut [i32],
) -> [[u32; JACKPOT_SIZE]; 4] {
    use std::arch::x86_64::*;

    let kk_per_r = r / 4;
    let grp_stride = (k / 4) * 64;
    let bias = _mm512_set1_epi8(-128i8);
    let ones = _mm512_set1_epi8(1i8);

    for x in jackpot[..512].iter_mut() {
        *x = 0;
    }
    let mut msg = [[0u32; JACKPOT_SIZE]; 4];

    let mut pa = [a_eff.as_ptr(); 8];
    for (i, p) in pa.iter_mut().enumerate() {
        *p = a_eff.as_ptr().add((a0 + i) * k);
    }
    let bp = b_packed.as_ptr();
    let goff = [
        grp_base * grp_stride,
        (grp_base + 1) * grp_stride,
        (grp_base + 2) * grp_stride,
        (grp_base + 3) * grp_stride,
    ];

    let mut ll = r;
    let mut blk = 0usize;
    while ll <= k {
        let l0 = ll - r;
        // Two passes over the 4 col-groups, 2 groups per pass: 8 rows × 2 groups
        // = 16 independent product accumulators in flight, B loaded once per kk
        // and shared across all 8 rows.
        for gp in 0..2 {
            let ga = gp * 2;
            let gb = gp * 2 + 1;
            let offa = goff[ga] + (l0 / 4) * 64;
            let offb = goff[gb] + (l0 / 4) * 64;
            let mut acc_a = [_mm512_setzero_si512(); 8];
            let mut acc_b = [_mm512_setzero_si512(); 8];
            let mut sa = _mm512_setzero_si512();
            let mut sb = _mm512_setzero_si512();

            let mut kk = 0usize;
            while kk < kk_per_r {
                let o = kk * 64;
                let bva = _mm512_loadu_si512(bp.add(offa + o) as *const __m512i);
                let bvb = _mm512_loadu_si512(bp.add(offb + o) as *const __m512i);
                for row in 0..8 {
                    let ad = (pa[row].add(l0 + kk * 4) as *const i32).read_unaligned();
                    let au = _mm512_xor_si512(_mm512_set1_epi32(ad), bias);
                    acc_a[row] = _mm512_dpbusd_epi32(acc_a[row], au, bva);
                    acc_b[row] = _mm512_dpbusd_epi32(acc_b[row], au, bvb);
                }
                sa = _mm512_dpbusd_epi32(sa, ones, bva);
                sb = _mm512_dpbusd_epi32(sb, ones, bvb);
                kk += 1;
            }

            // Accumulate this k-block into the running per-tile jackpot.
            for row in 0..8 {
                let t = row / 2;
                let lr = row % 2;
                let base = t * 128 + lr * 64;
                store_group(&mut jackpot[base + ga * 16..base + ga * 16 + 16], acc_a[row], sa);
                store_group(&mut jackpot[base + gb * 16..base + gb * 16 + 16], acc_b[row], sb);
            }
        }

        // Fold each tile's running jackpot into its own message (XOR + rotate),
        // exactly like the 2×64 path per r-block.
        for t in 0..4 {
            let xored = xor_reduce_128(jackpot.as_ptr().add(t * 128));
            msg[t][blk % JACKPOT_SIZE] =
                msg[t][blk % JACKPOT_SIZE].rotate_left(LROT_PER_TILE) ^ xored;
        }

        ll += r;
        blk += 1;
    }
    msg
}

// ===========================================================================
// AVX-VNNI 256-bit variant — for Intel Arrow Lake / Core Ultra (i9-285K) which
// have AVX-VNNI but NO AVX-512. The ymm register holds **8** i32 lanes (vs 16
// for zmm), so column-groups are **8 wide** (w=64 ⇒ 8 groups). Same idea as the
// AVX-512 8×64 kernel: each output column lives in a lane (no per-cell
// horizontal reduce), 8 independent product accumulators (one per A-row) hide
// the vpdpbusd latency and reuse each B load across 8 rows. VEX-encoded
// `_mm256_dpbusd_avx_epi32` (NOT the EVEX `_mm256_dpbusd_epi32`, which SIGILLs
// on AVX-512-less CPUs — cf vnni.rs). Bit-identical to the scalar reference.
// ===========================================================================

/// True when the AVX-VNNI 8-row micro-kernel is usable (i9-285K path).
#[inline]
pub fn micro_kernel_avxvnni_applicable(h: usize, w: usize, n_batch: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return h == 2
            && w == 64
            && n_batch % 16 == 0
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("avxvnni");
    }
    #[allow(unreachable_code)]
    {
        let _ = (h, w, n_batch);
        false
    }
}

/// Pack `b` into the **8-wide** column-interleaved layout the AVX-VNNI kernel
/// consumes: `out[((g*(k/4) + kk)*8 + j)*4 + e] = b[(g*8+j)*k + kk*4 + e]`.
/// `n_cols` must be a multiple of 8 and `k` a multiple of 4.
pub fn pack_b_8(b: &[i8], n_cols: usize, k: usize, out: &mut Vec<i8>) {
    debug_assert_eq!(n_cols % 8, 0);
    debug_assert_eq!(k % 4, 0);
    debug_assert_eq!(b.len(), n_cols * k);
    if out.len() != n_cols * k {
        out.resize(n_cols * k, 0);
    }
    let ng = n_cols / 8;
    let kk_n = k / 4;
    for g in 0..ng {
        for kk in 0..kk_n {
            let dst_base = (g * kk_n + kk) * 32;
            for j in 0..8 {
                let src = (g * 8 + j) * k + kk * 4;
                let d = dst_base + j * 4;
                out[d..d + 4].copy_from_slice(&b[src..src + 4]);
            }
        }
    }
}

/// Correct one 8-col group (`acc = Σ(a+128)·b`, `accb = Σb`) into `Σa·b` and add
/// (wrapping) into the 8 jackpot cells `dst`. AVX2 (256-bit, 8 i32 lanes).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_group_8(
    dst: &mut [i32],
    acc: std::arch::x86_64::__m256i,
    accb: std::arch::x86_64::__m256i,
) {
    use std::arch::x86_64::*;
    let corrected = _mm256_sub_epi32(acc, _mm256_slli_epi32(accb, 7));
    let cur = _mm256_loadu_si256(dst.as_ptr() as *const __m256i);
    _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, _mm256_add_epi32(cur, corrected));
}

/// XOR-reduce a tile's 128 contiguous jackpot cells to one `u32`, AVX2 (16 ymm
/// loads + horizontal fold). Bit-identical to the 128-iter scalar XOR.
///
/// # Safety
/// `p` must point to ≥128 readable contiguous `i32`. AVX2 required.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn xor_reduce_128_avx2(p: *const i32) -> u32 {
    use std::arch::x86_64::*;
    let mut v = _mm256_loadu_si256(p as *const __m256i);
    let mut i = 8usize;
    while i < 128 {
        v = _mm256_xor_si256(v, _mm256_loadu_si256(p.add(i) as *const __m256i));
        i += 8;
    }
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let x128 = _mm_xor_si128(lo, hi);
    let mut lanes = [0i32; 4];
    _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, x128);
    (lanes[0] ^ lanes[1] ^ lanes[2] ^ lanes[3]) as u32
}

/// **8-row AVX-VNNI variant** — process 4 tiles (8 contiguous A rows) at once
/// against the same 64-col block, B reused across 8 rows. ymm = 8 i32 lanes ⇒
/// 8 column-groups of 8. 8 product accumulators (one per row) + 1 shared bias
/// per group ⇒ latency hidden. Byte-for-byte equal to four independent 2×64
/// reference jackpots (hence to zk-pow).
///
/// `a0` is the first of 8 contiguous rows; tile `t` owns rows `a0+2t`,`a0+2t+1`.
///
/// # Safety
/// AVX2 + AVX-VNNI (gated by [`micro_kernel_avxvnni_applicable`]). `jackpot`
/// must hold ≥512 i32 scratch; `a0+8 <= m_batch`; groups `grp_base..grp_base+4`
/// (16-wide units, i.e. 8 packed 8-col groups) in range of `b_packed`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
pub unsafe fn jackpot_tile_8x64_avxvnni(
    a_eff: &[i8],
    b_packed: &[i8],
    a0: usize,
    grp8_base: usize,
    k: usize,
    r: usize,
    jackpot: &mut [i32],
) -> [[u32; JACKPOT_SIZE]; 4] {
    use std::arch::x86_64::*;

    let kk_per_r = r / 4;
    let grp_stride = (k / 4) * 32; // i8 per packed 8-col group
    let bias = _mm256_set1_epi8(-128i8);
    let ones = _mm256_set1_epi8(1i8);

    for x in jackpot[..512].iter_mut() {
        *x = 0;
    }
    let mut msg = [[0u32; JACKPOT_SIZE]; 4];

    let mut pa = [a_eff.as_ptr(); 8];
    for (i, p) in pa.iter_mut().enumerate() {
        *p = a_eff.as_ptr().add((a0 + i) * k);
    }
    let bp = b_packed.as_ptr();

    let mut ll = r;
    let mut blk = 0usize;
    while ll <= k {
        let l0 = ll - r;
        // 8 column-groups of 8 cols (w=64). B loaded once per (group,kk), shared
        // across all 8 rows.
        for g in 0..8 {
            let off = (grp8_base + g) * grp_stride + (l0 / 4) * 32;
            let mut acc = [_mm256_setzero_si256(); 8];
            let mut sb = _mm256_setzero_si256();

            let mut kk = 0usize;
            while kk < kk_per_r {
                let bv = _mm256_loadu_si256(bp.add(off + kk * 32) as *const __m256i);
                for row in 0..8 {
                    let ad = (pa[row].add(l0 + kk * 4) as *const i32).read_unaligned();
                    let au = _mm256_xor_si256(_mm256_set1_epi32(ad), bias);
                    acc[row] = _mm256_dpbusd_avx_epi32(acc[row], au, bv);
                }
                sb = _mm256_dpbusd_avx_epi32(sb, ones, bv);
                kk += 1;
            }

            // Store this group's 8 cells into each row's jackpot slice.
            // row r → tile r/2, half (r%2) ⇒ base = tile*128 + half*64; cols g*8..+8.
            for row in 0..8 {
                let t = row / 2;
                let lr = row % 2;
                let base = t * 128 + lr * 64 + g * 8;
                store_group_8(&mut jackpot[base..base + 8], acc[row], sb);
            }
        }

        // Fold each tile's running jackpot into its message (XOR + rotate).
        for t in 0..4 {
            let xored = xor_reduce_128_avx2(jackpot.as_ptr().add(t * 128));
            msg[t][blk % JACKPOT_SIZE] =
                msg[t][blk % JACKPOT_SIZE].rotate_left(LROT_PER_TILE) ^ xored;
        }

        ll += r;
        blk += 1;
    }
    msg
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::pearl_compute::{compute_jackpot_pearl, PearlNoise, PearlParams};

    /// The register-blocked AVX-512 kernel must be bit-identical to the scalar
    /// reference for the h=2,w=64 shape (and hence to zk-pow). Skips on CPUs
    /// without AVX-512 VNNI.
    #[test]
    fn micro_2x64_matches_reference() {
        if !is_x86_feature_detected!("avx512vnni")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512f")
        {
            eprintln!("skip: no AVX-512 VNNI on this host");
            return;
        }
        let (h, w, k, r) = (2usize, 64usize, 256usize, 64usize);
        let mk = |rows: usize, salt: i64| -> Vec<Vec<i8>> {
            (0..rows)
                .map(|u| {
                    (0..k)
                        .map(|l| (((u as i64 * 37 + l as i64 * 7 + salt) % 127) - 63) as i8)
                        .collect()
                })
                .collect()
        };
        let secret_a = mk(h, 1);
        let secret_b = mk(w, 2);
        let noise = PearlNoise { a: mk(h, 3), b: mk(w, 4) };

        let reference =
            compute_jackpot_pearl(PearlParams { h, w, k, r }, &secret_a, &secret_b, &noise);

        // Build flat effective strips (secret + noise, wrapping i8).
        let mut a_eff = vec![0i8; h * k];
        let mut b_eff = vec![0i8; w * k];
        for u in 0..h {
            for l in 0..k {
                a_eff[u * k + l] = secret_a[u][l].wrapping_add(noise.a[u][l]);
            }
        }
        for v in 0..w {
            for l in 0..k {
                b_eff[v * k + l] = secret_b[v][l].wrapping_add(noise.b[v][l]);
            }
        }

        let mut b_packed = Vec::new();
        pack_b_16(&b_eff, w, k, &mut b_packed);

        let mut jackpot = vec![0i32; h * w];
        let got =
            unsafe { jackpot_tile_2x64_avx512(&a_eff, &b_packed, 0, 1, 0, k, r, &mut jackpot) };

        assert_eq!(reference, got, "micro-kernel diverges from scalar reference");
    }

    /// Same check with an **odd** `kk_per_r` (r=68 ⇒ 17 dword-steps) so the
    /// 2-way k-unroll's remainder path is exercised and stays bit-identical.
    #[test]
    fn micro_2x64_matches_reference_odd_tail() {
        if !is_x86_feature_detected!("avx512vnni")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512f")
        {
            eprintln!("skip: no AVX-512 VNNI on this host");
            return;
        }
        // r=68 (kk_per_r=17, odd), k=136 (2 r-blocks).
        let (h, w, k, r) = (2usize, 64usize, 136usize, 68usize);
        let mk = |rows: usize, salt: i64| -> Vec<Vec<i8>> {
            (0..rows)
                .map(|u| {
                    (0..k)
                        .map(|l| (((u as i64 * 29 + l as i64 * 11 + salt) % 127) - 63) as i8)
                        .collect()
                })
                .collect()
        };
        let secret_a = mk(h, 5);
        let secret_b = mk(w, 6);
        let noise = PearlNoise { a: mk(h, 7), b: mk(w, 8) };

        let reference =
            compute_jackpot_pearl(PearlParams { h, w, k, r }, &secret_a, &secret_b, &noise);

        let mut a_eff = vec![0i8; h * k];
        let mut b_eff = vec![0i8; w * k];
        for u in 0..h {
            for l in 0..k {
                a_eff[u * k + l] = secret_a[u][l].wrapping_add(noise.a[u][l]);
            }
        }
        for v in 0..w {
            for l in 0..k {
                b_eff[v * k + l] = secret_b[v][l].wrapping_add(noise.b[v][l]);
            }
        }

        let mut b_packed = Vec::new();
        pack_b_16(&b_eff, w, k, &mut b_packed);

        let mut jackpot = vec![0i32; h * w];
        let got =
            unsafe { jackpot_tile_2x64_avx512(&a_eff, &b_packed, 0, 1, 0, k, r, &mut jackpot) };

        assert_eq!(reference, got, "micro-kernel (odd tail) diverges from reference");
    }

    /// The 8-row kernel must produce, for each of its 4 tiles, exactly what the
    /// scalar reference gives for that tile's 2 rows — i.e. bit-identical to four
    /// independent `jackpot_tile_2x64` calls, hence to zk-pow. Validates B reuse
    /// across 8 rows doesn't perturb any tile.
    #[test]
    fn micro_8x64_matches_reference() {
        if !is_x86_feature_detected!("avx512vnni")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512f")
        {
            eprintln!("skip: no AVX-512 VNNI on this host");
            return;
        }
        let (w, k, r) = (64usize, 256usize, 64usize);
        let rows = 8usize; // 4 tiles × 2
        let mk = |n: usize, salt: i64| -> Vec<Vec<i8>> {
            (0..n)
                .map(|u| {
                    (0..k)
                        .map(|l| (((u as i64 * 41 + l as i64 * 13 + salt) % 127) - 63) as i8)
                        .collect()
                })
                .collect()
        };
        let secret_a = mk(rows, 1);
        let secret_b = mk(w, 2);
        let noise_a = mk(rows, 3);
        let noise_b = mk(w, 4);

        // Effective flat strips (secret + noise, wrapping).
        let mut a_eff = vec![0i8; rows * k];
        let mut b_eff = vec![0i8; w * k];
        for u in 0..rows {
            for l in 0..k {
                a_eff[u * k + l] = secret_a[u][l].wrapping_add(noise_a[u][l]);
            }
        }
        for v in 0..w {
            for l in 0..k {
                b_eff[v * k + l] = secret_b[v][l].wrapping_add(noise_b[v][l]);
            }
        }

        let mut b_packed = Vec::new();
        pack_b_16(&b_eff, w, k, &mut b_packed);

        let mut jackpot = vec![0i32; 512];
        let got = unsafe { jackpot_tile_8x64_avx512(&a_eff, &b_packed, 0, 0, k, r, &mut jackpot) };

        // Reference, tile by tile (2 rows each).
        for t in 0..4 {
            let sa = vec![secret_a[2 * t].clone(), secret_a[2 * t + 1].clone()];
            let na = vec![noise_a[2 * t].clone(), noise_a[2 * t + 1].clone()];
            let noise = PearlNoise { a: na, b: noise_b.clone() };
            let reference = compute_jackpot_pearl(
                PearlParams { h: 2, w, k, r },
                &sa,
                &secret_b,
                &noise,
            );
            assert_eq!(reference, got[t], "8×64 tile {t} diverges from reference");
        }
    }

    /// AVX-VNNI 8-row kernel: each of its 4 tiles must equal the scalar
    /// reference for that tile's 2 rows — bit-identical, hence valid vs zk-pow.
    /// Skips on CPUs without AVX-VNNI.
    #[test]
    fn micro_8x64_avxvnni_matches_reference() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("avxvnni") {
            eprintln!("skip: no AVX-VNNI on this host");
            return;
        }
        let (w, k, r) = (64usize, 256usize, 64usize);
        let rows = 8usize;
        let mk = |n: usize, salt: i64| -> Vec<Vec<i8>> {
            (0..n)
                .map(|u| {
                    (0..k)
                        .map(|l| (((u as i64 * 41 + l as i64 * 13 + salt) % 127) - 63) as i8)
                        .collect()
                })
                .collect()
        };
        let secret_a = mk(rows, 1);
        let secret_b = mk(w, 2);
        let noise_a = mk(rows, 3);
        let noise_b = mk(w, 4);

        let mut a_eff = vec![0i8; rows * k];
        let mut b_eff = vec![0i8; w * k];
        for u in 0..rows {
            for l in 0..k {
                a_eff[u * k + l] = secret_a[u][l].wrapping_add(noise_a[u][l]);
            }
        }
        for v in 0..w {
            for l in 0..k {
                b_eff[v * k + l] = secret_b[v][l].wrapping_add(noise_b[v][l]);
            }
        }

        let mut b_packed = Vec::new();
        pack_b_8(&b_eff, w, k, &mut b_packed);

        let mut jackpot = vec![0i32; 512];
        let got = unsafe { jackpot_tile_8x64_avxvnni(&a_eff, &b_packed, 0, 0, k, r, &mut jackpot) };

        for t in 0..4 {
            let sa = vec![secret_a[2 * t].clone(), secret_a[2 * t + 1].clone()];
            let na = vec![noise_a[2 * t].clone(), noise_a[2 * t + 1].clone()];
            let noise = PearlNoise { a: na, b: noise_b.clone() };
            let reference =
                compute_jackpot_pearl(PearlParams { h: 2, w, k, r }, &sa, &secret_b, &noise);
            assert_eq!(reference, got[t], "AVX-VNNI 8×64 tile {t} diverges from reference");
        }
    }

    /// Same with an **odd** `kk_per_r` (r=68 ⇒ 17 dword-steps) and 2 r-blocks.
    #[test]
    fn micro_8x64_avxvnni_matches_reference_odd_tail() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("avxvnni") {
            eprintln!("skip: no AVX-VNNI on this host");
            return;
        }
        let (w, k, r) = (64usize, 136usize, 68usize);
        let rows = 8usize;
        let mk = |n: usize, salt: i64| -> Vec<Vec<i8>> {
            (0..n)
                .map(|u| {
                    (0..k)
                        .map(|l| (((u as i64 * 29 + l as i64 * 11 + salt) % 127) - 63) as i8)
                        .collect()
                })
                .collect()
        };
        let secret_a = mk(rows, 5);
        let secret_b = mk(w, 6);
        let noise_a = mk(rows, 7);
        let noise_b = mk(w, 8);

        let mut a_eff = vec![0i8; rows * k];
        let mut b_eff = vec![0i8; w * k];
        for u in 0..rows {
            for l in 0..k {
                a_eff[u * k + l] = secret_a[u][l].wrapping_add(noise_a[u][l]);
            }
        }
        for v in 0..w {
            for l in 0..k {
                b_eff[v * k + l] = secret_b[v][l].wrapping_add(noise_b[v][l]);
            }
        }

        let mut b_packed = Vec::new();
        pack_b_8(&b_eff, w, k, &mut b_packed);

        let mut jackpot = vec![0i32; 512];
        let got = unsafe { jackpot_tile_8x64_avxvnni(&a_eff, &b_packed, 0, 0, k, r, &mut jackpot) };

        for t in 0..4 {
            let sa = vec![secret_a[2 * t].clone(), secret_a[2 * t + 1].clone()];
            let na = vec![noise_a[2 * t].clone(), noise_a[2 * t + 1].clone()];
            let noise = PearlNoise { a: na, b: noise_b.clone() };
            let reference =
                compute_jackpot_pearl(PearlParams { h: 2, w, k, r }, &sa, &secret_b, &noise);
            assert_eq!(reference, got[t], "AVX-VNNI 8×64 (odd tail) tile {t} diverges");
        }
    }

    /// pack_b_8 round-trip: every byte lands where the 8-wide kernel expects it.
    #[test]
    fn pack_b_8_layout() {
        let (w, k) = (16usize, 16usize); // 2 groups of 8, kk_n=4
        let b: Vec<i8> = (0..(w * k)).map(|i| ((i as i64) % 127 - 63) as i8).collect();
        let mut out = Vec::new();
        pack_b_8(&b, w, k, &mut out);
        let kk_n = k / 4;
        for g in 0..w / 8 {
            for kk in 0..kk_n {
                for j in 0..8 {
                    for e in 0..4 {
                        let packed = out[(g * kk_n + kk) * 32 + j * 4 + e];
                        let orig = b[(g * 8 + j) * k + kk * 4 + e];
                        assert_eq!(packed, orig, "mismatch g={g} kk={kk} j={j} e={e}");
                    }
                }
            }
        }
    }

    /// Pack/unpack round-trip: every byte lands where the kernel expects it.
    #[test]
    fn pack_b_16_layout() {
        let (w, k) = (32usize, 16usize); // 2 groups, kk_n=4
        let b: Vec<i8> = (0..(w * k)).map(|i| ((i as i64) % 127 - 63) as i8).collect();
        let mut out = Vec::new();
        pack_b_16(&b, w, k, &mut out);
        let kk_n = k / 4;
        for g in 0..w / 16 {
            for kk in 0..kk_n {
                for j in 0..16 {
                    for e in 0..4 {
                        let packed = out[(g * kk_n + kk) * 64 + j * 4 + e];
                        let orig = b[(g * 16 + j) * k + kk * 4 + e];
                        assert_eq!(packed, orig, "mismatch g={g} kk={kk} j={j} e={e}");
                    }
                }
            }
        }
    }
}
