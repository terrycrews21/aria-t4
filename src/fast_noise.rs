//! Fast bit-identical Pearl noise generation.
//!
//! The reference `zk_pow::…::compute_noise_for_indices` spends ~all its time in
//! `matvec_sparse_perm`: for each of the `batch` rows it produces `k` outputs as
//! `row[perm[j].0] - row[perm[j].1]`, a gather over the `rank`-element row. At
//! mainnet (`rank=128`, `k=4096`, `batch=1024+`) that's millions of scalar
//! gathers per setup — the dominant cost, and the reason we were forced to a
//! large batch (where the GEMM kernel cache-thrashes). Speeding it up lets us
//! drop back to a small batch where the kernel runs ~2× faster.
//!
//! The gather source is exactly 128 bytes (`rank`), so a single AVX-512 VBMI
//! `vpermi2b` (`_mm512_permutex2var_epi8`) gathers 64 outputs at once from the
//! in-register row (raw index 0..127 maps directly: bit6 picks the hi/lo half).
//! The BLAKE3 matrix generation is reused verbatim from zk-pow (it's cheap), so
//! the result is **bit-identical** — validated against the reference in tests.
#![allow(unsafe_op_in_unsafe_fn)]

use zk_pow::circuit::pearl_noise::{
    MMSlice, generate_permutation_matrix, generate_uniform_random_matrix,
};

const SEED_LABEL_A: [u8; 32] = padded(*b"A_tensor");
const SEED_LABEL_B: [u8; 32] = padded(*b"B_tensor");

const fn padded(label: [u8; 8]) -> [u8; 32] {
    let mut r = [0u8; 32];
    let mut i = 0;
    while i < 8 {
        r[i] = label[i];
        i += 1;
    }
    r
}

/// True when the SIMD matvec applies (mainnet `rank=128`, AVX-512 VBMI host).
#[inline]
pub fn fast_noise_applicable(rank: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return rank == 128 && is_x86_feature_detected!("avx512vbmi") && is_x86_feature_detected!("avx512bw");
    }
    #[allow(unreachable_code)]
    {
        let _ = rank;
        false
    }
}

/// `out[j] = row[first[j]] - row[second[j]]` (wrapping i8), for `rank == 128`.
/// `row` is 128 i8; `first`/`second` are `k` raw indices in `0..128` (u8).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn matvec_perm_128(row: &[i8], first: &[u8], second: &[u8], out: &mut [i8]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(row.len(), 128);
    let k = out.len();
    // Row halves stay in registers across the whole k sweep.
    let lo = _mm512_loadu_si512(row.as_ptr() as *const __m512i);
    let hi = _mm512_loadu_si512(row.as_ptr().add(64) as *const __m512i);
    let fp = first.as_ptr();
    let sp = second.as_ptr();
    let op = out.as_mut_ptr();
    let mut j = 0usize;
    while j + 64 <= k {
        let ia = _mm512_loadu_si512(fp.add(j) as *const __m512i);
        let ib = _mm512_loadu_si512(sp.add(j) as *const __m512i);
        // raw index 0..127 → bit6 selects hi half, low6 the byte (vpermi2b).
        let pos = _mm512_permutex2var_epi8(lo, ia, hi);
        let neg = _mm512_permutex2var_epi8(lo, ib, hi);
        let res = _mm512_sub_epi8(pos, neg);
        _mm512_storeu_si512(op.add(j) as *mut __m512i, res);
        j += 64;
    }
    // tail (k is a multiple of 64 at mainnet, but stay correct otherwise).
    while j < k {
        let p = row[first[j] as usize] as i32;
        let n = row[second[j] as usize] as i32;
        *op.add(j) = (p - n) as i8;
        j += 1;
    }
}

/// Like `matvec_perm_128` but **adds** the result into `dst` in place
/// (`dst[j] = dst[j].wrapping_add(row[first[j]] - row[second[j]])`). Lets the
/// caller fuse noise generation with the pre-noise add so the noise matrices are
/// never materialised (saves an 8 MiB alloc + write + read-back per setup).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn matvec_perm_128_addinto(row: &[i8], first: &[u8], second: &[u8], dst: &mut [i8]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(row.len(), 128);
    let k = dst.len();
    let lo = _mm512_loadu_si512(row.as_ptr() as *const __m512i);
    let hi = _mm512_loadu_si512(row.as_ptr().add(64) as *const __m512i);
    let fp = first.as_ptr();
    let sp = second.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut j = 0usize;
    while j + 64 <= k {
        let ia = _mm512_loadu_si512(fp.add(j) as *const __m512i);
        let ib = _mm512_loadu_si512(sp.add(j) as *const __m512i);
        let pos = _mm512_permutex2var_epi8(lo, ia, hi);
        let neg = _mm512_permutex2var_epi8(lo, ib, hi);
        let res = _mm512_sub_epi8(pos, neg);
        // dst += res  (wrapping i8 add, matches scalar `wrapping_add`).
        let cur = _mm512_loadu_si512(dp.add(j) as *const __m512i);
        _mm512_storeu_si512(dp.add(j) as *mut __m512i, _mm512_add_epi8(cur, res));
        j += 64;
    }
    while j < k {
        let p = row[first[j] as usize] as i32;
        let n = row[second[j] as usize] as i32;
        *dp.add(j) = (*dp.add(j)).wrapping_add((p - n) as i8);
        j += 1;
    }
}

/// Fused noise generation + pre-noise add: computes the official Pearl noise and
/// adds it directly into the drawn strips `a_buf` (m_batch·k) and `b_buf`
/// (n_batch·k), in place. Bit-identical to
/// `compute_noise_for_indices_fast` followed by an element-wise `wrapping_add`,
/// but never allocates/writes the intermediate noise matrices — the dominant
/// memory-traffic cost at small batch. Falls back to the two-step reference
/// path on non-VBMI hosts / non-128 rank.
#[allow(clippy::too_many_arguments)]
pub fn add_noise_into_fast(
    k: usize,
    noise_rank: usize,
    commitment_hash: ([u8; 32], [u8; 32]),
    a_rows_indices: &[usize],
    b_cols_indices: &[usize],
    a_buf: &mut [i8],
    b_buf: &mut [i8],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if fast_noise_applicable(noise_rank) {
            let (b_noise_seed, a_noise_seed) = commitment_hash;
            let e_al =
                generate_uniform_random_matrix(&SEED_LABEL_A, &a_noise_seed, a_rows_indices, noise_rank);
            let e_ar_t = generate_permutation_matrix(&SEED_LABEL_A, &a_noise_seed, k, noise_rank);
            let e_bl = generate_permutation_matrix(&SEED_LABEL_B, &b_noise_seed, k, noise_rank);
            let e_br_t =
                generate_uniform_random_matrix(&SEED_LABEL_B, &b_noise_seed, b_cols_indices, noise_rank);

            let split = |perm: &[[u32; 2]]| -> (Vec<u8>, Vec<u8>) {
                let mut f = vec![0u8; perm.len()];
                let mut s = vec![0u8; perm.len()];
                for (i, p) in perm.iter().enumerate() {
                    f[i] = p[0] as u8;
                    s[i] = p[1] as u8;
                }
                (f, s)
            };
            let (fa, sa) = split(&e_ar_t);
            let (fb, sb) = split(&e_bl);

            for (i, row) in e_al.iter().enumerate() {
                unsafe { matvec_perm_128_addinto(row, &fa, &sa, &mut a_buf[i * k..i * k + k]) };
            }
            for (j, row) in e_br_t.iter().enumerate() {
                unsafe { matvec_perm_128_addinto(row, &fb, &sb, &mut b_buf[j * k..j * k + k]) };
            }
            return;
        }
    }
    // Fallback: reference noise, then element-wise add.
    let noise = compute_noise_for_indices_fast(
        k, noise_rank, commitment_hash, a_rows_indices, b_cols_indices,
    );
    for (i, nr) in noise.a.iter().enumerate() {
        let row = &mut a_buf[i * k..i * k + k];
        for l in 0..k { row[l] = row[l].wrapping_add(nr[l]); }
    }
    for (j, nr) in noise.b.iter().enumerate() {
        let row = &mut b_buf[j * k..j * k + k];
        for l in 0..k { row[l] = row[l].wrapping_add(nr[l]); }
    }
}

/// Bit-identical drop-in for `zk_pow::…::compute_noise_for_indices` using the
/// SIMD matvec when applicable. Reuses zk-pow's BLAKE3 matrix generation.
pub fn compute_noise_for_indices_fast(
    k: usize,
    noise_rank: usize,
    commitment_hash: ([u8; 32], [u8; 32]),
    a_rows_indices: &[usize],
    b_cols_indices: &[usize],
) -> MMSlice {
    #[cfg(target_arch = "x86_64")]
    {
        if fast_noise_applicable(noise_rank) {
            let (b_noise_seed, a_noise_seed) = commitment_hash;
            let e_al =
                generate_uniform_random_matrix(&SEED_LABEL_A, &a_noise_seed, a_rows_indices, noise_rank);
            let e_ar_t = generate_permutation_matrix(&SEED_LABEL_A, &a_noise_seed, k, noise_rank);
            let e_bl = generate_permutation_matrix(&SEED_LABEL_B, &b_noise_seed, k, noise_rank);
            let e_br_t =
                generate_uniform_random_matrix(&SEED_LABEL_B, &b_noise_seed, b_cols_indices, noise_rank);

            // Split the shared permutations into contiguous u8 index streams once
            // (reused across every row) so the kernel just streams two arrays.
            let split = |perm: &[[u32; 2]]| -> (Vec<u8>, Vec<u8>) {
                let mut f = vec![0u8; perm.len()];
                let mut s = vec![0u8; perm.len()];
                for (i, p) in perm.iter().enumerate() {
                    f[i] = p[0] as u8;
                    s[i] = p[1] as u8;
                }
                (f, s)
            };
            let (fa, sa) = split(&e_ar_t);
            let (fb, sb) = split(&e_bl);

            let mk = |rows: &[Vec<i8>], f: &[u8], s: &[u8]| -> Vec<Vec<i8>> {
                rows.iter()
                    .map(|row| {
                        let mut out = vec![0i8; k];
                        unsafe { matvec_perm_128(row, f, s, &mut out) };
                        out
                    })
                    .collect()
            };
            let a = mk(&e_al, &fa, &sa);
            let b = mk(&e_br_t, &fb, &sb);
            return MMSlice { a, b, routing: vec![] };
        }
    }
    // Fallback: the reference path (any rank / non-VBMI host).
    zk_pow::circuit::pearl_noise::compute_noise_for_indices(
        k,
        noise_rank,
        commitment_hash,
        a_rows_indices,
        b_cols_indices,
    )
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    /// SIMD noise must be byte-for-byte equal to the zk-pow reference.
    #[test]
    fn fast_noise_matches_reference() {
        if !fast_noise_applicable(128) {
            eprintln!("skip: no AVX-512 VBMI on this host");
            return;
        }
        let (k, rank) = (4096usize, 128usize);
        // Arbitrary but fixed commitment seeds.
        let a_seed = [0x37u8; 32];
        let b_seed = [0x9cu8; 32];
        let a_idx: Vec<usize> = (0..40).collect();
        let b_idx: Vec<usize> = (10..70).collect();

        let got = compute_noise_for_indices_fast(k, rank, (b_seed, a_seed), &a_idx, &b_idx);
        let reference = zk_pow::circuit::pearl_noise::compute_noise_for_indices(
            k, rank, (b_seed, a_seed), &a_idx, &b_idx,
        );
        assert_eq!(got.a, reference.a, "noise A diverges from reference");
        assert_eq!(got.b, reference.b, "noise B diverges from reference");
    }

    /// The fused noise+add path must equal: draw → compute_noise → wrapping_add.
    #[test]
    fn add_noise_into_fast_matches_two_step() {
        if !fast_noise_applicable(128) {
            eprintln!("skip: no AVX-512 VBMI on this host");
            return;
        }
        let (k, rank) = (4096usize, 128usize);
        let a_seed = [0x37u8; 32];
        let b_seed = [0x9cu8; 32];
        let (m_batch, n_batch) = (40usize, 60usize);
        let a_idx: Vec<usize> = (0..m_batch).collect();
        let b_idx: Vec<usize> = (0..n_batch).collect();

        // Arbitrary "drawn" strips (deterministic, no RNG dep in the test).
        let mut a0 = vec![0i8; m_batch * k];
        let mut b0 = vec![0i8; n_batch * k];
        for (i, v) in a0.iter_mut().enumerate() { *v = ((i * 31 + 7) as u8 as i8) >> 1; }
        for (i, v) in b0.iter_mut().enumerate() { *v = ((i * 17 + 3) as u8 as i8) >> 1; }

        // Reference two-step.
        let mut a_ref = a0.clone();
        let mut b_ref = b0.clone();
        let noise = compute_noise_for_indices_fast(k, rank, (b_seed, a_seed), &a_idx, &b_idx);
        for (i, nr) in noise.a.iter().enumerate() {
            let row = &mut a_ref[i * k..i * k + k];
            for l in 0..k { row[l] = row[l].wrapping_add(nr[l]); }
        }
        for (j, nr) in noise.b.iter().enumerate() {
            let row = &mut b_ref[j * k..j * k + k];
            for l in 0..k { row[l] = row[l].wrapping_add(nr[l]); }
        }

        // Fused.
        let mut a_fused = a0;
        let mut b_fused = b0;
        add_noise_into_fast(k, rank, (b_seed, a_seed), &a_idx, &b_idx, &mut a_fused, &mut b_fused);

        assert_eq!(a_fused, a_ref, "fused A diverges from two-step");
        assert_eq!(b_fused, b_ref, "fused B diverges from two-step");
    }
}
