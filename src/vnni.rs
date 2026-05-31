//! Explicit int8 dot product for the Pearl GEMM hot loop, via the VNNI
//! `vpdpbusd` instruction.
//!
//! `vpdpbusd` multiplies **u8 × i8 → i32** with a 4-deep accumulate. Pearl's
//! values are both signed i8, so we bias `a` to unsigned (`au = a + 128`, done
//! in-register with `XOR 0x80`) and correct the result:
//!
//! ```text
//! Σ a·b = Σ (au-128)·b = Σ au·b  −  128·Σ b
//! ```
//!
//! Both `Σ au·b` and `Σ b` are computed with `vpdpbusd` in the same pass
//! (the second one against a vector of ones). Everything stays in wrapping
//! i32, so the output is **bit-identical** to the scalar reference and remains
//! valid against zk-pow.
//!
//! Three paths, picked at runtime:
//!   - AVX-512 VNNI (Zen4/5, Intel w/ AVX-512) — 64 MACs/instr
//!   - AVX-VNNI 256-bit (Intel Arrow Lake, no AVX-512) — 32 MACs/instr
//!   - scalar fallback
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// `Σ a[l]·b[l]` as wrapping i32. `a` and `b` must have equal length.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512f")
        {
            return unsafe { dot_i8_avx512vnni(a, b) };
        }
        if is_x86_feature_detected!("avxvnni") {
            return unsafe { dot_i8_avxvnni(a, b) };
        }
    }
    dot_i8_scalar(a, b)
}

#[inline]
pub fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    let mut acc = 0i32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc = acc.wrapping_add((x as i32).wrapping_mul(y as i32));
    }
    acc
}

/// AVX-512 VNNI: 64 int8 MACs per `vpdpbusd`. 512-bit lanes (16 × i32).
/// `#[inline]` so it folds into same-feature callers (e.g. the per-tile loop)
/// — the dispatch must be hoisted out of the hot path, the kernel itself is
/// only ~2 instructions for r=128.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
pub unsafe fn dot_i8_avx512vnni(a: &[i8], b: &[i8]) -> i32 {
    let n = a.len();
    let bias = _mm512_set1_epi8(-128i8); // XOR 0x80 → i8 + 128 as u8
    let ones = _mm512_set1_epi8(1i8);
    let mut acc_ab = _mm512_setzero_si512();
    let mut acc_b = _mm512_setzero_si512();
    let mut l = 0usize;
    while l + 64 <= n {
        let va = _mm512_loadu_si512(a.as_ptr().add(l) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(l) as *const __m512i);
        let au = _mm512_xor_si512(va, bias);
        acc_ab = _mm512_dpbusd_epi32(acc_ab, au, vb);
        acc_b = _mm512_dpbusd_epi32(acc_b, ones, vb);
        l += 64;
    }
    let mut s = _mm512_reduce_add_epi32(acc_ab)
        .wrapping_sub(128i32.wrapping_mul(_mm512_reduce_add_epi32(acc_b)));
    while l < n {
        s = s.wrapping_add((*a.get_unchecked(l) as i32).wrapping_mul(*b.get_unchecked(l) as i32));
        l += 1;
    }
    s
}

/// AVX-VNNI (256-bit): 32 int8 MACs per `vpdpbusd`. For AVX-512-less CPUs
/// (Intel Arrow Lake / Core Ultra).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,avxvnni")]
pub unsafe fn dot_i8_avxvnni(a: &[i8], b: &[i8]) -> i32 {
    let n = a.len();
    let bias = _mm256_set1_epi8(-128i8);
    let ones = _mm256_set1_epi8(1i8);
    let mut acc_ab = _mm256_setzero_si256();
    let mut acc_b = _mm256_setzero_si256();
    let mut l = 0usize;
    while l + 32 <= n {
        let va = _mm256_loadu_si256(a.as_ptr().add(l) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(l) as *const __m256i);
        let au = _mm256_xor_si256(va, bias);
        // VEX-encoded AVX-VNNI vpdpbusd (`_avx_` suffix). The plain
        // `_mm256_dpbusd_epi32` is the EVEX/AVX-512VL encoding and SIGILLs on
        // AVX-512-less CPUs (Arrow Lake / Core Ultra) despite the `avxvnni`
        // target feature — this is the i9-285K path, so it MUST be the VEX form.
        acc_ab = _mm256_dpbusd_avx_epi32(acc_ab, au, vb);
        acc_b = _mm256_dpbusd_avx_epi32(acc_b, ones, vb);
        l += 32;
    }
    let mut s = hsum256_epi32(acc_ab).wrapping_sub(128i32.wrapping_mul(hsum256_epi32(acc_b)));
    while l < n {
        s = s.wrapping_add((*a.get_unchecked(l) as i32).wrapping_mul(*b.get_unchecked(l) as i32));
        l += 1;
    }
    s
}

/// Horizontal wrapping sum of the 8 i32 lanes of a 256-bit vector.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_epi32(v: __m256i) -> i32 {
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, v);
    let mut s = 0i32;
    for x in lanes {
        s = s.wrapping_add(x);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    #[test]
    fn vnni_matches_scalar_over_random_int7_plus_noise() {
        let mut rng = StdRng::seed_from_u64(0xA51A);
        // r=128 windows like Pearl; also a few odd lengths to exercise the tail.
        for &len in &[128usize, 256, 4096, 64, 32, 130, 257] {
            let mut a = vec![0i8; len];
            let mut b = vec![0i8; len];
            for buf in [&mut a, &mut b] {
                let bytes =
                    unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, len) };
                rng.fill_bytes(bytes);
            }
            let expected = dot_i8_scalar(&a, &b);
            let got = dot_i8(&a, &b);
            assert_eq!(expected, got, "VNNI dot diverges from scalar at len={len}");
        }
    }
}
