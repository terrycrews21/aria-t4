//! Sparse Int7×Int7→Int32 GEMM transcript builder for Pearl PoW.
//!
//! The pool tells us via `pearl.set_mining_params`:
//! - `m, n, k, rank` matrix dimensions (m×n result, k inner, rank decomposition)
//! - `rows_pattern` (~2 indices in PPLNS): which rows of the m×n product to compute
//! - `cols_pattern` (~64 indices): which cols to keep in the transcript
//! - `mma_type` = "Int7xInt7ToInt32" → tensor-core friendly format
//!
//! The CPU implementation here is reference / parity-check only. The real
//! mining loop calls into the CUDA kernels (cuda/pearl_gemm.cu, FFI in
//! `cuda_ffi.rs`).
//!
//! Pseudo-random generator: a BLAKE3-keyed PRF over (challenge_seed, row_idx,
//! col_idx). Pool uses the same PRF on the verification side; if our random
//! draws diverge by one byte the proof is rejected. The exact PRF must be
//! validated against a live capture before mainnet submission.

use crate::protocol::MiningParams;
use anyhow::Result;
use blake3::Hasher;

/// Result of one GEMM "tile" needed by the proof: the dot-product of selected
/// rows × cols, computed in Int32 accumulator.
#[derive(Debug, Clone)]
pub struct TranscriptCell {
    pub row: u32,
    pub col: u32,
    pub value: i32,
}

/// Builds the sparse transcript for a given challenge. Reference CPU impl —
/// slow but correct; used to seed and cross-check the CUDA kernel.
pub fn build_transcript_cpu(
    challenge_seed: &[u8; 32],
    params: &MiningParams,
) -> Result<Vec<TranscriptCell>> {
    let mut out = Vec::with_capacity(params.rows_pattern.len() * params.cols_pattern.len());
    for &row in &params.rows_pattern {
        for &col in &params.cols_pattern {
            let value = dot_product(challenge_seed, params, row, col);
            out.push(TranscriptCell { row, col, value });
        }
    }
    Ok(out)
}

/// One (row, col) cell of the GEMM. Uses a BLAKE3-derived PRF to materialize the
/// virtual matrices A (m × k) and B (k × n) on the fly. The dot product is
/// taken over the `k` inner dimension in Int7×Int7→Int32 arithmetic.
fn dot_product(seed: &[u8; 32], params: &MiningParams, row: u32, col: u32) -> i32 {
    let mut acc: i32 = 0;
    for kk in 0..params.k {
        let a = signed_int7(seed, 0, row, kk);
        let b = signed_int7(seed, 1, kk, col);
        acc = acc.wrapping_add(a as i32 * b as i32);
    }
    acc
}

/// Deterministic Int7 (signed, -64..=63) entry of the virtual matrix indexed by
/// `(matrix_id, i, j)`. Implemented as BLAKE3-keyed PRF, single byte read.
fn signed_int7(seed: &[u8; 32], matrix_id: u8, i: u32, j: u32) -> i8 {
    let mut h = Hasher::new_keyed(seed);
    h.update(&[matrix_id]);
    h.update(&i.to_le_bytes());
    h.update(&j.to_le_bytes());
    let bytes = h.finalize();
    // Take the low byte, mask to 7 bits then sign-extend.
    let raw = bytes.as_bytes()[0] & 0x7f;
    let val = if raw & 0x40 != 0 { (raw as i8) | -0x40i8 } else { raw as i8 };
    val
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_params() -> MiningParams {
        MiningParams {
            m: 8, n: 8, k: 4, rank: 1,
            rows_pattern: vec![0, 3],
            cols_pattern: vec![0, 1, 2, 3, 4, 5, 6, 7],
            mma_type: "Int7xInt7ToInt32".into(),
        }
    }

    #[test]
    fn transcript_is_deterministic() {
        let seed = [0x42u8; 32];
        let p = tiny_params();
        let a = build_transcript_cpu(&seed, &p).unwrap();
        let b = build_transcript_cpu(&seed, &p).unwrap();
        assert_eq!(a.len(), p.rows_pattern.len() * p.cols_pattern.len());
        for (ca, cb) in a.iter().zip(b.iter()) {
            assert_eq!(ca.value, cb.value);
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let p = tiny_params();
        let a = build_transcript_cpu(&[0u8; 32], &p).unwrap();
        let b = build_transcript_cpu(&[1u8; 32], &p).unwrap();
        assert_ne!(a[0].value, b[0].value);
    }

    #[test]
    fn int7_is_signed_range() {
        for i in 0..1000u32 {
            let v = signed_int7(&[0u8; 32], 0, i, 0);
            assert!(v >= -64 && v <= 63, "int7 out of range: {v}");
        }
    }
}
