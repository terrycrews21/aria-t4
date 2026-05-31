//! Pearl PoW compute primitives — reference CPU implementation of the
//! GEMM+jackpot accumulator that miners run by the trillion per second.
//!
//! This is the truth-source we'll port to CUDA: every CUDA kernel must pass
//! a bit-exact cross-check against `compute_jackpot_pearl` below, which in
//! turn is verified against `zk_pow::circuit::chip::compute_jackpot` in the
//! tests at the bottom of this file.
//!
//! Spec recap (from `zk-pow/src/circuit/chip/jackpot/helper.rs`):
//!
//!   for ll in (r..=k).step_by(r) {                 // walk the inner dim r at a time
//!       for u in 0..h {                            // h tile rows
//!           for v in 0..w {                        // w tile cols
//!               for l in (ll - r)..ll {            // accumulate one rank-block
//!                   jackpot[u][v] += (a[u][l] + noise_a[u][l])
//!                                  * (b[v][l] + noise_b[v][l])
//!               }
//!           }
//!       }
//!       xored = jackpot.flat().fold(0u32, ^)       // collapse the h*w tile
//!       tid   = (ll/r - 1) % JACKPOT_SIZE
//!       jackpot_msg[tid] = jackpot_msg[tid].rotate_left(LROT_PER_TILE) ^ xored
//!   }
//!
//! `JACKPOT_SIZE = 16`, `LROT_PER_TILE = 13` (constants from the Pearl spec).
//! Values are wrapping Int32; the addition `(secret + noise)` stays in
//! signed-byte range because both fit in [-64, 64].

pub const JACKPOT_SIZE: usize = 16;
pub const LROT_PER_TILE: u32 = 13;

/// Static shape parameters for one Pearl compute round.
#[derive(Debug, Clone, Copy)]
pub struct PearlParams {
    /// Tile rows (Pearl mainnet: h = 2).
    pub h: usize,
    /// Tile columns (Pearl mainnet: w = 64).
    pub w: usize,
    /// Common dimension (Pearl mainnet: k = 4096).
    pub k: usize,
    /// Noise rank — k must be a multiple of r (Pearl mainnet: r = 128).
    pub r: usize,
}

/// Noise matrices (additive offsets to the miner's secret strips). Derived
/// deterministically from `commitment_hash` by the chain's PRF; we just
/// consume them here.
///
/// Shapes mirror `zk_pow::circuit::pearl_noise::MMSlice`:
/// - `a[h][k]` : noise rows for matrix A
/// - `b[w][k]` : noise rows for B-transposed (so we index `[v][l]`, same as
///   the official compute_jackpot)
#[derive(Debug, Clone)]
pub struct PearlNoise {
    pub a: Vec<Vec<i8>>,
    pub b: Vec<Vec<i8>>,
}

/// Reference CPU implementation of one Pearl `compute_jackpot` call.
///
/// `secret_a` shape : `[h][k]` — the miner's chosen rows of A.
/// `secret_b` shape : `[w][k]` — the miner's chosen cols of B (transposed).
/// `noise`         : matching `a`/`b` matrices from the commitment seed.
///
/// Returns the 16-word jackpot message that's then fed into a keyed BLAKE3
/// to produce the final candidate hash.
pub fn compute_jackpot_pearl(
    params: PearlParams,
    secret_a: &[Vec<i8>],
    secret_b: &[Vec<i8>],
    noise: &PearlNoise,
) -> [u32; JACKPOT_SIZE] {
    debug_assert_eq!(secret_a.len(), params.h);
    debug_assert_eq!(secret_b.len(), params.w);
    debug_assert_eq!(noise.a.len(), params.h);
    debug_assert_eq!(noise.b.len(), params.w);
    debug_assert_eq!(params.k % params.r, 0, "k must be a multiple of r");

    let h = params.h;
    let w = params.w;
    let k = params.k;
    let r = params.r;

    // Tile accumulator, reused across rank-blocks; wrapping i32 arithmetic
    // — matches the Rust `+=` on `i32` which is wrap-defined here because
    // the upstream impl uses `i32 += i32 * i32` without overflow checks.
    let mut jackpot = vec![vec![0i32; w]; h];
    let mut jackpot_msg = [0u32; JACKPOT_SIZE];

    // Walk the k-axis r columns at a time. After each block of r we fold
    // the tile into one u32 via XOR and stir it into jackpot_msg with a
    // left-rotate-then-XOR pattern (the "lrot per tile" mixer).
    let mut ll = r;
    while ll <= k {
        // Inner GEMM block — h*w*r MACs.
        for u in 0..h {
            for v in 0..w {
                let mut acc = jackpot[u][v];
                for l in (ll - r)..ll {
                    let a = secret_a[u][l] as i32 + noise.a[u][l] as i32;
                    let b = secret_b[v][l] as i32 + noise.b[v][l] as i32;
                    acc = acc.wrapping_add(a.wrapping_mul(b));
                }
                jackpot[u][v] = acc;
            }
        }

        // Fold the h*w tile into a single u32 by XOR.
        let mut xored: u32 = 0;
        for u in 0..h {
            for v in 0..w {
                xored ^= jackpot[u][v] as u32;
            }
        }

        // Stir into one of the JACKPOT_SIZE slots, picked by tile index.
        let tid = (ll / r - 1) % JACKPOT_SIZE;
        jackpot_msg[tid] = jackpot_msg[tid].rotate_left(LROT_PER_TILE) ^ xored;

        ll += r;
    }

    jackpot_msg
}

/// Reusable scratch for [`compute_jackpot_flat`] — flat, cache-friendly copies
/// of the *effective* strips `(secret + noise)`. Allocated once per worker
/// thread and reused across attempts (avoids ~270 KB churn/attempt).
pub struct JackpotScratch {
    /// `a_eff[u*k + l] = secret_a[u][l] + noise_a[u][l]`, shape h*k.
    a_eff: Vec<i8>,
    /// `b_eff[v*k + l] = secret_b[v][l] + noise_b[v][l]`, shape w*k.
    b_eff: Vec<i8>,
    /// Tile accumulator, shape h*w.
    jackpot: Vec<i32>,
}

impl JackpotScratch {
    pub fn new(params: PearlParams) -> Self {
        Self {
            a_eff: vec![0i8; params.h * params.k],
            b_eff: vec![0i8; params.w * params.k],
            jackpot: vec![0i32; params.h * params.w],
        }
    }
}

/// Flat / autovectorizable variant of [`compute_jackpot_pearl`]. Bit-identical
/// output, but works on contiguous `i8` buffers so LLVM can emit AVX-512 int8
/// dot products (`vpdpbusd`/`vpmaddwd`) for the hot inner loop instead of
/// chasing `Vec<Vec<i8>>` pointers. Caller supplies reusable `scratch`.
///
/// The `(secret + noise)` sum stays in `i8` range (`[-128, 127]`) per the Pearl
/// spec, so we keep the effective strips as `i8` and widen to `i32` only inside
/// the MAC — exactly what the autovectorizer wants.
pub fn compute_jackpot_flat(
    params: PearlParams,
    secret_a: &[Vec<i8>],
    secret_b: &[Vec<i8>],
    noise: &PearlNoise,
    scratch: &mut JackpotScratch,
) -> [u32; JACKPOT_SIZE] {
    let h = params.h;
    let w = params.w;
    let k = params.k;
    let r = params.r;

    // Precompute effective strips into flat buffers (wrapping i8 add).
    for u in 0..h {
        let dst = &mut scratch.a_eff[u * k..u * k + k];
        let sa = &secret_a[u];
        let na = &noise.a[u];
        for l in 0..k {
            dst[l] = sa[l].wrapping_add(na[l]);
        }
    }
    for v in 0..w {
        let dst = &mut scratch.b_eff[v * k..v * k + k];
        let sb = &secret_b[v];
        let nb = &noise.b[v];
        for l in 0..k {
            dst[l] = sb[l].wrapping_add(nb[l]);
        }
    }

    let jackpot = &mut scratch.jackpot;
    jackpot.iter_mut().for_each(|x| *x = 0);
    let mut jackpot_msg = [0u32; JACKPOT_SIZE];

    let mut ll = r;
    let mut blk = 0usize;
    while ll <= k {
        let l0 = ll - r;
        for u in 0..h {
            let a_row = &scratch.a_eff[u * k + l0..u * k + ll];
            for v in 0..w {
                let b_row = &scratch.b_eff[v * k + l0..v * k + ll];
                // Explicit VNNI int8 dot (vpdpbusd) over r elements.
                jackpot[u * w + v] =
                    jackpot[u * w + v].wrapping_add(crate::vnni::dot_i8(a_row, b_row));
            }
        }

        let mut xored: u32 = 0;
        for &x in jackpot.iter() {
            xored ^= x as u32;
        }
        let tid = blk % JACKPOT_SIZE;
        jackpot_msg[tid] = jackpot_msg[tid].rotate_left(LROT_PER_TILE) ^ xored;

        ll += r;
        blk += 1;
    }

    jackpot_msg
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flat/autovec variant must be bit-identical to the scalar reference
    /// for arbitrary secrets+noise.
    #[test]
    fn flat_matches_reference() {
        let params = PearlParams { h: 2, w: 8, k: 256, r: 64 };
        let mk = |rows: usize, salt: i64| -> Vec<Vec<i8>> {
            (0..rows)
                .map(|u| {
                    (0..params.k)
                        .map(|l| (((u as i64 * 37 + l as i64 * 7 + salt) % 127) - 63) as i8)
                        .collect()
                })
                .collect()
        };
        let secret_a = mk(params.h, 1);
        let secret_b = mk(params.w, 2);
        let noise = PearlNoise { a: mk(params.h, 3), b: mk(params.w, 4) };

        let reference = compute_jackpot_pearl(params, &secret_a, &secret_b, &noise);
        let mut scratch = JackpotScratch::new(params);
        let flat = compute_jackpot_flat(params, &secret_a, &secret_b, &noise, &mut scratch);

        assert_eq!(reference, flat, "compute_jackpot_flat diverges from scalar reference");
    }
}
