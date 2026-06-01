//! Official-faithful Pearl grind (correctness-first reference).
//!
//! A byte-faithful reimplementation of `zk_pow::ffi::mine::try_mine_one`, with
//! ONE deliberate change: the difficulty bound is supplied by the caller instead
//! of being derived from `header.nbits`. That lets us grind at the **pool share**
//! difficulty while keeping the **real** block header (so `job_key =
//! blake3(header‖config)` and the commitment chain match what the network
//! expects). The produced `PlainProof` is the canonical one and passes the
//! node-side `verify_plain_proof` / `parse_plain_proof`.
//!
//! This module is intentionally ISOLATED from the optimized `cpu_engine` (the
//! production hashrate path): correctness first, perf later. Everything the
//! official miner does on the commitment side is reproduced exactly:
//!   - `job_key = blake3(header.to_bytes() ‖ config.to_bytes())`
//!   - `hash_a/hash_b = blake3_keyed(pad_to_chunk_boundary(flatten(M)), job_key)`
//!   - tiles taken at the **PeriodicPattern** offsets (not contiguous)
//!   - difficulty compared as a **little-endian** 256-bit int (`jackpot ≤ bound`)
//!   - Merkle proof built over the **signal** (pre-noise) matrix.

use pearl_blake3::{MerkleTree, blake3_digest, pad_to_chunk_boundary};
use rand::Rng;
use zk_pow::api::proof::{IncompleteBlockHeader, MiningConfiguration, PeriodicPattern};
use zk_pow::api::proof_utils::compute_jackpot_hash;
use zk_pow::ffi::plain_proof::{MatrixMerkleProof, PlainProof};

use crate::pearl_compute::LROT_PER_TILE;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Pearl signal values are valid in [-64, 64]; we draw into the strict subset
// [-64, 63] (one bulk mask, see `to_signal`), which keeps every proof valid.
const SIGNAL_MIN: i8 = -64;
/// Jackpot accumulator words (`compute_jackpot_hash` consumes `&[u32; 16]`).
const JACKPOT_SIZE: usize = 16;

/// Which int8-dot kernel the running CPU supports. Resolved ONCE per setup
/// (feature detection hoisted out of the per-cell hot loop), then the matching
/// `tile_jackpot_*` runs the whole tile sweep with that kernel inlined.
#[derive(Clone, Copy)]
enum DotKernel {
    #[cfg(target_arch = "x86_64")]
    Avx512Vnni,
    #[cfg(target_arch = "x86_64")]
    AvxVnni,
    Scalar,
}

fn detect_dot_kernel() -> DotKernel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512f")
        {
            return DotKernel::Avx512Vnni;
        }
        if is_x86_feature_detected!("avxvnni") {
            return DotKernel::AvxVnni;
        }
    }
    DotKernel::Scalar
}

/// Body of one tile's jackpot sweep, parameterised by the dot expression so the
/// SIMD kernel folds directly into a same-`target_feature` wrapper (no per-cell
/// call, no per-cell feature detection). Returns the `[u32; JACKPOT_SIZE]`
/// accumulator; the caller hashes it and checks the bound.
macro_rules! tile_jackpot_body {
    ($a_rows:expr, $b_cols:expr, $a_noised:expr, $b_noised_t:expr, $rank:expr, $k:expr, $dot:expr) => {{
        let a_rows = $a_rows;
        let b_cols = $b_cols;
        let mut jackpot_tile = vec![vec![0i32; b_cols.len()]; a_rows.len()];
        let mut jackpot = [0u32; JACKPOT_SIZE];
        let mut ll = $rank;
        while ll <= $k {
            for (u, &a_idx) in a_rows.iter().enumerate() {
                for (v, &b_idx) in b_cols.iter().enumerate() {
                    jackpot_tile[u][v] +=
                        $dot(&$a_noised[a_idx][ll - $rank..ll], &$b_noised_t[b_idx][ll - $rank..ll]);
                }
            }
            let xored_tile = jackpot_tile.iter().flatten().fold(0u32, |acc, &x| acc ^ x as u32);
            let tid = (ll / $rank - 1) % JACKPOT_SIZE;
            jackpot[tid] = jackpot[tid].rotate_left(LROT_PER_TILE as u32) ^ xored_tile;
            ll += $rank;
        }
        jackpot
    }};
}

fn tile_jackpot_scalar(
    a_rows: &[usize],
    b_cols: &[usize],
    a_noised: &[Vec<i8>],
    b_noised_t: &[Vec<i8>],
    rank: usize,
    k: usize,
) -> [u32; JACKPOT_SIZE] {
    tile_jackpot_body!(a_rows, b_cols, a_noised, b_noised_t, rank, k, crate::vnni::dot_i8_scalar)
}

/// Register-blocked AVX-512 VNNI tile microkernel. Same result as the per-cell
/// `dot_i8_avx512vnni` summed into `jackpot_tile`, but with the unsigned-bias
/// term `Σb[v]` (which `vpdpbusd` forces and which is **independent of the A
/// row `u`**) computed ONCE per (window, col) and reused across all `tile_h`
/// rows — instead of recomputing it for every (u, v) cell. Bit-identical to the
/// scalar/per-cell path (oracle-guarded): `Σa·b = Σ(a+128)·b − 128·Σb`, the
/// chunked `vpdpbusd` covers the 64-wide body and a scalar tail handles the
/// remainder exactly like `dot_i8_avx512vnni`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn tile_jackpot_avx512(
    a_rows: &[usize],
    b_cols: &[usize],
    a_noised: &[Vec<i8>],
    b_noised_t: &[Vec<i8>],
    rank: usize,
    k: usize,
) -> [u32; JACKPOT_SIZE] {
    let th = a_rows.len();
    let tw = b_cols.len();
    let n_chunks = rank / 64; // full 64-wide vpdpbusd chunks
    // Stack buffers (no per-tile heap alloc). Caps cover every real Pearl tile
    // shape (th·tw ≤ 256, tw ≤ 128, rank ≤ 4096); guarded in debug.
    const MAX_CELLS: usize = 256;
    const MAX_COLS: usize = 128;
    const MAX_CHUNKS: usize = 64;
    debug_assert!(th * tw <= MAX_CELLS && tw <= MAX_COLS && n_chunks <= MAX_CHUNKS);
    let ones = _mm512_set1_epi8(1);
    let bias = _mm512_set1_epi8(-128i8); // XOR 0x80 → (i8 + 128) as u8

    let mut jackpot_tile = [0i32; MAX_CELLS];
    let mut jackpot = [0u32; JACKPOT_SIZE];
    let mut sumb = [0i32; MAX_COLS]; // Σb over the chunk body, per col (bias term)
    // biased A-row chunks, reused across the v-loop. Zero-initialised (one vxor
    // splatted — negligible vs the tile MACs) rather than left uninit, which is
    // unsound for `__m512i`. Only [0..n_chunks) is ever written/read.
    let mut au = [_mm512_setzero_si512(); MAX_CHUNKS];
    let mut ll = rank;
    let mut widx = 0usize;
    while ll <= k {
        let off = ll - rank;

        // ── bias term, hoisted out of the u-loop: sumb[v] = Σ_body b[v].
        // (A batched 16×16 transpose-reduce was tried here and LOST: the 16
        // per-cell `_mm512_reduce_add` are independent and pipeline on the OoO
        // engine, while the transpose is a serial dependency chain + extra
        // memory traffic. Per-cell reduce wins — keep it.)
        for (v, &b_idx) in b_cols.iter().enumerate() {
            let bp = b_noised_t[b_idx].as_ptr().add(off);
            let mut acc = _mm512_setzero_si512();
            let mut c = 0;
            while c < n_chunks {
                acc = _mm512_dpbusd_epi32(acc, ones, _mm512_loadu_si512(bp.add(c * 64) as *const __m512i));
                c += 1;
            }
            sumb[v] = _mm512_reduce_add_epi32(acc);
        }

        // ── per A-row: ab[v] = Σ_body (a+128)·b, then C = ab − 128·sumb + tail.
        for (u, &a_idx) in a_rows.iter().enumerate() {
            let ap = a_noised[a_idx].as_ptr().add(off);
            let mut c = 0;
            while c < n_chunks {
                au[c] = _mm512_xor_si512(_mm512_loadu_si512(ap.add(c * 64) as *const __m512i), bias);
                c += 1;
            }
            for (v, &b_idx) in b_cols.iter().enumerate() {
                let bp = b_noised_t[b_idx].as_ptr().add(off);
                let mut acc = _mm512_setzero_si512();
                let mut c = 0;
                while c < n_chunks {
                    acc = _mm512_dpbusd_epi32(acc, au[c], _mm512_loadu_si512(bp.add(c * 64) as *const __m512i));
                    c += 1;
                }
                let mut cell = _mm512_reduce_add_epi32(acc).wrapping_sub(128i32.wrapping_mul(sumb[v]));
                // scalar tail (rank % 64), signed exactly like dot_i8_avx512vnni
                let mut l = n_chunks * 64;
                if l < rank {
                    while l < rank {
                        cell = cell.wrapping_add((*ap.add(l) as i32).wrapping_mul(*bp.add(l) as i32));
                        l += 1;
                    }
                }
                let idx = u * tw + v;
                jackpot_tile[idx] = jackpot_tile[idx].wrapping_add(cell);
            }
        }

        let xored = jackpot_tile[..th * tw].iter().fold(0u32, |acc, &x| acc ^ x as u32);
        let tid = widx % JACKPOT_SIZE;
        jackpot[tid] = jackpot[tid].rotate_left(LROT_PER_TILE as u32) ^ xored;
        ll += rank;
        widx += 1;
    }
    jackpot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn tile_jackpot_avxvnni(
    a_rows: &[usize],
    b_cols: &[usize],
    a_noised: &[Vec<i8>],
    b_noised_t: &[Vec<i8>],
    rank: usize,
    k: usize,
) -> [u32; JACKPOT_SIZE] {
    tile_jackpot_body!(a_rows, b_cols, a_noised, b_noised_t, rank, k, |x: &[i8], y: &[i8]| unsafe {
        crate::vnni::dot_i8_avxvnni(x, y)
    })
}

/// Dispatch one tile sweep to the resolved kernel (decision made once per setup).
#[inline]
fn tile_jackpot(
    kernel: DotKernel,
    a_rows: &[usize],
    b_cols: &[usize],
    a_noised: &[Vec<i8>],
    b_noised_t: &[Vec<i8>],
    rank: usize,
    k: usize,
) -> [u32; JACKPOT_SIZE] {
    match kernel {
        #[cfg(target_arch = "x86_64")]
        DotKernel::Avx512Vnni => unsafe {
            tile_jackpot_avx512(a_rows, b_cols, a_noised, b_noised_t, rank, k)
        },
        #[cfg(target_arch = "x86_64")]
        DotKernel::AvxVnni => unsafe {
            tile_jackpot_avxvnni(a_rows, b_cols, a_noised, b_noised_t, rank, k)
        },
        DotKernel::Scalar => tile_jackpot_scalar(a_rows, b_cols, a_noised, b_noised_t, rank, k),
    }
}

/// `a ≤ b` where both are 256-bit integers in little-endian byte order — the
/// exact comparison the node uses (`U256::from_little_endian(hash) ≤ bound`).
fn le_leq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    true
}

fn flatten_matrix(matrix: &[Vec<i8>]) -> Vec<u8> {
    matrix.iter().flatten().map(|&x| x as u8).collect()
}

/// `blake3(header.to_bytes() ‖ config.to_bytes())` — the canonical job key.
fn compute_job_key(header: &IncompleteBlockHeader, config: &MiningConfiguration) -> [u8; 32] {
    let mut data = Vec::with_capacity(IncompleteBlockHeader::SERIALIZED_SIZE + MiningConfiguration::SERIALIZED_SIZE);
    data.extend_from_slice(&header.to_bytes());
    data.extend_from_slice(&config.to_bytes());
    blake3_digest(&data, None)
}

/// Returns `(b_noise_seed, a_noise_seed)`. Keyed-blake3 over the chunk-padded
/// matrices, then the two-step seed stir — identical to the official
/// `compute_commitment_hash`.
fn compute_commitment_hash(
    job_key: &[u8; 32],
    a_row_major: &[u8],
    b_col_major: &[u8],
) -> ([u8; 32], [u8; 32]) {
    let hash_a = blake3_digest(a_row_major, Some(*job_key));
    let hash_b = blake3_digest(b_col_major, Some(*job_key));

    let mut b_seed_input = [0u8; 64];
    b_seed_input[..32].copy_from_slice(job_key);
    b_seed_input[32..].copy_from_slice(&hash_b);
    let b_noise_seed = blake3_digest(&b_seed_input, None);

    let mut a_seed_input = [0u8; 64];
    a_seed_input[..32].copy_from_slice(&b_noise_seed);
    a_seed_input[32..].copy_from_slice(&hash_a);
    let a_noise_seed = blake3_digest(&a_seed_input, None);

    (b_noise_seed, a_noise_seed)
}

/// Build a `MatrixMerkleProof` over the **signal** matrix, keyed by `job_key` —
/// mirror of the official `build_matrix_proof`. The Merkle root equals
/// `hash_a`/`hash_b`, which is exactly what `parse_plain_proof` re-checks.
fn build_matrix_proof(
    matrix: &[Vec<i8>],
    job_key: &[u8; 32],
    row_indices: &[usize],
    num_cols: usize,
) -> MatrixMerkleProof {
    let padded = pad_to_chunk_boundary(&flatten_matrix(matrix));
    let tree = MerkleTree::new(&padded, *job_key);
    let leaf_indices = MerkleTree::compute_leaf_indices_from_rows(row_indices, (matrix.len(), num_cols));
    let proof = tree.get_multileaf_proof(&leaf_indices);
    MatrixMerkleProof {
        proof,
        row_indices: row_indices.to_vec(),
    }
}

/// Expand a `PeriodicPattern` into the absolute index tiles that partition a
/// dimension of length `total_dimension` — mirror of the official
/// `threads_partition`.
fn threads_partition(pattern: &PeriodicPattern, total_dimension: usize) -> Vec<Vec<usize>> {
    let period = pattern.period() as usize;
    assert!(
        total_dimension.is_multiple_of(period),
        "total_dimension {total_dimension} must be divisible by pattern period {period}"
    );
    let base_indices: Vec<usize> = pattern.to_list().iter().map(|&i| i as usize).collect();
    (0..total_dimension)
        .filter(|&i| pattern.offset_is_valid(i as u32))
        .map(|offset| base_indices.iter().map(|&d| offset + d).collect())
        .collect()
}

/// One faithful mining attempt against a caller-supplied little-endian `bound`.
/// Draws random signal matrices A (m×k) and B (k×n), applies the official noise,
/// and sweeps every `PeriodicPattern` tile; returns the first tile whose jackpot
/// hash is `≤ bound` as a canonical `PlainProof`. `None` if no tile qualifies.
///
/// Mirrors `zk_pow::ffi::mine::try_mine_one` with `wrong_jackpot_hash = false`.
pub fn try_mine_one_bounded<R: Rng>(
    rng: &mut R,
    m: usize,
    n: usize,
    k: usize,
    header: &IncompleteBlockHeader,
    config: &MiningConfiguration,
    bound_le: &[u8; 32],
) -> Option<PlainProof> {
    let rank = config.rank as usize;

    // Draw the signal matrices fast: one bulk RNG fill per matrix + a mask into
    // [-64, 63] (a strict subset of the allowed [-64, 64], so the proof stays
    // valid). A and B are the miner's FREE choice — only validity matters, not
    // the exact distribution — so we skip rand's per-element `gen_range`
    // rejection sampling (the dominant draw cost). B is drawn directly in the
    // **transposed** (n×k) layout that every downstream consumer (commitment,
    // noise, proof, tile sweep) wants, so there is no separate transpose pass.
    let to_signal = |b: u8| ((b & 0x7F) as i8) + SIGNAL_MIN;
    let mut a_bytes = vec![0u8; m * k];
    let mut bt_bytes = vec![0u8; n * k];
    rng.fill(&mut a_bytes[..]);
    rng.fill(&mut bt_bytes[..]);
    let a_matrix: Vec<Vec<i8>> =
        a_bytes.chunks_exact(k).map(|c| c.iter().map(|&b| to_signal(b)).collect()).collect();
    let b_transposed: Vec<Vec<i8>> =
        bt_bytes.chunks_exact(k).map(|c| c.iter().map(|&b| to_signal(b)).collect()).collect();

    let job_key = compute_job_key(header, config);
    let a_row_major = pad_to_chunk_boundary(&flatten_matrix(&a_matrix));
    let b_col_major = pad_to_chunk_boundary(&flatten_matrix(&b_transposed));
    let (b_noise_seed, a_noise_seed) = compute_commitment_hash(&job_key, &a_row_major, &b_col_major);

    // Noise over the whole matrices (official reference PRF).
    let a_all_rows: Vec<usize> = (0..m).collect();
    let b_all_cols: Vec<usize> = (0..n).collect();
    // Bit-identical SIMD (VBMI) drop-in for the reference PRF; falls back to the
    // reference on non-VBMI hosts / non-128 rank. Oracle-guarded.
    let noise = crate::fast_noise::compute_noise_for_indices_fast(
        k,
        rank,
        (b_noise_seed, a_noise_seed),
        &a_all_rows,
        &b_all_cols,
    );

    // a_noised / b_noised are signal (i8, [-64,64]) + noise (i8, [-63,63]) ⇒ in
    // [-127,127] ⇒ they fit in i8 with no overflow. Storing them as i8 (instead of
    // the reference i32) lets the rank-window dot product run on the vpdpbusd
    // (AVX-VNNI "Int8") datapath — int7 GEMM computed on int8 lanes — exactly what
    // production miners do. The integer result is identical (dot_i8 is tested
    // bit-equal to scalar over int7+noise), so the jackpot/proof stay canonical.
    let a_noised: Vec<Vec<i8>> = a_matrix
        .iter()
        .zip(&noise.a)
        .map(|(a_row, n_row)| a_row.iter().zip(n_row).map(|(&a, &n)| (a as i32 + n as i32) as i8).collect())
        .collect();
    // b_noised_t[i][l] = b_transposed[i][l] + noise.b[i][l] (noise.b is the n×k
    // col-major noise) — built directly, no intermediate b_noised + transpose.
    let b_noised_t: Vec<Vec<i8>> = b_transposed
        .iter()
        .zip(&noise.b)
        .map(|(b_row, n_row)| b_row.iter().zip(n_row).map(|(&b, &n)| (b as i32 + n as i32) as i8).collect())
        .collect();

    // Feature detection hoisted out of the per-cell loop: decided once, then the
    // tile sweep runs with the SIMD kernel inlined (no per-call dispatch).
    let kernel = detect_dot_kernel();

    for a_rows in threads_partition(&config.rows_pattern, m) {
        for b_cols in threads_partition(&config.cols_pattern, n) {
            let jackpot =
                tile_jackpot(kernel, &a_rows, &b_cols, &a_noised, &b_noised_t, rank, k);

            let jackpot_hash = compute_jackpot_hash(&jackpot, a_noise_seed);
            if le_leq(&jackpot_hash, bound_le) {
                let a_proof = build_matrix_proof(&a_matrix, &job_key, &a_rows, k);
                let b_proof = build_matrix_proof(&b_transposed, &job_key, &b_cols, k);
                return Some(PlainProof {
                    m,
                    n,
                    k,
                    noise_rank: rank,
                    a: a_proof,
                    bt: b_proof,
                });
            }
        }
    }

    None
}

/// Grind attempts until one tile beats `bound_le`. Returns the canonical proof.
pub fn mine_share<R: Rng>(
    rng: &mut R,
    m: usize,
    n: usize,
    k: usize,
    header: &IncompleteBlockHeader,
    config: &MiningConfiguration,
    bound_le: &[u8; 32],
) -> PlainProof {
    loop {
        if let Some(p) = try_mine_one_bounded(rng, m, n, k, header, config, bound_le) {
            return p;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use zk_pow::api::proof::{MMAType, MiningConfiguration, PeriodicPattern};

    fn test_config(k: u32) -> MiningConfiguration {
        MiningConfiguration {
            common_dim: k,
            rank: 128,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[
                0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105,
            ])
            .unwrap(),
            reserved: MiningConfiguration::RESERVED_VALUE,
        }
    }

    fn easy_header(nbits: u32) -> IncompleteBlockHeader {
        IncompleteBlockHeader {
            version: 0,
            prev_block: [1u8; 32],
            merkle_root: [2u8; 32],
            timestamp: 0x6666_6666,
            nbits,
        }
    }

    /// Key property: a proof produced by OUR grind (not by
    /// zk-pow's `mine`) must pass the official node-side verifiers — both the
    /// structural `parse_plain_proof` (Merkle roots == hash_a/hash_b) and the
    /// full `verify_plain_proof` (roots + difficulty under the real header). The
    /// bound is the genuine network bound for this (easy) header, so the winning
    /// tile truly satisfies the official little-endian difficulty test.
    #[test]
    fn our_grind_proof_passes_official_verify() {
        let (m, n, k) = (256usize, 128usize, 4032usize);
        let header = easy_header(0x207f_ffff);
        let config = test_config(k as u32);

        let bound = zk_pow::api::sanity_checks::extract_difficulty_bound(header.nbits, &config);
        let mut bound_le = [0u8; 32];
        bound.to_little_endian(&mut bound_le);

        let mut rng = StdRng::seed_from_u64(0xA51A_C0FFEE);
        let proof = mine_share(&mut rng, m, n, k, &header, &config, &bound_le);

        // Structural contract the pool relies on before building the zk-cert.
        zk_pow::ffi::plain_proof::parse_plain_proof(header, &proof)
            .expect("our proof must parse (roots == hash_a/hash_b)");
        // Full plain (non-zk) verification: roots + jackpot difficulty.
        zk_pow::api::verify::verify_plain_proof(&header, &proof)
            .expect("our proof must pass verify_plain_proof");

        // And it survives our canonical wire codec untouched.
        let b64 = crate::official_proof::encode_base64(&proof).unwrap();
        let back = crate::official_proof::decode_base64(&b64).unwrap();
        assert_eq!(
            crate::official_proof::encode_bytes(&proof).unwrap(),
            crate::official_proof::encode_bytes(&back).unwrap(),
        );
    }

    /// `le_leq` is the difficulty comparator; guard its byte-order semantics.
    #[test]
    fn le_leq_orders_by_le_integer_value() {
        let mut small = [0u8; 32];
        small[0] = 1; // value 1
        let mut big = [0u8; 32];
        big[31] = 1; // value 2^248
        assert!(le_leq(&small, &big));
        assert!(!le_leq(&big, &small));
        assert!(le_leq(&small, &small));
    }
}
