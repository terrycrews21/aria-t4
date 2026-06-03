//! Official-faithful Pearl grind — chantier 1, brique B (correctness-first).
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

// Pearl signal values are valid in [-64, 64]; we draw into the strict subset
// [-64, 63] (one bulk mask, see `to_signal`), which keeps every proof valid.
const SIGNAL_MIN: i8 = -64;
/// Jackpot accumulator words (`compute_jackpot_hash` consumes `&[u32; 16]`).
const JACKPOT_SIZE: usize = 16;

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
/// Per-thread reusable scratch. All the big buffers one grind attempt needs,
/// kept alive across attempts so the hot loop allocates NOTHING — the
/// per-attempt alloc/free churn (thousands of small `Vec<Vec>` rows) was the
/// multi-core scaling wall (e.g. a 192-thread EPYC collapsing). Buffers are flat
/// (`m*k` / `n*k`); the `Vec<Vec>` views the proof needs are rebuilt only on a
/// winning tile (rare). Grow-only — sized up to the largest job seen.
#[derive(Default)]
pub struct Workspace {
    a_bytes: Vec<u8>,
    b_bytes: Vec<u8>,
    a_sig: Vec<i8>,      // signal A, row-major (m*k)
    b_sig: Vec<i8>,      // signal B, transposed (n*k)
    a_noised: Vec<i8>,   // A + noise (m*k)
    b_noised_t: Vec<i8>, // Bᵀ + noise (n*k)
    a_tiled: Vec<i8>,
    b_all: Vec<i8>,
    b_packed: Vec<i8>,
    scratch: Vec<i32>,
    a_idx: Vec<usize>,
    b_idx: Vec<usize>,
}

impl Workspace {
    pub fn new() -> Self {
        Self::default()
    }
    fn ensure(&mut self, m: usize, n: usize, k: usize) {
        let (mk, nk) = (m * k, n * k);
        if self.a_bytes.len() < mk {
            self.a_bytes.resize(mk, 0);
            self.a_sig.resize(mk, 0);
            self.a_noised.resize(mk, 0);
            self.a_tiled.resize(mk, 0);
        }
        if self.b_bytes.len() < nk {
            self.b_bytes.resize(nk, 0);
            self.b_sig.resize(nk, 0);
            self.b_noised_t.resize(nk, 0);
            self.b_all.resize(nk, 0);
            self.b_packed.resize(nk, 0);
        }
        if self.scratch.len() < 512 {
            self.scratch.resize(512, 0);
        }
        if self.a_idx.len() != m {
            self.a_idx = (0..m).collect();
        }
        if self.b_idx.len() != n {
            self.b_idx = (0..n).collect();
        }
    }
}

#[inline]
fn as_u8(s: &[i8]) -> &[u8] {
    // i8 and u8 share layout; reinterpret for the byte-oriented BLAKE3 commitment.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len()) }
}

/// Rebuild the canonical `PlainProof` for a winning tile from the flat signal
/// buffers. Only called on a hit (rare), so the `Vec<Vec>` reconstruction cost
/// is irrelevant.
#[allow(clippy::too_many_arguments)]
fn make_proof(
    a_sig: &[i8],
    b_sig: &[i8],
    m: usize,
    n: usize,
    k: usize,
    rank: usize,
    job_key: &[u8; 32],
    a_rows: &[usize],
    b_cols: &[usize],
) -> PlainProof {
    let a_vv: Vec<Vec<i8>> = a_sig.chunks_exact(k).map(|c| c.to_vec()).collect();
    let b_vv: Vec<Vec<i8>> = b_sig.chunks_exact(k).map(|c| c.to_vec()).collect();
    let a = build_matrix_proof(&a_vv, job_key, a_rows, k);
    let bt = build_matrix_proof(&b_vv, job_key, b_cols, k);
    PlainProof { m, n, k, noise_rank: rank, a, bt }
}

/// Per-cell tile jackpot over the FLAT noised buffers — used by the fallback
/// (non-AVX-512 / non-h2×w64 shapes) and the AVX-VNNI remainder. Bit-identical
/// to the reference (window XOR + rotate, `dot_i8` per cell). Stack scratch, no
/// allocation.
fn tile_jackpot_flat(
    a_rows: &[usize],
    b_cols: &[usize],
    a_noised: &[i8],
    b_noised_t: &[i8],
    rank: usize,
    k: usize,
) -> [u32; JACKPOT_SIZE] {
    let th = a_rows.len();
    let tw = b_cols.len();
    let mut jackpot_tile = [0i32; 256];
    let mut jackpot = [0u32; JACKPOT_SIZE];
    let mut ll = rank;
    while ll <= k {
        for (u, &ai) in a_rows.iter().enumerate() {
            for (v, &bi) in b_cols.iter().enumerate() {
                jackpot_tile[u * tw + v] += crate::vnni::dot_i8(
                    &a_noised[ai * k + ll - rank..ai * k + ll],
                    &b_noised_t[bi * k + ll - rank..bi * k + ll],
                );
            }
        }
        let xored = jackpot_tile[..th * tw].iter().fold(0u32, |acc, &x| acc ^ x as u32);
        let tid = (ll / rank - 1) % JACKPOT_SIZE;
        jackpot[tid] = jackpot[tid].rotate_left(LROT_PER_TILE as u32) ^ xored;
        ll += rank;
    }
    jackpot
}

pub fn try_mine_one_bounded<R: Rng>(
    ws: &mut Workspace,
    rng: &mut R,
    m: usize,
    n: usize,
    k: usize,
    header: &IncompleteBlockHeader,
    config: &MiningConfiguration,
    bound_le: &[u8; 32],
) -> Option<PlainProof> {
    let rank = config.rank as usize;
    ws.ensure(m, n, k);
    let (mk, nk) = (m * k, n * k);

    // Draw signal A and Bᵀ into REUSED flat buffers: bulk RNG fill + mask into
    // [-64,63] (a valid subset; A/B are the miner's free choice). No allocation.
    let to_signal = |b: u8| ((b & 0x7F) as i8) + SIGNAL_MIN;
    rng.fill(&mut ws.a_bytes[..mk]);
    rng.fill(&mut ws.b_bytes[..nk]);
    for i in 0..mk {
        ws.a_sig[i] = to_signal(ws.a_bytes[i]);
    }
    for i in 0..nk {
        ws.b_sig[i] = to_signal(ws.b_bytes[i]);
    }

    let job_key = compute_job_key(header, config);
    let a_row_major = pad_to_chunk_boundary(as_u8(&ws.a_sig[..mk]));
    let b_col_major = pad_to_chunk_boundary(as_u8(&ws.b_sig[..nk]));
    let (b_noise_seed, a_noise_seed) = compute_commitment_hash(&job_key, &a_row_major, &b_col_major);

    // Noise: copy signal → noised buffers, then add the official noise IN PLACE.
    // `add_noise_into_fast` is bit-identical to `compute_noise_for_indices_fast`
    // + element-wise wrapping add, but allocates no intermediate noise matrices.
    ws.a_noised[..mk].copy_from_slice(&ws.a_sig[..mk]);
    ws.b_noised_t[..nk].copy_from_slice(&ws.b_sig[..nk]);
    crate::fast_noise::add_noise_into_fast(
        k,
        rank,
        (b_noise_seed, a_noise_seed),
        &ws.a_idx[..m],
        &ws.b_idx[..n],
        &mut ws.a_noised[..mk],
        &mut ws.b_noised_t[..nk],
    );

    let tile_h = config.rows_pattern.to_list().len();
    let tile_w = config.cols_pattern.to_list().len();

    // ── FAST PATH: the real Pearl tile shape (h=2 row-offsets × w=64 col-offsets)
    // on AVX-512 VNNI runs the register-blocked micro-kernel, which
    // is bit-identical to the per-cell tile (oracle-guarded) — it just computes the
    // SAME jackpot far faster. The proof is still built from the signal matrices +
    // pattern indices, exactly like the fallback, so shares stay canonical.
    #[cfg(target_arch = "x86_64")]
    if tile_h == 2 && tile_w == 64 && k % 4 == 0 {
        let avx512 = is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512f");
        // AVX-VNNI (256-bit) path for AVX-512-less CPUs (Intel Arrow Lake / Core
        // Ultra — the i9 cluster). Same micro-kernel family, 8-col groups.
        let avxvnni = !avx512 && is_x86_feature_detected!("avxvnni");
        if avx512 || avxvnni {
            let row_tiles = threads_partition(&config.rows_pattern, m);
            let col_tiles = threads_partition(&config.cols_pattern, n);
            let nt = row_tiles.len();
            // A noised, TILE-ORDERED: tile t's two rows at 2t,2t+1, so 4 tiles = 8
            // contiguous rows for the 8×64 kernel. Built once per setup.
            for (t, rt) in row_tiles.iter().enumerate() {
                let (r0, r1) = (rt[0] * k, rt[1] * k);
                ws.a_tiled[(2 * t) * k..(2 * t) * k + k].copy_from_slice(&ws.a_noised[r0..r0 + k]);
                ws.a_tiled[(2 * t + 1) * k..(2 * t + 1) * k + k].copy_from_slice(&ws.a_noised[r1..r1 + k]);
            }
            // ALL B columns gathered tile-ordered, packed ONCE. Packed-once + the
            // row-panel loop mean A is read from DRAM once per panel, not once per
            // col-tile (A re-streaming was the multi-thread bandwidth wall).
            let total_cols = col_tiles.len() * tile_w;
            for (ci, b_cols) in col_tiles.iter().enumerate() {
                for (j, &c) in b_cols.iter().enumerate() {
                    let dst = (ci * tile_w + j) * k;
                    ws.b_all[dst..dst + k].copy_from_slice(&ws.b_noised_t[c * k..c * k + k]);
                }
            }
            // 16-col packing groups on AVX-512, 8-col on AVX-VNNI → col-tile c's
            // first packed group is c*4 vs c*8 respectively.
            let grp_per_tile = if avx512 { 4 } else { 8 };
            if avx512 {
                crate::microkernel::pack_b_16(&ws.b_all[..total_cols * k], total_cols, k, &mut ws.b_packed);
            } else {
                crate::microkernel::pack_b_8(&ws.b_all[..total_cols * k], total_cols, k, &mut ws.b_packed);
            }

            let panel = std::env::var("ARIA_PANEL")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&v| v > 0)
                .unwrap_or(256);

            let mut p0 = 0;
            while p0 < nt {
                let p1 = (p0 + panel).min(nt);
                for (ci, b_cols) in col_tiles.iter().enumerate() {
                    let grp_base = ci * grp_per_tile;
                    let mut t = p0;
                    while t + 4 <= p1 {
                        let msgs = if avx512 {
                            unsafe {
                                crate::microkernel::jackpot_tile_8x64_avx512(
                                    &ws.a_tiled, &ws.b_packed, 2 * t, grp_base, k, rank, &mut ws.scratch,
                                )
                            }
                        } else {
                            unsafe {
                                crate::microkernel::jackpot_tile_8x64_avxvnni(
                                    &ws.a_tiled, &ws.b_packed, 2 * t, grp_base, k, rank, &mut ws.scratch,
                                )
                            }
                        };
                        for (tt, msg) in msgs.iter().enumerate() {
                            if le_leq(&compute_jackpot_hash(msg, a_noise_seed), bound_le) {
                                return Some(make_proof(&ws.a_sig[..mk], &ws.b_sig[..nk], m, n, k, rank, &job_key, &row_tiles[t + tt], b_cols));
                            }
                        }
                        t += 4;
                    }
                    // Remainder (<4 tiles): AVX-512 → 2×64 kernel; AVX-VNNI → per-cell.
                    while t < p1 {
                        let rt = &row_tiles[t];
                        let jp = if avx512 {
                            unsafe {
                                crate::microkernel::jackpot_tile_2x64_avx512(
                                    &ws.a_tiled, &ws.b_packed, 2 * t, 2 * t + 1, grp_base, k, rank, &mut ws.scratch,
                                )
                            }
                        } else {
                            tile_jackpot_flat(rt, b_cols, &ws.a_noised[..mk], &ws.b_noised_t[..nk], rank, k)
                        };
                        if le_leq(&compute_jackpot_hash(&jp, a_noise_seed), bound_le) {
                            return Some(make_proof(&ws.a_sig[..mk], &ws.b_sig[..nk], m, n, k, rank, &job_key, rt, b_cols));
                        }
                        t += 1;
                    }
                }
                p0 = p1;
            }
            return None;
        }
    }

    // ── FALLBACK: any other shape / non-AVX-512 CPU — per-cell kernel.
    for a_rows in threads_partition(&config.rows_pattern, m) {
        for b_cols in threads_partition(&config.cols_pattern, n) {
            let jp = tile_jackpot_flat(&a_rows, &b_cols, &ws.a_noised[..mk], &ws.b_noised_t[..nk], rank, k);
            if le_leq(&compute_jackpot_hash(&jp, a_noise_seed), bound_le) {
                return Some(make_proof(&ws.a_sig[..mk], &ws.b_sig[..nk], m, n, k, rank, &job_key, &a_rows, &b_cols));
            }
        }
    }
    None
}

/// Grind attempts until one tile beats `bound_le`. Returns the canonical proof.
/// Allocates one reusable [`Workspace`] for the whole grind (no per-attempt
/// allocation in the hot loop).
pub fn mine_share<R: Rng>(
    rng: &mut R,
    m: usize,
    n: usize,
    k: usize,
    header: &IncompleteBlockHeader,
    config: &MiningConfiguration,
    bound_le: &[u8; 32],
) -> PlainProof {
    let mut ws = Workspace::new();
    loop {
        if let Some(p) = try_mine_one_bounded(&mut ws, rng, m, n, k, header, config, bound_le) {
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

    /// THE chantier-1.B deliverable: a proof produced by OUR grind (not by
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

    /// h=2 × w=64 config — the real Pearl tile shape, which triggers the AVX-512
    /// register-blocked micro-kernel fast path in `try_mine_one_bounded`.
    fn config_2x64(k: u32) -> MiningConfiguration {
        let cols: Vec<u32> = (0..64).collect();
        MiningConfiguration {
            common_dim: k,
            rank: 128,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 1]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&cols).unwrap(),
            reserved: MiningConfiguration::RESERVED_VALUE,
        }
    }

    /// THE phase-1 deliverable: the fast micro-kernel path must produce a proof
    /// that passes the official node-side verifiers, exactly like the slow path.
    /// Runs on AVX-512 VNNI hosts (where the fast path engages); on others it
    /// exercises the fallback — either way the proof must verify.
    #[test]
    fn fast_2x64_proof_passes_official_verify() {
        let (m, n, k) = (256usize, 64usize, 4096usize);
        let header = easy_header(0x207f_ffff);
        let config = config_2x64(k as u32);

        let bound = zk_pow::api::sanity_checks::extract_difficulty_bound(header.nbits, &config);
        let mut bound_le = [0u8; 32];
        bound.to_little_endian(&mut bound_le);

        let mut rng = StdRng::seed_from_u64(0x2064_BEEF);
        let proof = mine_share(&mut rng, m, n, k, &header, &config, &bound_le);

        zk_pow::ffi::plain_proof::parse_plain_proof(header, &proof)
            .expect("fast-path proof must parse (roots == hash_a/hash_b)");
        zk_pow::api::verify::verify_plain_proof(&header, &proof)
            .expect("fast-path proof must pass verify_plain_proof");
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
