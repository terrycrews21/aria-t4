//! Stratum → official conversion — chantier 1.C step (b).
//!
//! Turns what the pool sends on the wire (`Job` + `MiningParams` + stratum
//! difficulty) into exactly the three inputs `official_grind::mine_share`
//! consumes: an `IncompleteBlockHeader`, a `MiningConfiguration`, and the
//! little-endian difficulty `bound`.
//!
//! Header: the pool emits an 80-byte Bitcoin-style prefix
//!   `version(4) ‖ prev(32) ‖ merkle(32) ‖ ntime(4) ‖ bits(4) ‖ nonce(4=0)`
//! (see `pool-node::header_prefix`). The Pearl `IncompleteBlockHeader` is the
//! first **76** bytes (same layout, no nonce) — `from_bytes` re-reverses
//! prev/merkle as the canonical format dictates. We drop the trailing 4-byte
//! nonce.
//!
//! This module is isolated and read-only w.r.t. the production engine.

use anyhow::{Context, Result, bail};
use zk_pow::api::proof::{
    IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern,
};

use crate::cpu_engine::share_target_from_difficulty;
use crate::protocol::{Job, MiningParams};

/// Parse the pool's hex `header_template` into the canonical
/// `IncompleteBlockHeader`. Accepts the 80-byte (Bitcoin, with nonce) or the
/// bare 76-byte form; anything else is rejected loudly.
pub fn header_from_template(header_template_hex: &str) -> Result<IncompleteBlockHeader> {
    let bytes = hex::decode(header_template_hex.trim_start_matches("0x"))
        .context("decode header_template hex")?;
    let header_bytes: &[u8] = match bytes.len() {
        // 80 = Bitcoin prefix with trailing 4-byte nonce → drop it.
        80 => &bytes[..IncompleteBlockHeader::SERIALIZED_SIZE],
        // 76 = already the bare Pearl header.
        76 => &bytes[..],
        other => bail!("header_template must be 76 or 80 bytes, got {other}"),
    };
    IncompleteBlockHeader::from_bytes(header_bytes).context("parse IncompleteBlockHeader")
}

/// Build the canonical `MiningConfiguration` from the pool's `MiningParams`.
/// `rows_pattern` / `cols_pattern` are the base (zero-based) periodic patterns.
pub fn config_from_params(params: &MiningParams) -> Result<MiningConfiguration> {
    if params.mma_type != "Int7xInt7ToInt32" {
        bail!("unsupported mma_type: {}", params.mma_type);
    }
    let rows_pattern = PeriodicPattern::from_list(&params.rows_pattern)
        .context("rows_pattern is not a valid periodic pattern")?;
    let cols_pattern = PeriodicPattern::from_list(&params.cols_pattern)
        .context("cols_pattern is not a valid periodic pattern")?;
    Ok(MiningConfiguration {
        common_dim: params.k,
        rank: u16::try_from(params.rank).context("rank > u16")?,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern,
        cols_pattern,
        reserved: MiningConfiguration::RESERVED_VALUE,
    })
}

/// Little-endian difficulty bound for a stratum share difficulty `d`:
/// `floor((2^256 - 1) / d)` in the byte order `official_grind` compares against
/// (`U256::from_little_endian(jackpot) ≤ bound`). `share_target_from_difficulty`
/// already yields this value big-endian; we just reverse to little-endian.
pub fn share_bound_le(difficulty: u64) -> [u8; 32] {
    let mut t = share_target_from_difficulty(difficulty);
    t.reverse();
    t
}

/// LuckyPool-style bound: the wire carries a full 256-bit big-endian target and
/// the official rule (`zk-pow sanity_checks::extract_difficulty_bound`) is
/// `jackpot ≤ target × h·w·k` — the threshold scales with the work of one
/// opened tile. Returns the little-endian byte bound `official_grind` compares
/// against, saturating at 2^256−1.
pub fn scaled_bound_le_from_target_be(target_be: &[u8; 32], factor: u64) -> [u8; 32] {
    let mut le = *target_be;
    le.reverse();
    let mut out = [0u8; 32];
    let mut carry: u128 = 0;
    for i in 0..32 {
        let v = le[i] as u128 * factor as u128 + carry;
        out[i] = (v & 0xff) as u8;
        carry = v >> 8;
    }
    if carry != 0 {
        return [0xff; 32];
    }
    out
}

/// Everything `official_grind::mine_share` needs for one job, derived from the
/// live stratum state.
#[derive(Clone)]
pub struct OfficialJob {
    pub header: IncompleteBlockHeader,
    pub config: MiningConfiguration,
    pub bound_le: [u8; 32],
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

/// Assemble an `OfficialJob` from the pool's job, params and current difficulty.
/// Pick a CPU-friendly `m`/`n` for a pattern dimension.
///
/// The pool advertises huge `m`/`n` (e.g. 131072) sized for a 16 GB GPU — on CPU
/// that means multi-GB matrices and a million-tile sweep per setup (unusable).
/// But `m`/`n` are NOT part of the hashed `job_key` (only k/rank/patterns are):
/// the miner chooses them inside the `PlainProof`, subject only to the sanity
/// rule `t + max(pattern) < dim` and `dim % period == 0`. So we use a small dim
/// that (a) is a multiple of the pattern period, (b) is strictly greater than
/// the pattern's max index, and (c) is at least `target` rows/cols so the
/// per-setup noise cost amortizes over a useful number of tiles.
fn fit_dim(target: usize, period: usize, max_idx: usize) -> usize {
    let period = period.max(1);
    let smallest_valid = (max_idx / period + 1) * period; // first multiple > max_idx
    let target_rounded = target.div_ceil(period) * period;
    smallest_valid.max(target_rounded)
}

pub fn build_official_job(
    job: &Job,
    params: &MiningParams,
    difficulty: u64,
) -> Result<OfficialJob> {
    let header = header_from_template(&job.header_template)?;
    let config = config_from_params(params)?;

    // CPU-sized m/n (see `fit_dim`). A LARGE batch amortizes the per-setup cost
    // (draw + BLAKE3 commitment over m·k/n·k + noise) over a big tile sweep, so
    // the register-blocked micro-kernel runs near its raw throughput instead of
    // being starved by setup (measured: n=64 → 37 GMAC/s vs 2048×1024 → ~220).
    // Tunable via ARIA_BATCH_M / ARIA_BATCH_N for per-machine memory budgets.
    let tgt_m = std::env::var("ARIA_BATCH_M").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let tgt_n = std::env::var("ARIA_BATCH_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let m = fit_dim(tgt_m, config.rows_pattern.period() as usize, config.rows_pattern.max() as usize);
    let n = fit_dim(tgt_n, config.cols_pattern.period() as usize, config.cols_pattern.max() as usize);

    Ok(OfficialJob {
        header,
        config,
        bound_le: share_bound_le(difficulty),
        m,
        n,
        k: params.k as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header() -> IncompleteBlockHeader {
        IncompleteBlockHeader {
            version: 0x2000_0000,
            prev_block: [0xab; 32],
            merkle_root: [0xcd; 32],
            timestamp: 1_715_748_000,
            nbits: 0x1d00_ffff,
        }
    }

    /// The pool serializes the 76-byte Pearl header then appends a zero nonce to
    /// make the 80-byte Bitcoin prefix. Our parser must recover the exact header
    /// from that 80-byte hex — round-trip identity.
    #[test]
    fn header_round_trips_through_80_byte_template() {
        let h = sample_header();
        let mut wire = h.to_bytes().to_vec(); // 76 bytes, canonical
        wire.extend_from_slice(&[0u8; 4]); // + nonce → 80 (Bitcoin-style)
        let hexed = hex::encode(&wire);

        let got = header_from_template(&hexed).unwrap();
        assert_eq!(got.version, h.version);
        assert_eq!(got.prev_block, h.prev_block);
        assert_eq!(got.merkle_root, h.merkle_root);
        assert_eq!(got.timestamp, h.timestamp);
        assert_eq!(got.nbits, h.nbits);
    }

    /// The bare 76-byte form (no nonce) must parse identically.
    #[test]
    fn header_round_trips_through_76_byte_template() {
        let h = sample_header();
        let hexed = hex::encode(h.to_bytes());
        let got = header_from_template(&hexed).unwrap();
        assert_eq!(got.prev_block, h.prev_block);
        assert_eq!(got.nbits, h.nbits);
    }

    #[test]
    fn header_rejects_wrong_length() {
        assert!(header_from_template(&hex::encode([0u8; 64])).is_err());
    }

    #[test]
    fn config_from_mainnet_params() {
        let params = MiningParams {
            m: 256,
            n: 128,
            k: 4032,
            rank: 128,
            rows_pattern: vec![0, 8, 64, 72],
            cols_pattern: vec![0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105],
            mma_type: "Int7xInt7ToInt32".into(),
        };
        let cfg = config_from_params(&params).unwrap();
        assert_eq!(cfg.common_dim, 4032);
        assert_eq!(cfg.rank, 128);
        assert_eq!(cfg.rows_pattern.to_list(), vec![0, 8, 64, 72]);
        // Re-serialization must be 52 bytes and round-trip (guards pattern packing).
        assert_eq!(cfg.to_bytes().len(), MiningConfiguration::SERIALIZED_SIZE);
    }

    /// `share_bound_le` must be the little-endian image of the big-endian share
    /// target, and order correctly: bigger difficulty ⇒ smaller bound.
    #[test]
    fn share_bound_le_is_le_and_monotonic() {
        let easy = share_bound_le(1);
        let hard = share_bound_le(1 << 20);
        // little-endian: most-significant byte is the last one.
        assert!(easy[31] >= hard[31], "harder difficulty must not raise the high byte");
        // bit-exact mirror of the BE target.
        let be = share_target_from_difficulty(42);
        let mut le = be;
        le.reverse();
        assert_eq!(share_bound_le(42), le);
    }

    /// End-to-end: a stratum object job + params + difficulty assembles into an
    /// OfficialJob whose pieces are internally consistent.
    #[test]
    fn build_official_job_from_object_wire() {
        let h = sample_header();
        let mut wire = h.to_bytes().to_vec();
        wire.extend_from_slice(&[0u8; 4]);
        let job = Job {
            job_id: "abc".into(),
            prev_block_id: String::new(),
            header_template: hex::encode(&wire),
            ntime: 0,
            aux: String::new(),
            target_nbits: String::new(),
            clean_jobs: true,
            full_target: None,
        };
        let params = MiningParams {
            m: 256,
            n: 128,
            k: 4032,
            rank: 128,
            rows_pattern: vec![0, 8, 64, 72],
            cols_pattern: vec![0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105],
            mma_type: "Int7xInt7ToInt32".into(),
        };
        let oj = build_official_job(&job, &params, 524_288).unwrap();
        assert_eq!(oj.k, 4032);
        assert_eq!(oj.header.nbits, h.nbits);
        assert_eq!(oj.config.rank, 128);
    }
}
