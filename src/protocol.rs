//! Pearl Stratum v1 protocol — a Pearl-specific extension of the classic
//! Stratum v1 wire format.
//!
//! Message flow:
//! 1. C→S `mining.subscribe`        — empty params, pool returns extranonce data
//! 2. C→S `mining.subscribe.pearl`  — `[["pearl/v1"], {}]` enables the v1 wire
//! 3. S→C `result {pearl/v1:true, pearl/v1.share_format:"base64"}`
//! 4. C→S `mining.authorize`        — `[wallet.worker, password]` (password = "x;d=N" for static diff)
//! 5. S→C `mining.set_difficulty`   — `[N]`
//! 6. S→C `pearl.set_mining_params` — `[{m, n, k, rank, rows_pattern, cols_pattern, mma_type}]`
//! 7. S→C `mining.notify`           — `[job_id, prev_hash_hex, header_template_hex, ntime, target_nbits_hex, clean_jobs]`
//! 8. C→S `mining.submit`           — `[wallet.worker, job_id, base64_proof]`
//!
//! All messages are line-delimited JSON-RPC 2.0.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcRequest {
    pub id: Option<u64>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcResponse {
    pub id: Option<u64>,
    pub result: Value,
    #[serde(default)]
    pub error: Option<RpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
}

/// `pearl.set_mining_params` payload — defines exactly what GEMM transcript the
/// miner must produce. The pool re-validates every submitted proof against
/// these params, so the miner cannot reduce work without rejection.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MiningParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub rank: u32,
    /// Indices of rows we must compute. Length is small (~2 in PPLNS at our diff).
    pub rows_pattern: Vec<u32>,
    /// Indices of cols we must report. Length up to 64 typically.
    pub cols_pattern: Vec<u32>,
    /// String like "Int7xInt7ToInt32" — matrix multiply-accumulate dtype.
    pub mma_type: String,
}

/// `mining.notify` payload. The Pearl wire uses 7 params (one more than the
/// classic Bitcoin/Stratum notify):
/// ```text
/// [job_id, prev_block_id, header_template, ntime, ntime_or_extranonce_aux,
///  target_nbits, clean_jobs]
/// ```
/// `params[4]` looks like a second timestamp / extra-nonce control (e.g.
/// "6a0c1af6") — kept opaque until validated; we don't depend on it for
/// proof construction.
#[derive(Debug, Clone)]
pub struct Job {
    pub job_id: String,
    /// 32-byte previous block id hex.
    pub prev_block_id: String,
    /// Header template hex, used as BLAKE3 seed prefix.
    pub header_template: String,
    /// Pool's nominal timestamp for this job (unix seconds).
    pub ntime: u32,
    /// Auxiliary 4-byte field — purpose to confirm.
    pub aux: String,
    /// Compact nbits target — used for the BLAKE3 grind difficulty (array wire).
    pub target_nbits: String,
    pub clean_jobs: bool,
    /// Full 32-byte big-endian target, when the pool sends it directly (AriaPool
    /// object wire). Takes precedence over `target_nbits` when present.
    pub full_target: Option<[u8; 32]>,
}

impl Job {
    pub fn from_params(params: &Value) -> anyhow::Result<Self> {
        // Some pools send an object {header, height, job_id, target}; others send
        // the classic 7-element array. Handle both.
        if let Some(obj) = params.as_object() {
            return Self::from_object(obj);
        }
        let arr = params.as_array().ok_or_else(|| anyhow::anyhow!("notify: params not array"))?;
        let s = |i: usize| -> anyhow::Result<String> {
            arr.get(i)
                .and_then(Value::as_str)
                .map(String::from)
                .ok_or_else(|| anyhow::anyhow!("notify: missing string param {i}"))
        };
        let u = |i: usize| -> anyhow::Result<u64> {
            arr.get(i)
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow::anyhow!("notify: missing u64 param {i}"))
        };
        let target_nbits = s(5)?;
        let clean_target = target_nbits.trim_start_matches("0x");
        let full_target = if clean_target.len() == 64 {
            hex::decode(clean_target).ok().and_then(|b| <[u8; 32]>::try_from(b.as_slice()).ok())
        } else if clean_target.len() == 8 {
            decode_nbits(clean_target).ok()
        } else {
            None
        };
        Ok(Job {
            job_id: s(0)?,
            prev_block_id: s(1)?,
            header_template: s(2)?,
            ntime: u(3)? as u32,
            aux: s(4)?,
            target_nbits,
            clean_jobs: arr.get(6).and_then(Value::as_bool).unwrap_or(true),
            full_target,
        })
    }

    /// Parse the AriaPool object wire: `{header, height, job_id, target}` where
    /// `target` is the full 32-byte big-endian target hex.
    fn from_object(obj: &serde_json::Map<String, Value>) -> anyhow::Result<Self> {
        let s = |k: &str| obj.get(k).and_then(Value::as_str).map(String::from);
        let job_id = s("job_id").ok_or_else(|| anyhow::anyhow!("notify obj: missing job_id"))?;
        let header_template = s("header").ok_or_else(|| anyhow::anyhow!("notify obj: missing header"))?;
        let target_hex = s("target").ok_or_else(|| anyhow::anyhow!("notify obj: missing target"))?;
        let tb = hex::decode(target_hex.trim_start_matches("0x"))?;
        if tb.len() != 32 {
            anyhow::bail!("notify obj: target must be 32 bytes, got {}", tb.len());
        }
        let mut full_target = [0u8; 32];
        full_target.copy_from_slice(&tb);
        Ok(Job {
            job_id,
            prev_block_id: String::new(),
            header_template,
            ntime: 0,
            aux: String::new(),
            target_nbits: String::new(),
            clean_jobs: true,
            full_target: Some(full_target),
        })
    }
}

/// 256-bit big-endian target decoded from the `nbits` compact form. nbits is the
/// classic Bitcoin-style 4-byte representation: high byte = exponent, low 3
/// bytes = mantissa. `target = mantissa << (8 * (exponent - 3))`.
pub fn decode_nbits(nbits_hex: &str) -> anyhow::Result<[u8; 32]> {
    let bytes = hex::decode(nbits_hex.trim_start_matches("0x"))?;
    if bytes.len() != 4 {
        anyhow::bail!("nbits must be 4 bytes, got {}", bytes.len());
    }
    let exponent = bytes[0] as usize;
    let mantissa: u32 = ((bytes[1] as u32) << 16) | ((bytes[2] as u32) << 8) | (bytes[3] as u32);
    let mut target = [0u8; 32];
    if exponent <= 3 {
        let shift = 8 * (3 - exponent);
        let m = mantissa >> shift;
        target[29] = ((m >> 16) & 0xff) as u8;
        target[30] = ((m >> 8) & 0xff) as u8;
        target[31] = (m & 0xff) as u8;
    } else {
        let offset = 32 - exponent;
        if offset + 3 <= 32 {
            target[offset] = ((mantissa >> 16) & 0xff) as u8;
            target[offset + 1] = ((mantissa >> 8) & 0xff) as u8;
            target[offset + 2] = (mantissa & 0xff) as u8;
        }
    }
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_nbits_observed_value() {
        // From wire capture: target_nbits = "1a1fffe0".
        let t = decode_nbits("1a1fffe0").unwrap();
        // First byte must be 0 (target is small enough).
        assert_eq!(t[0], 0x00);
        // The mantissa 0x1fffe0 lands somewhere in the high bytes.
        let nonzero: Vec<usize> = t.iter().enumerate().filter(|(_, b)| **b != 0).map(|(i, _)| i).collect();
        assert!(!nonzero.is_empty(), "target must have at least one non-zero byte");
    }

    #[test]
    fn decode_nbits_exponent_3() {
        // Edge case: exponent = 3 → mantissa fills last 3 bytes.
        let t = decode_nbits("03abcdef").unwrap();
        assert_eq!(t[29], 0xab);
        assert_eq!(t[30], 0xcd);
        assert_eq!(t[31], 0xef);
    }
}
