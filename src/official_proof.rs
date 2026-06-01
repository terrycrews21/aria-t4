//! Canonical Pearl `PlainProof` (de)serialization.
//!
//! Replaces the *guessed* layout in `proof.rs` (cells / chunks / ad-hoc Merkle
//! nodes) with the **canonical wire format** the Pearl network actually uses:
//!
//!   base64( bincode::serialize(&zk_pow::ffi::plain_proof::PlainProof) )
//!
//! This is byte-for-byte what `PlainProof::to_base64` produces in zk-pow's pyo3
//! bindings (`ffi/plain_proof.rs`): `bincode::serialize(self)` then STANDARD
//! base64. We reuse the *official* `PlainProof` struct (path dep `zk-pow`) so
//! there is a single source of truth for the format — no reverse-engineering.
//!
//! A proof encoded here round-trips through `parse_plain_proof`, the node-side
//! check that the matrix Merkle roots equal `hash_a` / `hash_b` (the
//! `ensure_eq!` in `parse_plain_proof`). That is the contract the pool relies on
//! before it can build the zk-certificate and `submitblock`.

use anyhow::{Context, Result};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use zk_pow::ffi::plain_proof::PlainProof;

/// Encode a `PlainProof` to the canonical base64 wire form (bincode + base64).
///
/// Mirror of zk-pow's `PlainProof::to_base64`.
pub fn encode_base64(proof: &PlainProof) -> Result<String> {
    let bytes = bincode::serialize(proof).context("bincode serialize PlainProof")?;
    Ok(B64.encode(bytes))
}

/// Encode a `PlainProof` to the canonical bincode bytes (no base64 wrapper).
pub fn encode_bytes(proof: &PlainProof) -> Result<Vec<u8>> {
    bincode::serialize(proof).context("bincode serialize PlainProof")
}

/// Decode a base64 wire `PlainProof`.
///
/// Mirror of zk-pow's `PlainProof::from_base64`.
pub fn decode_base64(s: &str) -> Result<PlainProof> {
    let bytes = B64.decode(s.trim()).context("base64 decode PlainProof")?;
    bincode::deserialize(&bytes).context("bincode deserialize PlainProof")
}

#[cfg(test)]
mod tests {
    use super::*;
    use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern};

    /// Mainnet-shaped Pearl config (mirrors zk-pow's own test vector in
    /// `api/proof.rs::PublicProofParams::new_for_tests`): rank 128, the canonical
    /// rows/cols patterns, k a multiple of 64.
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

    /// A freshly *mined* PlainProof, run through
    /// our encoder and back, must still be accepted by the node-side
    /// `parse_plain_proof` (Merkle roots == hash_a/hash_b). This proves our wire
    /// format is the canonical one and that nothing is lost in the round-trip.
    #[test]
    fn mined_proof_round_trips_and_parses() {
        // k = 4096 - 64 keeps m*k / n*k off the chunk boundary like the real
        // network vector, exercising the padded-chunk path. nbits 0x207fffff is
        // the easiest target (used by zk-pow's own `new_for_test`) so `mine`
        // returns on the first viable tile.
        let (m, n, k) = (256usize, 128usize, 4032usize);
        let header = easy_header(0x207f_ffff);
        let config = test_config(k as u32);

        let proof = zk_pow::ffi::mine::mine(m, n, k, header, config, None, false)
            .expect("mine should produce a PlainProof at the easiest difficulty");

        // Sanity: the mined proof itself parses (roots consistent) before we
        // touch serialization — isolates a format bug from a mining bug.
        zk_pow::ffi::plain_proof::parse_plain_proof(header, &proof)
            .expect("freshly mined proof must parse");

        // Round-trip through our canonical encoder.
        let b64 = encode_base64(&proof).expect("encode");
        let decoded = decode_base64(&b64).expect("decode");

        // Byte-identical re-encode ⇒ deserialization is lossless.
        assert_eq!(
            encode_bytes(&proof).unwrap(),
            encode_bytes(&decoded).unwrap(),
            "re-encoded bytes diverge after round-trip"
        );

        // And the decoded proof still satisfies the node-side contract.
        parse_plain_proof_ok(header, &decoded);
    }

    /// base64 of bincode must be exactly STANDARD base64 of the bincode bytes —
    /// guards against accidentally swapping the engine (URL_SAFE etc.).
    #[test]
    fn base64_is_standard_of_bincode() {
        let (m, n, k) = (256usize, 128usize, 4032usize);
        let header = easy_header(0x207f_ffff);
        let config = test_config(k as u32);
        let proof = zk_pow::ffi::mine::mine(m, n, k, header, config, None, false).unwrap();

        let bytes = bincode::serialize(&proof).unwrap();
        assert_eq!(encode_base64(&proof).unwrap(), B64.encode(&bytes));
    }

    fn parse_plain_proof_ok(header: IncompleteBlockHeader, proof: &PlainProof) {
        zk_pow::ffi::plain_proof::parse_plain_proof(header, proof)
            .expect("decoded proof must parse (roots == hash_a/hash_b)");
    }
}
