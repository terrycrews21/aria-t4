//! PlainProof serializer — packs the GEMM transcript cells, the Merkle path,
//! and the winning nonce into the base64 blob the pool expects on
//! `mining.submit`.
//!
//! Wire format (~2-5 KB base64 payload):
//!
//! ```text
//! struct PlainProof {
//!   u32        magic        = 0x01000000  (LE)
//!   u32        version      = 0x00000001
//!   u32        num_cells    : count of transcript cells
//!   [Cell]     cells        : (row u16, col u16, value i32) * num_cells
//!   u32        num_chunks   : GEMM chunks proven
//!   [Chunk]    chunks       : raw int32 dot-product values
//!   u32        num_nodes    : Merkle nodes along the path
//!   [Node]     nodes        : 32-byte hashes
//!   [32]u8     merkle_root
//!   u64        nonce        (LE)
//!   [32]u8     pow_hash
//! }
//! ```
//!
//! The exact layout will be reconciled against a successful pool submission;
//! the magic + version + nonce + pow_hash tail is enough for the pool to
//! identify our claim. Reserved bytes are conservatively zero-padded.

use crate::gemm::TranscriptCell;
use crate::merkle::MerkleTree;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use std::io::Write;

const MAGIC: u32 = 0x0100_0000;
const VERSION: u32 = 1;

pub struct ProofInputs<'a> {
    pub cells: &'a [TranscriptCell],
    pub tree: &'a MerkleTree,
    pub nonce: u64,
    pub pow_hash: [u8; 32],
}

pub fn encode_base64(inputs: &ProofInputs<'_>) -> String {
    let raw = encode_raw(inputs);
    B64.encode(&raw)
}

pub fn encode_raw(inputs: &ProofInputs<'_>) -> Vec<u8> {
    let mut buf = Vec::with_capacity(512);
    buf.write_all(&MAGIC.to_le_bytes()).unwrap();
    buf.write_all(&VERSION.to_le_bytes()).unwrap();

    // Cells
    buf.write_all(&(inputs.cells.len() as u32).to_le_bytes()).unwrap();
    for c in inputs.cells {
        buf.write_all(&(c.row as u16).to_le_bytes()).unwrap();
        buf.write_all(&(c.col as u16).to_le_bytes()).unwrap();
        buf.write_all(&c.value.to_le_bytes()).unwrap();
    }

    // GEMM chunks (one int32 per cell, raw value). The pool re-validates each
    // dot product against its own copy of the matrices.
    buf.write_all(&(inputs.cells.len() as u32).to_le_bytes()).unwrap();
    for c in inputs.cells {
        buf.write_all(&c.value.to_le_bytes()).unwrap();
    }

    // Merkle nodes: serialize every level except the leaves (which are
    // re-derived from cells) and the root (sent separately).
    let inner: Vec<&[u8; 32]> = inputs
        .tree
        .levels
        .iter()
        .skip(1) // skip leaves
        .take(inputs.tree.levels.len().saturating_sub(2)) // skip root
        .flatten()
        .collect();
    buf.write_all(&(inner.len() as u32).to_le_bytes()).unwrap();
    for n in &inner {
        buf.write_all(*n).unwrap();
    }

    // Root + winning nonce + pow_hash.
    buf.write_all(&inputs.tree.root()).unwrap();
    buf.write_all(&inputs.nonce.to_le_bytes()).unwrap();
    buf.write_all(&inputs.pow_hash).unwrap();

    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::TranscriptCell;

    fn mk_tree() -> (Vec<TranscriptCell>, MerkleTree) {
        let cells: Vec<TranscriptCell> = (0..16u32)
            .map(|i| TranscriptCell { row: i / 4, col: i % 4, value: i as i32 })
            .collect();
        let tree = MerkleTree::build(&cells);
        (cells, tree)
    }

    #[test]
    fn encode_round_trip_length_consistent() {
        let (cells, tree) = mk_tree();
        let inputs = ProofInputs {
            cells: &cells,
            tree: &tree,
            nonce: 42,
            pow_hash: [0xab; 32],
        };
        let raw = encode_raw(&inputs);
        // 4 magic + 4 version + 4 num_cells + cells (8 bytes each) + 4 num_chunks
        // + chunks (4 bytes each) + 4 num_nodes + 32*num_nodes + 32 root + 8 nonce + 32 hash
        let expected = 4 + 4 + 4 + cells.len() * 8 + 4 + cells.len() * 4
            + 4 + 0 /* small tree, no inner */ + 32 + 8 + 32;
        assert_eq!(raw.len(), expected);
    }

    #[test]
    fn base64_is_decodable() {
        let (cells, tree) = mk_tree();
        let inputs = ProofInputs {
            cells: &cells,
            tree: &tree,
            nonce: 1,
            pow_hash: [0; 32],
        };
        let b64 = encode_base64(&inputs);
        let decoded = B64.decode(b64).unwrap();
        assert_eq!(decoded, encode_raw(&inputs));
    }
}
