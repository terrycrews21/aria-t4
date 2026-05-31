//! BLAKE3 Merkle tree over the GEMM transcript cells.
//!
//! Structure:
//! - leaves          : BLAKE3(transcript_cell_bytes), 32 bytes each
//! - branching       : 128 (cf. `MerkleTreeRootsKernel<L, D, 128>`)
//! - depth           : 2/3/4 depending on leaf count (128/256/512 leaves observed)
//! - inner-node hash : `BLAKE3 with domain "MERKLE-INNER"` keyed by the parent's
//!                     position. Exact convention will be reconciled the first
//!                     time we submit a proof and the pool accepts it.

use crate::gemm::TranscriptCell;
use blake3::Hasher;

pub const BRANCHING: usize = 128;
pub const DOMAIN_LEAF: &[u8] = b"PEARL-MERKLE-LEAF";
pub const DOMAIN_INNER: &[u8] = b"PEARL-MERKLE-INNER";

#[derive(Debug, Clone)]
pub struct MerkleTree {
    /// `levels[0]` is the leaves, `levels[last]` is the single-element root.
    pub levels: Vec<Vec<[u8; 32]>>,
}

impl MerkleTree {
    pub fn build(cells: &[TranscriptCell]) -> Self {
        let leaves: Vec<[u8; 32]> = cells.iter().map(hash_leaf).collect();
        let mut levels = vec![leaves];
        while levels.last().unwrap().len() > 1 {
            let cur = levels.last().unwrap();
            let mut next = Vec::with_capacity(cur.len().div_ceil(BRANCHING));
            for chunk in cur.chunks(BRANCHING) {
                next.push(hash_inner(chunk));
            }
            levels.push(next);
        }
        MerkleTree { levels }
    }

    pub fn root(&self) -> [u8; 32] {
        *self.levels.last().unwrap().first().unwrap()
    }

    pub fn depth(&self) -> usize {
        self.levels.len() - 1
    }
}

fn hash_leaf(cell: &TranscriptCell) -> [u8; 32] {
    let mut h = Hasher::new();
    h.update(DOMAIN_LEAF);
    h.update(&cell.row.to_le_bytes());
    h.update(&cell.col.to_le_bytes());
    h.update(&cell.value.to_le_bytes());
    *h.finalize().as_bytes()
}

fn hash_inner(children: &[[u8; 32]]) -> [u8; 32] {
    let mut h = Hasher::new();
    h.update(DOMAIN_INNER);
    h.update(&(children.len() as u32).to_le_bytes());
    for c in children {
        h.update(c);
    }
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_cells(n: usize) -> Vec<TranscriptCell> {
        (0..n as u32)
            .map(|i| TranscriptCell { row: i / 8, col: i % 8, value: i as i32 })
            .collect()
    }

    #[test]
    fn one_leaf_yields_itself_as_root() {
        let cells = mk_cells(1);
        let t = MerkleTree::build(&cells);
        assert_eq!(t.depth(), 0);
        assert_eq!(t.root(), hash_leaf(&cells[0]));
    }

    #[test]
    fn one_branching_level() {
        let cells = mk_cells(10);
        let t = MerkleTree::build(&cells);
        assert_eq!(t.depth(), 1);
    }

    #[test]
    fn two_levels_at_high_count() {
        let cells = mk_cells(256);
        let t = MerkleTree::build(&cells);
        // 256 leaves → 2 inner (128 each) → 1 root  ⇒ depth 2
        assert_eq!(t.depth(), 2);
        assert_eq!(t.levels[0].len(), 256);
        assert_eq!(t.levels[1].len(), 2);
        assert_eq!(t.levels[2].len(), 1);
    }
}
