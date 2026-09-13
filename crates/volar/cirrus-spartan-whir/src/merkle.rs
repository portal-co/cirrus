//! Poseidon Merkle commitments for fixed-shape row tables.
//!
//! The initial profile intentionally supports one nonempty, power-of-two-high
//! table. Leaf hashing uses the Poseidon2-24 fixed-length sponge and internal
//! nodes use the Poseidon2-16 truncated compressor, matching upstream
//! single-matrix Poseidon MMCS commitments. Multi-matrix injection, caps, and
//! pruned batch proofs are later PCS work.

use alloc::vec::Vec;
use core::fmt;

use crate::{KoalaBear, PoseidonDigest, poseidon_compress2, poseidon_hash_fixed};

/// A binary Poseidon Merkle tree over hashed rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoseidonMerkleTree {
    layers: Vec<Vec<PoseidonDigest>>,
}

/// One Merkle authentication path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoseidonMerklePath {
    /// Hash of the opened row.
    pub leaf: PoseidonDigest,
    /// Sibling digests from leaf layer upward.
    pub siblings: Vec<PoseidonDigest>,
}

/// Why Merkle construction or verification failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MerkleError {
    /// A tree must contain at least one row.
    Empty,
    /// The initial profile requires a power-of-two row count.
    NonPowerOfTwoRows,
    /// The opening index is outside the tree.
    IndexOutOfBounds,
    /// The path length does not match the tree height.
    WrongPathLength,
    /// The recomputed root differs from the commitment.
    RootMismatch,
}

impl fmt::Display for MerkleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("cannot commit an empty row table"),
            Self::NonPowerOfTwoRows => {
                f.write_str("initial Poseidon Merkle profile requires a power-of-two row count")
            }
            Self::IndexOutOfBounds => f.write_str("Merkle opening index is out of bounds"),
            Self::WrongPathLength => f.write_str("Merkle path length does not match tree height"),
            Self::RootMismatch => f.write_str("Merkle root mismatch"),
        }
    }
}

impl core::error::Error for MerkleError {}

impl PoseidonMerkleTree {
    /// Commit one nonempty power-of-two-high table of field-element rows.
    pub fn commit_rows(rows: &[Vec<KoalaBear>]) -> Result<Self, MerkleError> {
        if rows.is_empty() {
            return Err(MerkleError::Empty);
        }
        if !rows.len().is_power_of_two() {
            return Err(MerkleError::NonPowerOfTwoRows);
        }
        let mut layers: Vec<Vec<PoseidonDigest>> = Vec::new();
        layers.push(rows.iter().map(|row| poseidon_hash_fixed(row)).collect());
        while layers.last().expect("leaf layer").len() > 1 {
            let previous = layers.last().expect("nonempty layers");
            let next = previous
                .chunks_exact(2)
                .map(|pair| poseidon_compress2([pair[0], pair[1]]))
                .collect::<Vec<_>>();
            layers.push(next);
        }
        Ok(Self { layers })
    }

    /// Commitment root.
    pub fn root(&self) -> PoseidonDigest {
        self.layers
            .last()
            .and_then(|layer| layer.first())
            .copied()
            .expect("validated tree has a root")
    }

    /// Number of table rows.
    pub fn leaf_count(&self) -> usize {
        self.layers.first().map_or(0, Vec::len)
    }

    /// Open one row index.
    pub fn open(&self, index: usize) -> Result<PoseidonMerklePath, MerkleError> {
        if index >= self.leaf_count() {
            return Err(MerkleError::IndexOutOfBounds);
        }
        let mut siblings = Vec::with_capacity(self.layers.len().saturating_sub(1));
        let mut cursor = index;
        for layer in self.layers.iter().take(self.layers.len().saturating_sub(1)) {
            siblings.push(layer[cursor ^ 1]);
            cursor >>= 1;
        }
        Ok(PoseidonMerklePath {
            leaf: self.layers[0][index],
            siblings,
        })
    }

    /// Verify a row opening against `root`.
    pub fn verify(
        root: &PoseidonDigest,
        index: usize,
        row: &[KoalaBear],
        path: &PoseidonMerklePath,
    ) -> Result<(), MerkleError> {
        if path.leaf != poseidon_hash_fixed(row) {
            return Err(MerkleError::RootMismatch);
        }
        let mut digest = path.leaf;
        let mut cursor = index;
        for sibling in &path.siblings {
            digest = if cursor & 1 == 0 {
                poseidon_compress2([digest, *sibling])
            } else {
                poseidon_compress2([*sibling, digest])
            };
            cursor >>= 1;
        }
        if cursor != 0 {
            return Err(MerkleError::IndexOutOfBounds);
        }
        if &digest == root {
            Ok(())
        } else {
            Err(MerkleError::RootMismatch)
        }
    }

    /// Verify a path with an explicit expected tree height.
    pub fn verify_with_height(
        root: &PoseidonDigest,
        index: usize,
        row: &[KoalaBear],
        path: &PoseidonMerklePath,
        leaf_count: usize,
    ) -> Result<(), MerkleError> {
        if leaf_count == 0 || !leaf_count.is_power_of_two() {
            return Err(MerkleError::NonPowerOfTwoRows);
        }
        if index >= leaf_count {
            return Err(MerkleError::IndexOutOfBounds);
        }
        if path.siblings.len() != leaf_count.trailing_zeros() as usize {
            return Err(MerkleError::WrongPathLength);
        }
        Self::verify(root, index, row, path)
    }
}
