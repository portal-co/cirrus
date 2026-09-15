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
    /// The opened-row count does not match the queried-index count, or the
    /// committed table size.
    WrongBatchSize,
    /// An opened row has the wrong width.
    WrongWidth,
    /// The pruned multiproof carries the wrong number of sibling digests.
    SiblingCountMismatch,
    /// Duplicate queries on one leaf opened different rows.
    InconsistentDuplicateOpenings,
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
            Self::WrongBatchSize => {
                f.write_str("opened row count does not match the query count or table size")
            }
            Self::WrongWidth => f.write_str("opened row has the wrong width"),
            Self::SiblingCountMismatch => {
                f.write_str("pruned multiproof carries the wrong number of sibling digests")
            }
            Self::InconsistentDuplicateOpenings => {
                f.write_str("duplicate queries on one leaf opened different rows")
            }
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

    /// Open many rows under one pruned multiproof.
    ///
    /// The proof carries the minimal set of boundary sibling digests in
    /// upstream `PrunedMerklePaths` wire order: level 0 first, groups by
    /// ascending parent index, missing child positions ascending. Indices
    /// are sorted and deduplicated internally; returned rows follow the
    /// caller's index order.
    pub fn open_multi(
        &self,
        indices: &[usize],
        rows: &[Vec<KoalaBear>],
    ) -> Result<(Vec<Vec<KoalaBear>>, PrunedMerklePaths), MerkleError> {
        if rows.len() != self.leaf_count() {
            return Err(MerkleError::WrongBatchSize);
        }
        for &index in indices {
            if index >= self.leaf_count() {
                return Err(MerkleError::IndexOutOfBounds);
            }
        }
        let opened: Vec<Vec<KoalaBear>> = indices.iter().map(|&i| rows[i].clone()).collect();

        // Sorted-unique frontier of leaf positions.
        let mut frontier: Vec<usize> = indices.to_vec();
        frontier.sort_unstable();
        frontier.dedup();

        let mut sibling_hashes = Vec::new();
        let levels = self.layers.len().saturating_sub(1);
        for level in 0..levels {
            let layer = &self.layers[level];
            let mut parents = Vec::with_capacity(frontier.len());
            let mut i = 0;
            while i < frontier.len() {
                let parent = frontier[i] / 2;
                let mut covered = [false; 2];
                while i < frontier.len() && frontier[i] / 2 == parent {
                    covered[frontier[i] % 2] = true;
                    i += 1;
                }
                for (position, &is_covered) in covered.iter().enumerate() {
                    if !is_covered {
                        sibling_hashes.push(layer[parent * 2 + position]);
                    }
                }
                parents.push(parent);
            }
            frontier = parents;
        }
        Ok((opened, PrunedMerklePaths { sibling_hashes }))
    }

    /// Verify a pruned multi-opening against `root`.
    ///
    /// Mirrors upstream `verify_batch_pruned` for a single binary matrix:
    /// the queried indices come from the verifier, duplicate queries must
    /// agree on their opened rows, the proof must supply exactly the
    /// frontier's digest count, and every index must be in bounds.
    pub fn verify_multi(
        root: &PoseidonDigest,
        leaf_count: usize,
        width: usize,
        indices: &[usize],
        rows: &[Vec<KoalaBear>],
        proof: &PrunedMerklePaths,
    ) -> Result<(), MerkleError> {
        if leaf_count == 0 || !leaf_count.is_power_of_two() {
            return Err(MerkleError::NonPowerOfTwoRows);
        }
        if rows.len() != indices.len() {
            return Err(MerkleError::WrongBatchSize);
        }
        if indices.is_empty() {
            return if proof.sibling_hashes.is_empty() {
                Ok(())
            } else {
                Err(MerkleError::SiblingCountMismatch)
            };
        }
        for row in rows {
            if row.len() != width {
                return Err(MerkleError::WrongWidth);
            }
        }

        let mut sorted_unique: Vec<usize> = indices.to_vec();
        sorted_unique.sort_unstable();
        sorted_unique.dedup();
        if *sorted_unique.last().expect("nonempty") >= leaf_count {
            return Err(MerkleError::IndexOutOfBounds);
        }

        // Representative row per unique leaf; duplicates must agree.
        let mut reps: Vec<Option<usize>> = alloc::vec![None; sorted_unique.len()];
        for (original, &leaf) in indices.iter().enumerate() {
            let slot = sorted_unique
                .binary_search(&leaf)
                .expect("from the query set");
            match reps[slot] {
                None => reps[slot] = Some(original),
                Some(rep) => {
                    if rows[rep] != rows[original] {
                        return Err(MerkleError::InconsistentDuplicateOpenings);
                    }
                }
            }
        }

        let levels = leaf_count.trailing_zeros() as usize;
        let mut digests: Vec<PoseidonDigest> = reps
            .iter()
            .map(|rep| poseidon_hash_fixed(&rows[rep.expect("every slot has a rep")]))
            .collect();
        let mut frontier = sorted_unique;
        let mut cursor = 0_usize;
        for _level in 0..levels {
            let mut parents: Vec<usize> = Vec::with_capacity(frontier.len());
            let mut parent_digests: Vec<PoseidonDigest> = Vec::with_capacity(frontier.len());
            let mut i = 0;
            while i < frontier.len() {
                let parent = frontier[i] / 2;
                let mut children: [Option<PoseidonDigest>; 2] = [None, None];
                while i < frontier.len() && frontier[i] / 2 == parent {
                    children[frontier[i] % 2] = Some(digests[i]);
                    i += 1;
                }
                for position in 0..2 {
                    if children[position].is_none() {
                        let digest = proof
                            .sibling_hashes
                            .get(cursor)
                            .ok_or(MerkleError::SiblingCountMismatch)?;
                        children[position] = Some(*digest);
                        cursor += 1;
                    }
                }
                parents.push(parent);
                parent_digests.push(poseidon_compress2([
                    children[0].expect("filled"),
                    children[1].expect("filled"),
                ]));
            }
            frontier = parents;
            digests = parent_digests;
        }
        if cursor != proof.sibling_hashes.len() {
            return Err(MerkleError::SiblingCountMismatch);
        }
        if digests.len() == 1 && &digests[0] == root {
            Ok(())
        } else {
            Err(MerkleError::RootMismatch)
        }
    }
}

/// Compact multi-opening proof: the minimal set of boundary sibling
/// digests in upstream `PrunedMerklePaths` wire order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrunedMerklePaths {
    /// Boundary sibling digests: level 0 first, then ascending levels;
    /// within a level, groups by ascending parent index; within a group,
    /// missing child positions ascending.
    pub sibling_hashes: Vec<PoseidonDigest>,
}
