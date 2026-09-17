#![no_std]
#![warn(missing_docs)]

//! Allocation-free scheduling primitives shared by ERT loop adapters.
//!
//! A [`CandidateTable`] borrows storage from its caller. It retains a live
//! prefix, accumulates the next generation in the disjoint tail, rejects
//! overflow, deduplicates deterministically, and compacts only after every
//! current candidate has been examined. The substrate has no ISA state,
//! decoder, or circuit dependency: adapters own those semantics and use this
//! type only for virtual-IP candidate lifecycle management.

/// A caller-owned, fixed-capacity set of current and next virtual-IP
/// candidates.
///
/// The underlying slice must hold both the current generation and the next
/// one. Consequently, an adapter needing at most `N` live candidates must
/// supply at least `2 * N` slots. An undersized table returns [`TableError`]
/// rather than allocating or overwriting a live candidate.
pub struct CandidateTable<'a, T> {
    entries: &'a mut [T],
    len: usize,
}

/// A candidate-table operation could not preserve its fixed-capacity
/// invariants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TableError {
    /// The caller supplied no storage or the next generation exceeded it.
    Capacity,
    /// A next-generation operation was attempted outside an active generation.
    NoActiveGeneration,
}

impl<'a, T: Copy + Eq> CandidateTable<'a, T> {
    /// Initialize a table with one live candidate.
    pub fn new(entries: &'a mut [T], initial: T) -> Result<Self, TableError> {
        let Some(first) = entries.first_mut() else {
            return Err(TableError::Capacity);
        };
        *first = initial;
        Ok(Self { entries, len: 1 })
    }

    /// The current live candidates.
    pub fn current(&self) -> &[T] {
        &self.entries[..self.len]
    }

    /// Number of current live candidates.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no current candidates.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append one candidate to the next generation accumulating in the table
    /// tail, returning its new size. The current generation remains intact.
    pub fn append_next(&mut self, next_len: usize, candidate: T) -> Result<usize, TableError> {
        append_unique(&mut self.entries[self.len..], next_len, candidate)
    }

    /// Commit `next_len` tail candidates as the next live generation.
    pub fn finish_next(&mut self, next_len: usize) -> Result<usize, TableError> {
        if self.len + next_len > self.entries.len() {
            return Err(TableError::Capacity);
        }
        self.entries.copy_within(self.len..self.len + next_len, 0);
        self.len = next_len;
        Ok(next_len)
    }

    ///
    /// The returned builder borrows the table until [`NextCandidates::finish`]
    /// commits its deduplicated candidates into the live prefix.
    pub fn next(&mut self) -> Result<NextCandidates<'_, 'a, T>, TableError> {
        if self.len > self.entries.len() {
            return Err(TableError::NoActiveGeneration);
        }
        Ok(NextCandidates {
            table: self,
            next_len: 0,
        })
    }

    /// Remove every live candidate.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// path whose successor set is already fully known.
    pub fn replace(&mut self, candidates: &[T]) -> Result<(), TableError> {
        if candidates.len() > self.entries.len() {
            return Err(TableError::Capacity);
        }
        let mut len = 0;
        for candidate in candidates {
            if !self.entries[..len].contains(candidate) {
                self.entries[len] = *candidate;
                len += 1;
            }
        }
        self.len = len;
        Ok(())
    }
}

/// Append `candidate` to a caller-owned next-generation slice unless already
/// present, preserving first-seen order.
///
/// This is the low-level form used by adapters that must keep their current
/// generation readable while executing a body. Prefer [`CandidateTable`] when
/// the adapter can borrow the whole lifecycle at once.
pub fn append_unique<T: Copy + Eq>(
    next: &mut [T],
    len: usize,
    candidate: T,
) -> Result<usize, TableError> {
    if len > next.len() {
        return Err(TableError::NoActiveGeneration);
    }
    if next[..len].contains(&candidate) {
        return Ok(len);
    }
    let Some(slot) = next.get_mut(len) else {
        return Err(TableError::Capacity);
    };
    *slot = candidate;
    Ok(len + 1)
}

/// An in-progress next generation for a [`CandidateTable`].
pub struct NextCandidates<'table, 'entries, T> {
    table: &'table mut CandidateTable<'entries, T>,
    next_len: usize,
}

impl<T: Copy + Eq> NextCandidates<'_, '_, T> {
    /// Append `candidate` unless it is already in the next generation.
    pub fn push(&mut self, candidate: T) -> Result<(), TableError> {
        let base = self.table.len;
        if base + self.next_len >= self.table.entries.len() {
            return Err(TableError::Capacity);
        }
        if self.table.entries[base..base + self.next_len].contains(&candidate) {
            return Ok(());
        }
        self.table.entries[base + self.next_len] = candidate;
        self.next_len += 1;
        Ok(())
    }

    /// Append every candidate, preserving first-seen order after deduplication.
    pub fn extend(&mut self, candidates: &[T]) -> Result<(), TableError> {
        for candidate in candidates {
            self.push(*candidate)?;
        }
        Ok(())
    }

    /// Commit the next generation and return its size.
    pub fn finish(self) -> usize {
        let base = self.table.len;
        self.table
            .entries
            .copy_within(base..base + self.next_len, 0);
        self.table.len = self.next_len;
        self.next_len
    }
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::{CandidateTable, TableError};

    #[test]
    fn next_generation_deduplicates_and_compacts_in_first_seen_order() {
        let mut backing = [0; 8];
        let mut table = CandidateTable::new(&mut backing, 1).unwrap();
        let mut next = table.next().unwrap();
        next.extend(&[4, 2, 4, 3, 2]).unwrap();
        assert_eq!(next.finish(), 3);
        assert_eq!(table.current(), &[4, 2, 3]);
    }

    #[test]
    fn overflow_leaves_the_current_generation_intact() {
        let mut backing = [0; 2];
        let mut table = CandidateTable::new(&mut backing, 1).unwrap();
        let mut next = table.next().unwrap();
        assert_eq!(next.push(2), Ok(()));
        assert_eq!(next.push(3), Err(TableError::Capacity));
    }
}
