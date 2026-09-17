#![no_std]
#![warn(missing_docs)]

//! Allocation-free scheduling primitives shared by ERT loop adapters.
//!
//! A [`CandidateTable`] borrows storage from its caller. It retains a live
//! prefix, accumulates the next generation in the disjoint tail, rejects
//! overflow, deduplicates deterministically, and compacts only after every
//! current candidate has been examined. The substrate has no ISA state,
use cirrus_core::{ContextWithBitAnd, ContextWithBitXor, HasError};

/// Select a candidate's `value` only while `active`, retaining `old`
/// otherwise.
///
/// The XOR/AND form is deliberately used instead of a generic mux so loop
/// adapters can predicate virtual-stack writes with the same circuit shape:
/// `old ^ (active & (value ^ old))`.
pub fn predicated_value<C, W>(context: &mut C, active: W, value: W, old: W) -> Result<W, C::Error>
where
    C: ContextWithBitAnd<bool, Wrapped = W> + ContextWithBitXor<bool, Wrapped = W> + HasError,
    W: Clone,
{
    let difference = context.bitxor(value, old.clone())?;
    let gated = context.bitand(active, difference)?;
    context.bitxor(old, gated)
}

/// An adapter callback invoked once for every candidate in a generation.
///
/// The adapter owns ISA state, body execution, and state folding. It calls the
/// supplied successor sink for every candidate that can be live in the next
/// generation. The core owns only deterministic successor scheduling and
/// caller-buffer capacity enforcement.
pub trait CandidateDriver<T: Copy + Eq> {
    /// Error reported by the ISA adapter.
    type Error;

    /// Execute the body rooted at `candidate` and report each successor.
    ///
    /// An adapter may report no successor for an exited body. Repeated
    /// successors are harmless: the core preserves only their first-seen
    /// occurrence.
    fn execute(
        &mut self,
        candidate: T,
        successors: &mut dyn FnMut(T) -> Result<(), DriveError<Self::Error>>,
    ) -> Result<(), DriveError<Self::Error>>;
}

/// An error from [`drive_generation`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DriveError<E> {
    /// The adapter failed while executing or folding a candidate body.
    Driver(E),
    /// The caller-owned candidate table could not represent the next set.
    Table(TableError),
}

/// Run one candidate generation through an ISA-owned [`CandidateDriver`].
///
/// Current candidates are presented in stable order. Their successors are
/// accumulated in the table's disjoint tail and committed only after every
/// callback succeeds; therefore a capacity error never overwrites the current
/// generation. The driver retains its own ISA snapshot/fold discipline.
pub fn drive_generation<T, D>(
    table: &mut CandidateTable<'_, T>,
    driver: &mut D,
) -> Result<usize, DriveError<D::Error>>
where
    T: Copy + Eq,
    D: CandidateDriver<T>,
{
    let current_len = table.len();
    let mut next_len = 0;
    for index in 0..current_len {
        let candidate = table.current()[index];
        let mut append = |successor| {
            next_len = table
                .append_next(next_len, successor)
                .map_err(DriveError::Table)?;
            Ok(())
        };
        driver.execute(candidate, &mut append)?;
    }
    table.finish_next(next_len).map_err(DriveError::Table)
}

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
    use super::{
        CandidateDriver, CandidateTable, DriveError, TableError, drive_generation, predicated_value,
    };

    #[test]
    fn next_generation_deduplicates_and_compacts_in_first_seen_order() {
        let mut backing = [0; 8];
        let mut table = CandidateTable::new(&mut backing, 1).unwrap();
        let mut next = table.next().unwrap();
        next.extend(&[4, 2, 4, 3, 2]).unwrap();
        assert_eq!(next.finish(), 3);
        assert_eq!(table.current(), &[4, 2, 3]);
    }

    struct Branches;

    impl CandidateDriver<u8> for Branches {
        type Error = ();

        fn execute(
            &mut self,
            candidate: u8,
            successors: &mut dyn FnMut(u8) -> Result<(), DriveError<Self::Error>>,
        ) -> Result<(), DriveError<Self::Error>> {
            successors(candidate + 1)?;
            successors(candidate + 2)?;
            successors(candidate + 1)
        }
    }

    #[test]
    fn driver_commits_a_deduplicated_generation_in_candidate_order() {
        let mut backing = [0; 8];
        let mut table = CandidateTable::new(&mut backing, 1).unwrap();
        table.replace(&[1, 2]).unwrap();
        assert_eq!(drive_generation(&mut table, &mut Branches), Ok(3));
        assert_eq!(table.current(), &[2, 3, 4]);
    }

    #[test]
    fn predication_keeps_an_inactive_value_and_selects_an_active_value() {
        assert_eq!(predicated_value(&mut (), false, true, false), Ok(false));
        assert_eq!(predicated_value(&mut (), true, true, false), Ok(true));
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
