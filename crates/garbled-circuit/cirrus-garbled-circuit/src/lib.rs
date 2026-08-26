#![no_std]

//! Streaming baseline garbled-circuit construction.
//!
//! [`GC`] is the stable, four-row-table baseline for the Boolean context seam
//! shared by the symbolic interpreters. XOR is free; AND emits one table, and
//! OR is synthesized from those two operations. It deliberately knows nothing
//! about transport, allocation, or task scheduling.
//!
//! # Streaming contract
//!
//! A caller supplies a [`Pusher`] as the table sink. Every non-free gate calls
//! [`Pusher::push`] synchronously, in circuit order. Production adapters must
//! consume or durably hand off that table before returning; they must not retain
//! an unbounded circuit or silently drop a table when a network buffer is full.
//! The sink is the seam at which an embedded integrator can use a small frame
//! buffer and its own coroutine or event loop to transmit tables while the
//! circuit is garbled.
//!
//! The synchronous interface intentionally does not prescribe an async runtime
//! or a buffering policy. A blocking network write, a bounded driver queue, and
//! a cooperative producer are all valid adapters, provided that backpressure is
//! resolved before the next table is accepted.
//!
//! # Embedded ERT use
//!
//! `GC` implements the native-`bool` Boolean context traits used by
//! `cirrus-ert` and `cirrus-armv8m-ert`. The locked SHA-256 self-tests provide
//! the current traffic baseline: at 16-byte labels, RV32 emits 429,216 tables
//! (27.5 MB) and Thumb emits 164,288 tables (10.5 MB). These are generated
//! bytes, not required RAM when the sink streams. Integrators must still budget
//! wire registers, the symbolic stack, the return stack, and their bounded
//! transport buffer.

#[cfg(test)]
extern crate std;

use core::{array, convert::Infallible, fmt, marker::PhantomData};

use cirrus_core::{
    Bit, ContextWithAdd, ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithMul,
    ContextWithSub, ContextWithValue, HasError, Pusher,
};
use digest::{Digest, array::Array};

/// A garbler-side logical-zero wire label.
///
/// Garbling tracks a wire through the raw label for logical zero, not the
/// selected label currently held by an evaluator. Consequently, [`Label::not`]
/// retains that zero-label handle: the paired evaluator represents the
/// complement by XORing its selected label with the free-XOR offset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Label<const N: usize> {
    zero_label: [u8; N],
}

impl<const N: usize> Label<N> {
    /// Create a wire from its raw logical-zero label.
    pub const fn new(zero_label: [u8; N]) -> Self {
        Self { zero_label }
    }

    /// Return the logical complement of this wire's zero-label handle.
    pub const fn not(self) -> Self {
        self
    }

    /// Return the raw label for logical zero.
    pub const fn zero_label(self) -> [u8; N] {
        self.zero_label
    }
}

/// A four-row garbling context that emits each non-free gate to a streaming sink.
pub struct GC<'a, 'b, D: Digest, const N: usize> {
    /// The ordered streaming destination for four-row AND tables.
    pub queue: &'a mut (dyn Pusher<[[u8; N]; 4]> + 'b),
    /// The evolving digest state used to form each new output-wire label.
    pub seed: Array<u8, D::OutputSize>,
    /// The global free-XOR offset; its low bit must be set by the caller.
    pub delta: Array<u8, D::OutputSize>,
}

/// One record in the baseline garbling stream.
///
/// The baseline presently emits only four-row tables. Future backends may have
/// different record types, including hints, but retain the same pull-based
/// evaluator shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GarblingRecord<const N: usize> {
    /// A four-row AND table in the order emitted by [`GC`].
    Table([[u8; N]; 4]),
}

/// An error while replaying a garbling stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluationError {
    /// An AND operation required a table after the record iterator ended.
    Exhausted,
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhausted => formatter.write_str("garbling record iterator is exhausted"),
        }
    }
}

impl core::error::Error for EvaluationError {}

/// A pull-based evaluator for the four-row baseline garbling format.
///
/// The evaluator consumes exactly one [`GarblingRecord::Table`] for every AND
/// operation and none for XOR. It is a host-side completeness adapter, not a
/// network protocol, authentication mechanism, or durable table buffer.
/// `N` is the nonzero byte width of a wire label. The matching [`GC`] run
/// propagates zero labels; the evaluator starts from the selected input labels.
pub struct Evaluator<I, const N: usize> {
    records: I,
    marker: PhantomData<[u8; N]>,
}

impl<I, const N: usize> Evaluator<I, N>
where
    I: Iterator<Item = GarblingRecord<N>>,
{
    /// Construct an evaluator that pulls ordered records from `records`.
    pub fn new(records: I) -> Self {
        Self {
            records,
            marker: PhantomData,
        }
    }

    fn next_table(&mut self) -> Result<[[u8; N]; 4], EvaluationError> {
        match self.records.next() {
            Some(GarblingRecord::Table(table)) => Ok(table),
            None => Err(EvaluationError::Exhausted),
        }
    }
}

impl<I, const N: usize> HasError for Evaluator<I, N>
where
    I: Iterator<Item = GarblingRecord<N>>,
{
    type Error = EvaluationError;
}

impl<I, const N: usize> ContextWithValue<bool> for Evaluator<I, N>
where
    I: Iterator<Item = GarblingRecord<N>>,
{
    type Wrapped = [u8; N];
}

impl<I, const N: usize> ContextWithBitXor<bool> for Evaluator<I, N>
where
    I: Iterator<Item = GarblingRecord<N>>,
{
    fn bitxor(
        &mut self,
        left: <Self as ContextWithValue<bool>>::Wrapped,
        right: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<bool>>::Wrapped, Self::Error> {
        Ok(array::from_fn(|index| left[index] ^ right[index]))
    }

    fn bitxor_assign(
        &mut self,
        left: &mut <Self as ContextWithValue<bool>>::Wrapped,
        right: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<(), Self::Error> {
        for (left, right) in left.iter_mut().zip(right) {
            *left ^= right;
        }
        Ok(())
    }
}

impl<I, const N: usize> ContextWithBitAnd<bool> for Evaluator<I, N>
where
    I: Iterator<Item = GarblingRecord<N>>,
{
    fn bitand(
        &mut self,
        left: <Self as ContextWithValue<bool>>::Wrapped,
        right: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<bool>>::Wrapped, Self::Error> {
        let table = self.next_table()?;
        let row = usize::from(left[0] & 1 == 0) | (usize::from(right[0] & 1 == 0) << 1);
        Ok(table[row])
    }

    fn bitand_assign(
        &mut self,
        left: &mut <Self as ContextWithValue<bool>>::Wrapped,
        right: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<(), Self::Error> {
        *left = self.bitand(*left, right)?;
        Ok(())
    }
}

impl<I, const N: usize> ContextWithBitOr<bool> for Evaluator<I, N>
where
    I: Iterator<Item = GarblingRecord<N>>,
{
    fn bitor(
        &mut self,
        left: <Self as ContextWithValue<bool>>::Wrapped,
        right: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<bool>>::Wrapped, Self::Error> {
        let either = self.bitxor(left, right)?;
        let both = self.bitand(left, right)?;
        self.bitxor(either, both)
    }

    fn bitor_assign(
        &mut self,
        left: &mut <Self as ContextWithValue<bool>>::Wrapped,
        right: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<(), Self::Error> {
        *left = self.bitor(*left, right)?;
        Ok(())
    }
}

impl<D: Digest, const N: usize> HasError for GC<'_, '_, D, N> {
    type Error = Infallible;
}
impl<D: Digest, const N: usize> ContextWithValue<Bit> for GC<'_, '_, D, N> {
    type Wrapped = Label<N>;
}
impl<D: Digest, const N: usize> ContextWithValue<bool> for GC<'_, '_, D, N> {
    type Wrapped = Label<N>;
}
impl<D: Digest, const N: usize> ContextWithBitXor<bool> for GC<'_, '_, D, N> {
    fn bitxor(
        &mut self,
        a: <Self as ContextWithValue<bool>>::Wrapped,
        b: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<bool>>::Wrapped, Self::Error> {
        Ok(Label::new(array::from_fn(|i| {
            a.zero_label[i] ^ b.zero_label[i]
        })))
    }

    fn bitxor_assign(
        &mut self,
        a: &mut <Self as ContextWithValue<bool>>::Wrapped,
        b: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<(), Self::Error> {
        *a = self.bitxor(*a, b)?;
        Ok(())
    }
}
impl<D: Digest, const N: usize> ContextWithBitAnd<bool> for GC<'_, '_, D, N> {
    fn bitand(
        &mut self,
        a: <Self as ContextWithValue<bool>>::Wrapped,
        b: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<bool>>::Wrapped, Self::Error> {
        <Self as ContextWithMul<Bit>>::mul(self, a, b)
    }

    fn bitand_assign(
        &mut self,
        a: &mut <Self as ContextWithValue<bool>>::Wrapped,
        b: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<(), Self::Error> {
        *a = self.bitand(*a, b)?;
        Ok(())
    }
}
impl<D: Digest, const N: usize> ContextWithBitOr<bool> for GC<'_, '_, D, N> {
    fn bitor(
        &mut self,
        a: <Self as ContextWithValue<bool>>::Wrapped,
        b: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<bool>>::Wrapped, Self::Error> {
        let either = self.bitxor(a, b)?;
        let both = self.bitand(a, b)?;
        self.bitxor(either, both)
    }

    fn bitor_assign(
        &mut self,
        a: &mut <Self as ContextWithValue<bool>>::Wrapped,
        b: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<(), Self::Error> {
        *a = self.bitor(*a, b)?;
        Ok(())
    }
}
impl<D: Digest, const N: usize> ContextWithAdd<Bit> for GC<'_, '_, D, N> {
    fn add(
        &mut self,
        a: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
    ) -> Result<
        <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        <Self as cirrus_core::HasError>::Error,
    > {
        self.bitxor(a, b)
    }

    fn add_assign(
        &mut self,
        a: &mut <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
    ) -> Result<(), <Self as cirrus_core::HasError>::Error> {
        *a = self.bitxor(*a, b)?;
        Ok(())
    }
}
impl<D: Digest, const N: usize> ContextWithSub<Bit> for GC<'_, '_, D, N> {
    fn sub(
        &mut self,
        a: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
    ) -> Result<
        <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        <Self as cirrus_core::HasError>::Error,
    > {
        self.bitxor(a, b)
    }

    fn sub_assign(
        &mut self,
        a: &mut <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
    ) -> Result<(), <Self as cirrus_core::HasError>::Error> {
        *a = self.bitxor(*a, b)?;
        Ok(())
    }
}
impl<D: Digest, const N: usize> ContextWithMul<Bit> for GC<'_, '_, D, N> {
    fn mul(
        &mut self,
        a: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
    ) -> Result<
        <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        <Self as cirrus_core::HasError>::Error,
    > {
        self.seed = D::digest(&self.seed);
        let new = self.seed.clone();
        let new: [u8; N] = array::from_fn(|i| new[i]);
        self.queue.push(array::from_fn(|i| {
            let a = ((i & 1) == 1) ^ (a.zero_label[0] & 0x01 == 1) ^ (self.delta[0] & 0x01 == 1);
            let b = ((i & 2) == 2) ^ (b.zero_label[0] & 0x01 == 1) ^ (self.delta[0] & 0x01 == 1);
            let r = a & b;
            let mut x = new.clone();
            if r {
                for (a, b) in x.iter_mut().zip(self.delta.clone()) {
                    *a ^= b
                }
            }
            return x;
        }));
        Ok(Label::new(new))
    }

    fn mul_assign(
        &mut self,
        a: &mut <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
    ) -> Result<(), <Self as cirrus_core::HasError>::Error> {
        *a = self.mul(*a, b)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use core::{array, convert::Infallible};
    use std::{
        env, eprintln, fs,
        path::{Path, PathBuf},
        process::{Command, Output},
        string::String,
        vec::Vec,
    };

    use cirrus_armv8m_ert::{
        ArmDefaultHandler, DefaultHandler as ArmHashHandler, SecurityAttribute, SecurityState,
        ert_func as arm_ert_func,
    };
    use cirrus_core::{
        ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithValue, HasError, Pusher,
    };
    use cirrus_ert::{
        DefaultHandler as RvHashHandler, RawMemory, RvDefaultHandler, ert_emit,
        ert_func as riscv_ert_func,
    };
    use lazy_repo::{CacheConfig, ChunkCodec, ChunkSink, MemorySource, Repository};
    use rv_asm::{Inst, Reg, Xlen};
    use sha2::Sha256;

    use super::{EvaluationError, Evaluator, GC, GarblingRecord, Label};

    #[derive(Default)]
    struct RecordedTables<const N: usize> {
        tables: Vec<[[u8; N]; 4]>,
    }

    impl<const N: usize> Pusher<[[u8; N]; 4]> for RecordedTables<N> {
        fn push(&mut self, table: [[u8; N]; 4]) {
            self.tables.push(table);
        }
    }

    #[derive(Default)]
    struct CountingPusher {
        tables: usize,
    }

    impl<const N: usize> Pusher<[[u8; N]; 4]> for CountingPusher {
        fn push(&mut self, _: [[u8; N]; 4]) {
            self.tables += 1;
        }
    }

    struct BoundedPusher<const N: usize> {
        tables: Vec<[[u8; N]; 4]>,
        capacity: usize,
        overflowed: bool,
    }

    impl<const N: usize> BoundedPusher<N> {
        fn new(capacity: usize) -> Self {
            Self {
                tables: Vec::new(),
                capacity,
                overflowed: false,
            }
        }
    }

    impl<const N: usize> Pusher<[[u8; N]; 4]> for BoundedPusher<N> {
        fn push(&mut self, table: [[u8; N]; 4]) {
            if self.tables.len() == self.capacity {
                self.overflowed = true;
            } else {
                self.tables.push(table);
            }
        }
    }

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    struct CircuitCounts {
        bitand: usize,
        bitor: usize,
        bitxor: usize,
    }

    impl CircuitCounts {
        fn and_tables(self) -> usize {
            self.bitand + self.bitor
        }

        fn free_xors(self) -> usize {
            self.bitxor + self.bitor * 2
        }
    }

    struct MeasuredGc<'a, 'b, const N: usize> {
        gc: GC<'a, 'b, Sha256, N>,
        counts: CircuitCounts,
    }

    impl<'a, 'b, const N: usize> MeasuredGc<'a, 'b, N> {
        fn new(gc: GC<'a, 'b, Sha256, N>) -> Self {
            Self {
                gc,
                counts: CircuitCounts::default(),
            }
        }
    }

    impl<const N: usize> HasError for MeasuredGc<'_, '_, N> {
        type Error = Infallible;
    }

    impl<const N: usize> ContextWithValue<bool> for MeasuredGc<'_, '_, N> {
        type Wrapped = Label<N>;
    }

    impl<const N: usize> ContextWithBitAnd<bool> for MeasuredGc<'_, '_, N> {
        fn bitand(&mut self, left: Label<N>, right: Label<N>) -> Result<Label<N>, Self::Error> {
            self.counts.bitand += 1;
            self.gc.bitand(left, right)
        }

        fn bitand_assign(
            &mut self,
            left: &mut Label<N>,
            right: Label<N>,
        ) -> Result<(), Self::Error> {
            *left = self.bitand(*left, right)?;
            Ok(())
        }
    }

    impl<const N: usize> ContextWithBitOr<bool> for MeasuredGc<'_, '_, N> {
        fn bitor(&mut self, left: Label<N>, right: Label<N>) -> Result<Label<N>, Self::Error> {
            self.counts.bitor += 1;
            self.gc.bitor(left, right)
        }

        fn bitor_assign(
            &mut self,
            left: &mut Label<N>,
            right: Label<N>,
        ) -> Result<(), Self::Error> {
            *left = self.bitor(*left, right)?;
            Ok(())
        }
    }

    impl<const N: usize> ContextWithBitXor<bool> for MeasuredGc<'_, '_, N> {
        fn bitxor(&mut self, left: Label<N>, right: Label<N>) -> Result<Label<N>, Self::Error> {
            self.counts.bitxor += 1;
            self.gc.bitxor(left, right)
        }

        fn bitxor_assign(
            &mut self,
            left: &mut Label<N>,
            right: Label<N>,
        ) -> Result<(), Self::Error> {
            *left = self.bitxor(*left, right)?;
            Ok(())
        }
    }

    fn context(queue: &mut RecordedTables<16>) -> GC<'_, '_, Sha256, 16> {
        let mut gc = GC {
            queue,
            seed: Default::default(),
            delta: Default::default(),
        };
        gc.delta[0] = 1;
        gc
    }

    fn measuring_context<const N: usize>(queue: &mut CountingPusher) -> GC<'_, '_, Sha256, N> {
        let mut gc = GC {
            queue,
            seed: Default::default(),
            delta: Default::default(),
        };
        gc.delta[0] = 1;
        gc
    }

    fn bounded_context(queue: &mut BoundedPusher<16>) -> GC<'_, '_, Sha256, 16> {
        let mut gc = GC {
            queue,
            seed: Default::default(),
            delta: Default::default(),
        };
        gc.delta[0] = 1;
        gc
    }

    fn program(instructions: impl IntoIterator<Item = Inst>) -> Vec<u8> {
        instructions
            .into_iter()
            .flat_map(|instruction| instruction.encode_normal(Xlen::Rv32).to_le_bytes())
            .collect()
    }

    fn no_hash<const N: usize, C>(_: &mut C, _: &[[Label<N>; 32]]) -> Result<[u8; 32], Infallible> {
        Ok([0; 32])
    }

    fn evaluator_no_hash<const N: usize, C>(
        _: &mut C,
        _: &[[[u8; N]; 32]],
    ) -> Result<[u8; 32], EvaluationError> {
        Ok([0; 32])
    }

    fn permit_all<H>(_: &mut H, _: SecurityState) -> bool {
        true
    }

    fn always_secure<H>(_: &mut H, _: u32) -> SecurityAttribute {
        SecurityAttribute::Secure
    }

    fn encoded_label(zero_label: [u8; 16], value: bool) -> [u8; 16] {
        array::from_fn(|byte| zero_label[byte] ^ if value && byte == 0 { 1 } else { 0 })
    }

    fn garbler_label<const N: usize>(zero_label: [u8; N]) -> Label<N> {
        Label::new(zero_label)
    }

    fn evaluate_and(table: &[[u8; 16]; 4], left: [u8; 16], right: [u8; 16]) -> [u8; 16] {
        let row = usize::from(left[0] & 1 == 0) | (usize::from(right[0] & 1 == 0) << 1);
        table[row]
    }

    #[test]
    fn boolean_xor_is_free_and_xors_wire_labels() {
        let mut tables = RecordedTables::default();
        let mut gc = context(&mut tables);

        let result = gc
            .bitxor(garbler_label([0x0f; 16]), garbler_label([0xf0; 16]))
            .expect("garbling cannot fail");

        assert_eq!(result.zero_label(), [0xff; 16]);
        assert!(tables.tables.is_empty());
    }

    #[test]
    fn boolean_and_emits_one_four_row_table() {
        let mut tables = RecordedTables::default();
        let mut gc = context(&mut tables);

        let result = gc
            .bitand(garbler_label([0; 16]), garbler_label([0; 16]))
            .expect("garbling cannot fail");

        assert_eq!(
            result.zero_label(),
            [
                0x66, 0x68, 0x7a, 0xad, 0xf8, 0x62, 0xbd, 0x77, 0x6c, 0x8f, 0xc1, 0x8b, 0x8e, 0x9f,
                0x8e, 0x20,
            ]
        );
        assert_eq!(tables.tables.len(), 1);
        assert_eq!(tables.tables[0][0][0], result.zero_label()[0] ^ 1);
        assert_eq!(tables.tables[0][1], result.zero_label());
        assert_eq!(tables.tables[0][2], result.zero_label());
        assert_eq!(tables.tables[0][3], result.zero_label());
        for left in [false, true] {
            for right in [false, true] {
                let actual = evaluate_and(
                    &tables.tables[0],
                    encoded_label([0; 16], left),
                    encoded_label([0; 16], right),
                );
                assert_eq!(actual, encoded_label(result.zero_label(), left & right));
            }
        }
    }

    #[test]
    fn evaluator_replays_and_labels_from_an_iterator() {
        let mut tables = RecordedTables::default();
        let mut gc = context(&mut tables);
        let zero_label = gc
            .bitand(garbler_label([0; 16]), garbler_label([0; 16]))
            .expect("garbling cannot fail");
        drop(gc);

        let table = tables.tables[0];
        let mut evaluator = Evaluator::new([table; 4].into_iter().map(GarblingRecord::Table));
        for left in [false, true] {
            for right in [false, true] {
                assert_eq!(
                    evaluator
                        .bitand(encoded_label([0; 16], left), encoded_label([0; 16], right))
                        .expect("the matching table is available"),
                    encoded_label(zero_label.zero_label(), left & right),
                );
            }
        }
        assert_eq!(
            evaluator.bitand([0; 16], [0; 16]),
            Err(EvaluationError::Exhausted)
        );
    }

    #[derive(Clone, Copy)]
    struct TableCodec;

    impl ChunkCodec<[[u8; 16]; 4]> for TableCodec {
        type Error = Infallible;

        fn encode(&self, table: &[[u8; 16]; 4], out: &mut Vec<u8>) -> Result<(), Self::Error> {
            for row in table {
                out.extend_from_slice(row);
            }
            Ok(())
        }

        fn decode(&self, bytes: &[u8]) -> Result<[[u8; 16]; 4], Self::Error> {
            let mut table = [[0; 16]; 4];
            for (row, encoded) in table.iter_mut().zip(bytes.chunks_exact(16)) {
                row.copy_from_slice(encoded);
            }
            Ok(table)
        }
    }

    #[test]
    fn baseline_evaluator_accepts_a_digest_checked_lazy_table() {
        let mut tables = RecordedTables::default();
        let mut gc = context(&mut tables);
        let zero = gc
            .bitand(garbler_label([0; 16]), garbler_label([0; 16]))
            .unwrap();
        drop(gc);

        let mut source = MemorySource::default();
        let mut bytes = Vec::new();
        TableCodec.encode(&tables.tables[0], &mut bytes).unwrap();
        let chunk = source.store(bytes).unwrap();
        let mut repository = Repository::new(
            source,
            CacheConfig {
                max_resident_bytes: 64,
                max_chunk_bytes: 64,
            },
        );
        let table = repository.decode(&chunk, &TableCodec).unwrap().value;
        let mut evaluator = Evaluator::new([GarblingRecord::Table(table)].into_iter());
        assert_eq!(
            evaluator
                .bitand(encoded_label([0; 16], true), encoded_label([0; 16], true))
                .unwrap(),
            encoded_label(zero.zero_label(), true),
        );
    }

    #[test]
    fn evaluator_replays_an_and_after_a_symbolic_inversion() {
        let mut tables = RecordedTables::default();
        let mut gc = context(&mut tables);
        let left_zero = garbler_label([0; 16]);
        let right_zero = garbler_label([2; 16]);
        let result_zero = gc
            .bitand(left_zero.not(), right_zero)
            .expect("garbling cannot fail");
        drop(gc);
        let table = tables.tables[0];
        let delta = array::from_fn(|byte| (byte == 0) as u8);

        for (left, right) in [(false, false), (false, true), (true, false), (true, true)] {
            let mut evaluator = Evaluator::new([table].into_iter().map(GarblingRecord::Table));
            let inverted_left = evaluator
                .bitxor(encoded_label(left_zero.zero_label(), left), delta)
                .expect("XOR is free");
            assert_eq!(
                evaluator
                    .bitand(inverted_left, encoded_label(right_zero.zero_label(), right))
                    .expect("one table is available"),
                encoded_label(result_zero.zero_label(), !left & right),
                "left={left}, right={right}",
            );
        }
    }

    #[test]
    fn evaluator_replays_an_ert_add_from_the_complete_table_iterator() {
        let instructions = program([
            Inst::Add {
                dest: Reg::T0,
                src1: Reg::A1,
                src2: Reg::A2,
            },
            Inst::Ecall,
        ]);
        let zero = [0; 16];
        let one = array::from_fn(|byte| (byte == 0) as u8);
        let garbling_zero = garbler_label(zero);
        let garbling_one = garbling_zero.not();
        let left_value: u32 = 0x1020_3040;
        let right_value: u32 = 0x0102_0304;
        let garbling_left = [garbling_zero; 32];
        let left = array::from_fn(|bit| encoded_label(zero, (left_value >> bit) & 1 != 0));
        let right_zero = [2; 16];
        let garbling_right = [garbler_label(right_zero); 32];
        let right = array::from_fn(|bit| encoded_label(right_zero, (right_value >> bit) & 1 != 0));
        let mut garbled_registers = [[garbling_zero; 32]; 32];
        garbled_registers[Reg::A0.0 as usize] = [garbling_one; 32];
        garbled_registers[Reg::A1.0 as usize] = garbling_left;
        garbled_registers[Reg::A2.0 as usize] = garbling_right;
        let mut garbled_constants = [None; 32];
        garbled_constants[Reg::A0.0 as usize] = Some(u32::MAX);
        let mut garbled_rstack = [0; 8];
        let mut garbled_vstack = [garbling_zero; 64];
        let mut tables = RecordedTables::default();
        let gc = context(&mut tables);
        let mut handler = RvDefaultHandler {
            inner: RvHashHandler {
                context: gc,
                hash: no_hash,
            },
        };

        let garbled = ert_emit(
            &mut handler,
            RawMemory::from(instructions.as_slice()),
            &mut garbled_rstack,
            &mut garbled_vstack,
            0,
            &mut garbled_registers,
            &mut garbled_constants,
            garbling_zero,
            garbling_one,
        );
        assert!(
            garbled.is_ok(),
            "garbling the supported add program succeeds"
        );
        drop(handler);

        let garbled_result = garbled_registers[Reg::T0.0 as usize];
        let mut evaluator = Evaluator::new(tables.tables.into_iter().map(GarblingRecord::Table));
        let mut evaluated_registers = [[zero; 32]; 32];
        evaluated_registers[Reg::A0.0 as usize] = [one; 32];
        evaluated_registers[Reg::A1.0 as usize] = left;
        evaluated_registers[Reg::A2.0 as usize] = right;
        let mut evaluated_constants = [None; 32];
        evaluated_constants[Reg::A0.0 as usize] = Some(u32::MAX);
        let mut evaluated_rstack = [0; 8];
        let mut evaluated_vstack = [zero; 64];
        let mut evaluator_handler = RvDefaultHandler {
            inner: RvHashHandler {
                context: evaluator,
                hash: evaluator_no_hash,
            },
        };

        let evaluated = ert_emit(
            &mut evaluator_handler,
            RawMemory::from(instructions.as_slice()),
            &mut evaluated_rstack,
            &mut evaluated_vstack,
            0,
            &mut evaluated_registers,
            &mut evaluated_constants,
            zero,
            one,
        );
        assert!(
            evaluated.is_ok(),
            "evaluation consumes every add table in order"
        );

        let result = left_value.wrapping_add(right_value);
        for bit in 0..32 {
            assert_eq!(
                evaluated_registers[Reg::T0.0 as usize][bit],
                encoded_label(garbled_result[bit].zero_label(), (result >> bit) & 1 != 0),
                "result bit {bit}",
            );
        }
        assert_eq!(
            evaluator_handler.inner.context.bitand(zero, zero),
            Err(EvaluationError::Exhausted)
        );
    }

    #[test]
    fn boolean_or_uses_one_and_table_and_free_xor_labels() {
        let mut tables = RecordedTables::default();
        let mut gc = context(&mut tables);

        let result = gc
            .bitor(garbler_label([0; 16]), garbler_label([0xff; 16]))
            .expect("garbling cannot fail");

        assert_eq!(
            result.zero_label(),
            [
                0x99, 0x97, 0x85, 0x52, 0x07, 0x9d, 0x42, 0x88, 0x93, 0x70, 0x3e, 0x74, 0x71, 0x60,
                0x71, 0xdf,
            ]
        );
        assert_eq!(tables.tables.len(), 1);
        for left in [false, true] {
            for right in [false, true] {
                let left_label = encoded_label([0; 16], left);
                let right_label = encoded_label([0xff; 16], right);
                let either: [u8; 16] = array::from_fn(|byte| left_label[byte] ^ right_label[byte]);
                let both = evaluate_and(&tables.tables[0], left_label, right_label);
                let actual: [u8; 16] = array::from_fn(|byte| either[byte] ^ both[byte]);
                assert_eq!(actual, encoded_label(result.zero_label(), left | right));
            }
        }
    }

    #[test]
    fn bounded_test_sink_reports_table_capacity_exhaustion() {
        let mut tables = BoundedPusher::new(1);
        let mut gc = bounded_context(&mut tables);

        gc.bitand(garbler_label([0; 16]), garbler_label([0; 16]))
            .expect("garbling cannot fail");
        gc.bitand(garbler_label([0; 16]), garbler_label([0; 16]))
            .expect("garbling cannot fail");
        drop(gc);

        assert_eq!(tables.tables.len(), 1);
        assert!(tables.overflowed);
    }

    #[test]
    fn riscv_ert_emit_garbles_a_symbolic_add_through_the_public_context_seam() {
        let instructions = program([
            Inst::Add {
                dest: Reg::T0,
                src1: Reg::A1,
                src2: Reg::A2,
            },
            Inst::Ecall,
        ]);
        let zero = [0; 16];
        let garbling_zero = garbler_label(zero);
        let garbling_one = garbling_zero.not();
        let mut registers = [[garbling_zero; 32]; 32];
        registers[Reg::A1.0 as usize] = [garbler_label([0x22; 16]); 32];
        registers[Reg::A2.0 as usize] = [garbler_label([0x44; 16]); 32];
        registers[Reg::A0.0 as usize] = [garbling_one; 32];
        let mut constants = [None; 32];
        constants[Reg::A0.0 as usize] = Some(u32::MAX);
        let mut rstack = [0; 8];
        let mut vstack = [garbling_zero; 64];
        let mut tables = RecordedTables::default();
        let gc = MeasuredGc::new(context(&mut tables));
        let mut handler = RvDefaultHandler {
            inner: RvHashHandler {
                context: gc,
                hash: no_hash,
            },
        };

        let result = ert_emit(
            &mut handler,
            RawMemory::from(instructions.as_slice()),
            &mut rstack,
            &mut vstack,
            0,
            &mut registers,
            &mut constants,
            garbling_zero,
            garbling_one,
        );

        let counts = handler.inner.context.counts;
        drop(handler);

        assert!(result.is_ok());
        assert_eq!(constants[Reg::T0.0 as usize], None);
        assert_eq!(tables.tables.len(), 160);
        assert_eq!(counts.and_tables(), tables.tables.len());
        assert_eq!(counts.free_xors(), 192);
    }

    fn garbled_riscv_mul_tables(
        right_constant: Option<u32>,
    ) -> (usize, CircuitCounts, Option<u32>) {
        let instructions = program([
            Inst::Mul {
                dest: Reg::T0,
                src1: Reg::A1,
                src2: Reg::A2,
            },
            Inst::Ecall,
        ]);
        let zero = [0; 16];
        let garbling_zero = garbler_label(zero);
        let garbling_one = garbling_zero.not();
        let mut registers = [[garbling_zero; 32]; 32];
        registers[Reg::A1.0 as usize] = [garbler_label([0x22; 16]); 32];
        registers[Reg::A2.0 as usize] = array::from_fn(|bit| {
            if right_constant.unwrap_or(0) & (1 << bit) == 0 {
                garbling_zero
            } else {
                garbling_one
            }
        });
        registers[Reg::A0.0 as usize] = [garbling_one; 32];
        let mut constants = [None; 32];
        constants[Reg::A0.0 as usize] = Some(u32::MAX);
        constants[Reg::A2.0 as usize] = right_constant;
        let mut rstack = [0; 8];
        let mut vstack = [garbling_zero; 64];
        let mut sink = CountingPusher::default();
        let gc = MeasuredGc::new(measuring_context(&mut sink));
        let mut handler = RvDefaultHandler {
            inner: RvHashHandler {
                context: gc,
                hash: no_hash::<16, _>,
            },
        };

        let result = ert_emit(
            &mut handler,
            RawMemory::from(instructions.as_slice()),
            &mut rstack,
            &mut vstack,
            0,
            &mut registers,
            &mut constants,
            garbling_zero,
            garbling_one,
        );
        let counts = handler.inner.context.counts;
        drop(handler);

        assert!(result.is_ok());
        assert_eq!(sink.tables, counts.and_tables());
        (sink.tables, counts, constants[Reg::T0.0 as usize])
    }

    #[test]
    fn a_concrete_multiplicand_uses_fewer_garbled_tables_than_a_symbolic_one() {
        let (symbolic_tables, symbolic_counts, symbolic_constant) = garbled_riscv_mul_tables(None);
        let (constant_tables, constant_counts, constant_result) =
            garbled_riscv_mul_tables(Some(0x0001_0001));

        assert!(symbolic_tables > constant_tables);
        assert!(symbolic_counts.free_xors() > constant_counts.free_xors());
        assert_eq!(symbolic_constant, None);
        assert_eq!(constant_result, None);
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct WorkloadMetrics {
        label_bytes: usize,
        tables: usize,
        table_bytes: usize,
        free_xors: usize,
        register_bytes: usize,
        supplied_stack_bytes: usize,
        touched_stack_bytes: usize,
        return_stack_slots: usize,
    }

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .and_then(|path| path.parent())
            .expect("the garbled-circuit crate lives under the workspace family directory")
            .to_path_buf()
    }

    fn assert_success(action: &str, output: &Output) {
        assert!(
            output.status.success(),
            "{action} failed with {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    fn ensure_target(target: &str) {
        let installed = Command::new("rustup")
            .args(["target", "list", "--installed"])
            .output()
            .expect("rustup is required to inspect installed targets");
        assert_success("listing Rust targets", &installed);
        if installed
            .stdout
            .split(|byte| *byte == b'\n')
            .any(|installed| installed == target.as_bytes())
        {
            return;
        }
        let install = Command::new("rustup")
            .args(["target", "add", target])
            .output()
            .expect("rustup is required to install the self-test target");
        assert_success("installing the bare-metal target", &install);
    }

    fn build_self_test(package: &str, target: &str) -> PathBuf {
        ensure_target(target);
        let root = workspace_root();
        let target_dir = root
            .join("target")
            .join("cirrus-garbled-circuit-self-tests")
            .join(package);
        let build = Command::new("cargo")
            .current_dir(&root)
            .args([
                "build",
                "-p",
                package,
                "--features",
                "bare-metal",
                "--target",
                target,
                "--release",
                "--target-dir",
            ])
            .arg(&target_dir)
            .env("RUSTFLAGS", "-C panic=abort")
            .output()
            .expect("cargo must build the bare-metal self-test");
        assert_success("building the bare-metal self-test", &build);
        target_dir.join(target).join("release").join(package)
    }

    fn elf_u16(image: &[u8], offset: usize) -> usize {
        u16::from_le_bytes(
            image[offset..offset + 2]
                .try_into()
                .expect("ELF header is complete"),
        ) as usize
    }

    fn elf_u32(image: &[u8], offset: usize) -> usize {
        u32::from_le_bytes(
            image[offset..offset + 4]
                .try_into()
                .expect("ELF header is complete"),
        ) as usize
    }

    fn elf_header(image: &[u8]) -> (usize, usize, usize, usize) {
        assert_eq!(&image[..4], b"\x7fELF", "image is an ELF file");
        assert_eq!(image[4], 1, "self-test ELF is 32-bit");
        assert_eq!(image[5], 1, "self-test ELF is little-endian");
        (
            elf_u32(image, 32),
            elf_u16(image, 46),
            elf_u16(image, 48),
            elf_u16(image, 50),
        )
    }

    fn section_header(image: &[u8], index: usize) -> usize {
        let (offset, size, count, _) = elf_header(image);
        assert!(index < count, "ELF section index is in range");
        offset + index * size
    }

    fn section_name<'a>(image: &'a [u8], header: usize) -> &'a str {
        let (_, _, _, strings) = elf_header(image);
        let strings = section_header(image, strings);
        let strings_offset = elf_u32(image, strings + 16);
        let strings_size = elf_u32(image, strings + 20);
        let start = strings_offset + elf_u32(image, header);
        let bytes = &image[start..strings_offset + strings_size];
        let length = bytes
            .iter()
            .position(|byte| *byte == 0)
            .expect("ELF section name is terminated");
        core::str::from_utf8(&bytes[..length]).expect("ELF section name is UTF-8")
    }

    fn selected_section(image: &[u8], wanted: &str) -> (usize, usize, usize) {
        let (_, _, count, _) = elf_header(image);
        for index in 0..count {
            let header = section_header(image, index);
            if section_name(image, header) == wanted {
                return (
                    elf_u32(image, header + 12),
                    elf_u32(image, header + 16),
                    elf_u32(image, header + 20),
                );
            }
        }
        panic!("ELF image lacks required section {wanted}");
    }

    fn mapped_image(image: &[u8], base: u32, names: &[&str]) -> Vec<u8> {
        let selected: Vec<_> = names
            .iter()
            .map(|name| selected_section(image, name))
            .collect();
        let end = selected
            .iter()
            .map(|(address, _, size)| address + size)
            .max()
            .expect("at least one ELF section is mapped");
        let base = base as usize;
        assert!(selected.iter().all(|(address, _, _)| *address >= base));
        let mut mapping = std::vec![0; end - base];
        for (address, offset, size) in selected {
            mapping[address - base..address - base + size]
                .copy_from_slice(&image[offset..offset + size]);
        }
        mapping
    }

    fn symbol_address(image: &[u8], wanted: &str) -> u32 {
        const SYMBOL_TABLE: usize = 2;
        let (_, _, count, _) = elf_header(image);
        for index in 0..count {
            let header = section_header(image, index);
            if elf_u32(image, header + 4) != SYMBOL_TABLE {
                continue;
            }
            let strings = section_header(image, elf_u32(image, header + 24));
            let strings_offset = elf_u32(image, strings + 16);
            let strings_size = elf_u32(image, strings + 20);
            let symbols_offset = elf_u32(image, header + 16);
            let symbols_size = elf_u32(image, header + 20);
            let entry_size = elf_u32(image, header + 36);
            assert_eq!(entry_size, 16, "self-test uses ELF32 symbol entries");
            for entry in (symbols_offset..symbols_offset + symbols_size).step_by(entry_size) {
                let name_offset = strings_offset + elf_u32(image, entry);
                let name_bytes = &image[name_offset..strings_offset + strings_size];
                let length = name_bytes
                    .iter()
                    .position(|byte| *byte == 0)
                    .expect("ELF symbol name is terminated");
                if core::str::from_utf8(&name_bytes[..length]).expect("symbol name is UTF-8")
                    == wanted
                {
                    return elf_u32(image, entry + 4) as u32;
                }
            }
        }
        panic!("ELF image lacks required symbol {wanted}");
    }

    fn mapped_memory(mapping: &[u8], base: u32) -> RawMemory<'static> {
        // SAFETY: `base` maps the supplied contiguous guest section range onto
        // `mapping`; the self-test executes only the selected code and rodata.
        unsafe { RawMemory::new(mapping.as_ptr().wrapping_sub(base as usize), None) }
    }

    fn symbolic_args<const N: usize>() -> [([Label<N>; 32], Option<u32>); 16] {
        array::from_fn(|word| {
            (
                array::from_fn(|bit| {
                    let wire = (word * 32 + bit + 2) as u16;
                    Label::new(array::from_fn(|byte| match byte {
                        0 => wire as u8,
                        1 => (wire >> 8) as u8,
                        _ => 0,
                    }))
                }),
                None,
            )
        })
    }

    fn touched_stack<const N: usize>(stack: &[Label<N>], sentinel: Label<N>) -> usize {
        stack
            .iter()
            .position(|label| *label != sentinel)
            .map(|lowest_touched| stack.len() - lowest_touched)
            .unwrap_or(0)
    }

    fn used_return_slots(stack: &[u32]) -> usize {
        stack
            .iter()
            .rposition(|slot| *slot != u32::MAX)
            .map(|last_touched| last_touched + 1)
            .unwrap_or(0)
    }

    fn metrics<const N: usize>(
        tables: usize,
        counts: CircuitCounts,
        register_words: usize,
        supplied_stack_slots: usize,
        touched_stack_slots: usize,
        return_stack_slots: usize,
    ) -> WorkloadMetrics {
        assert_eq!(tables, counts.and_tables());
        WorkloadMetrics {
            label_bytes: N,
            tables,
            table_bytes: tables * 4 * N,
            free_xors: counts.free_xors(),
            register_bytes: register_words * 32 * N,
            supplied_stack_bytes: supplied_stack_slots * N,
            touched_stack_bytes: touched_stack_slots * N,
            return_stack_slots,
        }
    }

    fn garble_riscv_sha256_self_test<const N: usize>(image: &Path) -> WorkloadMetrics {
        const BASE: u32 = 0x8000_0000;
        const STACK_SLOTS: usize = 65_536;
        let elf = fs::read(image).expect("built RV32 ELF is readable");
        let mapping = mapped_image(&elf, BASE, &[".text.ert_workload", ".rodata.ert_workload"]);
        let entry = symbol_address(&elf, "__ert_workload_entry");
        let memory = mapped_memory(&mapping, BASE);
        let zero = Label::new([0; N]);
        let one = zero.not();
        let sentinel = Label::new([0xa5; N]);
        let mut registers = [[zero; 32]; 32];
        let mut constants = [None; 32];
        let mut rstack = [u32::MAX; 256];
        let mut vstack = std::vec![sentinel; STACK_SLOTS];
        let mut sink = CountingPusher::default();
        let gc = MeasuredGc::new(measuring_context(&mut sink));
        let mut handler = RvDefaultHandler {
            inner: RvHashHandler {
                context: gc,
                hash: no_hash::<N, _>,
            },
        };

        let outcome = riscv_ert_func::<_, _, 16, 2>(
            &mut handler,
            memory,
            &mut rstack,
            &mut vstack,
            entry,
            &mut registers,
            &mut constants,
            zero,
            one,
            symbolic_args(),
        );
        let counts = handler.inner.context.counts;
        drop(handler);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(_) => panic!("the locked RV32 SHA-256 workload must stay in the ERT subset"),
        };
        assert_eq!(outcome[0].1, Some(u32::MAX));
        assert_eq!(outcome[1].1, None);
        metrics::<N>(
            sink.tables,
            counts,
            32,
            STACK_SLOTS,
            touched_stack(&vstack, sentinel),
            used_return_slots(&rstack),
        )
    }

    fn garble_arm_sha256_self_test<const N: usize>(image: &Path) -> WorkloadMetrics {
        const BASE: u32 = 0x2000_0000;
        const STACK_SLOTS: usize = 131_072;
        let elf = fs::read(image).expect("built Arm ELF is readable");
        let mapping = mapped_image(&elf, BASE, &[".text.ert_workload", ".rodata.ert_workload"]);
        let entry = symbol_address(&elf, "__ert_workload_entry") | 1;
        let memory = mapped_memory(&mapping, BASE);
        let zero = Label::new([0; N]);
        let one = zero.not();
        let sentinel = Label::new([0xa5; N]);
        let mut registers = [[zero; 32]; 16];
        let mut constants = [None; 16];
        let mut rstack = [u32::MAX; 512];
        let mut vstack = std::vec![sentinel; STACK_SLOTS];
        let mut sink = CountingPusher::default();
        let gc = MeasuredGc::new(measuring_context(&mut sink));
        let mut handler = ArmDefaultHandler {
            inner: ArmHashHandler {
                context: gc,
                hash: no_hash::<N, _>,
            },
            svc_permitted: permit_all,
            security_attribute: always_secure,
        };

        let outcome = arm_ert_func::<_, _, 16, 2>(
            &mut handler,
            memory,
            &mut rstack,
            &mut vstack,
            entry,
            &mut registers,
            &mut constants,
            zero,
            one,
            symbolic_args(),
        );
        let counts = handler.inner.context.counts;
        drop(handler);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(_) => panic!("the locked Arm SHA-256 workload must stay in the ERT subset"),
        };
        assert_eq!(outcome[0].1, Some(u32::MAX));
        assert_eq!(outcome[1].1, None);
        metrics::<N>(
            sink.tables,
            counts,
            16,
            STACK_SLOTS,
            touched_stack(&vstack, sentinel),
            used_return_slots(&rstack),
        )
    }

    #[test]
    fn locked_rv32_and_thumb_sha256_selftests_garble_with_a_streaming_table_sink() {
        let riscv_image = build_self_test("cirrus-ert-selftest", "riscv32im-unknown-none-elf");
        let arm_image = build_self_test("cirrus-armv8m-ert-selftest", "thumbv8m.main-none-eabi");

        let rv16 = garble_riscv_sha256_self_test::<16>(&riscv_image);
        let rv32 = garble_riscv_sha256_self_test::<32>(&riscv_image);
        let arm16 = garble_arm_sha256_self_test::<16>(&arm_image);
        let arm32 = garble_arm_sha256_self_test::<32>(&arm_image);

        assert_eq!(rv16.tables, rv32.tables);
        assert_eq!(arm16.tables, arm32.tables);
        assert!(rv16.tables > 0 && arm16.tables > 0);
        assert!(rv16.free_xors > 0 && arm16.free_xors > 0);
        assert_eq!(rv16.return_stack_slots, 2);
        assert_eq!(arm16.return_stack_slots, 2);
        assert_eq!(
            core::mem::size_of::<CountingPusher>(),
            core::mem::size_of::<usize>()
        );
        eprintln!("RV32 SHA-256 garbling (16-byte labels): {rv16:?}");
        eprintln!("RV32 SHA-256 garbling (32-byte labels): {rv32:?}");
        eprintln!("Arm SHA-256 garbling (16-byte labels): {arm16:?}");
        eprintln!("Arm SHA-256 garbling (32-byte labels): {arm32:?}");
    }
}
