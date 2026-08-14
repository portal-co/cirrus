#![no_std]
#![warn(missing_docs)]

//! First-row-fixed streaming garbling construction.

use core::{array, fmt, marker::PhantomData};

use cirrus_core::{
    Bit, ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithValue, HasError, Pusher,
};
use digest::{Digest, array::Array};

/// A garbler-side logical-zero wire label.
///
/// Garbling tracks a wire through the raw label for logical zero, not an
/// evaluator's selected label. [`Label::not`] therefore retains that
/// zero-label handle; the paired [`Evaluator`] represents the complement by
/// XORing its selected label with the free-XOR offset.
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

/// A three-row AND record produced by [`GC`].
///
/// Row zero is omitted. The evaluator derives it from the selected input
/// labels and the gate number; the stored rows are the original rows one
/// through three, in that order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GarblingRecord<const N: usize> {
    /// The three transmitted rows of one AND gate.
    Table([[u8; N]; 3]),
}

/// An error while replaying a row-reduced garbling stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluationError {
    /// An AND operation required a record after the iterator ended.
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

/// A streaming first-row-fixed garbler.
///
/// It retains only the free-XOR offset and an incrementing gate number. Every
/// AND emits three rows to `queue`; row zero is deterministically derived so
/// that the paired [`Evaluator`] can reconstruct it without a stored table.
/// This construction is a completeness and cost baseline, not an
/// authenticated-garbling protocol.
pub struct GC<'a, 'b, D: Digest, const N: usize> {
    queue: &'a mut (dyn Pusher<[[u8; N]; 3]> + 'b),
    delta: Array<u8, D::OutputSize>,
    gate: u64,
}

impl<'a, 'b, D: Digest, const N: usize> GC<'a, 'b, D, N> {
    /// Construct a three-row garbler with a free-XOR `delta`.
    ///
    /// `delta` must have a set low bit and at least `N` bytes. The latter is
    /// also required by the baseline garbler because a wire label is a prefix
    /// of the digest output.
    pub fn new(
        queue: &'a mut (dyn Pusher<[[u8; N]; 3]> + 'b),
        delta: Array<u8, D::OutputSize>,
    ) -> Self {
        assert!(N > 0, "wire labels must not be empty");
        assert!(N <= delta.len(), "wire labels exceed the digest output");
        assert!(
            delta[0] & 1 == 1,
            "the free-XOR offset must have a set low bit"
        );
        Self {
            queue,
            delta,
            gate: 0,
        }
    }

    fn row_zero_label(&self, left: [u8; N], right: [u8; N], gate: u64) -> [u8; N] {
        let mut digest = D::new();
        digest.update(b"cirrus/gc/first-row-fixed/v1");
        digest.update(gate.to_le_bytes());
        digest.update(left);
        digest.update(right);
        let digest = digest.finalize();
        array::from_fn(|index| digest[index])
    }

    fn selected_for_row_zero(&self, zero_label: [u8; N]) -> [u8; N] {
        // The evaluator's point-and-permute index is the inverse of the
        // label's low bit. Therefore its omitted row zero receives the label
        // whose low bit is one, not zero.
        if zero_label[0] & 1 == 1 {
            zero_label
        } else {
            array::from_fn(|index| zero_label[index] ^ self.delta[index])
        }
    }

    fn output_zero_label(&self, left: Label<N>, right: Label<N>, gate: u64) -> [u8; N] {
        let row_zero_left = self.selected_for_row_zero(left.zero_label);
        let row_zero_right = self.selected_for_row_zero(right.zero_label);
        let mut output = self.row_zero_label(row_zero_left, row_zero_right, gate);
        let row_zero_value = (left.zero_label[0] & 1 == 0) && (right.zero_label[0] & 1 == 0);
        if row_zero_value {
            for (output, delta) in output.iter_mut().zip(self.delta.clone()) {
                *output ^= delta;
            }
        }
        output
    }

    fn row(&self, output_zero: [u8; N], left: Label<N>, right: Label<N>, row: usize) -> [u8; N] {
        let left = (row & 1 != 0) ^ (left.zero_label[0] & 1 != 0) ^ (self.delta[0] & 1 != 0);
        let right = (row & 2 != 0) ^ (right.zero_label[0] & 1 != 0) ^ (self.delta[0] & 1 != 0);
        if left & right {
            array::from_fn(|index| output_zero[index] ^ self.delta[index])
        } else {
            output_zero
        }
    }
}

impl<D: Digest, const N: usize> HasError for GC<'_, '_, D, N> {
    type Error = core::convert::Infallible;
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
        left: <Self as ContextWithValue<bool>>::Wrapped,
        right: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<bool>>::Wrapped, Self::Error> {
        Ok(Label::new(array::from_fn(|index| {
            left.zero_label[index] ^ right.zero_label[index]
        })))
    }

    fn bitxor_assign(
        &mut self,
        left: &mut <Self as ContextWithValue<bool>>::Wrapped,
        right: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<(), Self::Error> {
        *left = self.bitxor(*left, right)?;
        Ok(())
    }
}

impl<D: Digest, const N: usize> ContextWithBitAnd<bool> for GC<'_, '_, D, N> {
    fn bitand(
        &mut self,
        left: <Self as ContextWithValue<bool>>::Wrapped,
        right: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<bool>>::Wrapped, Self::Error> {
        let gate = self.gate;
        self.gate = self.gate.wrapping_add(1);
        let output = self.output_zero_label(left, right, gate);
        self.queue.push(array::from_fn(|index| {
            self.row(output, left, right, index + 1)
        }));
        Ok(Label::new(output))
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

impl<D: Digest, const N: usize> ContextWithBitOr<bool> for GC<'_, '_, D, N> {
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

/// A pull-based evaluator for the first-row-fixed garbling format.
///
/// It derives row zero locally and still pulls one three-row record for every
/// AND gate, preserving the stream position even when row zero is selected.
/// As with [`GC`], this is a completeness adapter, not a deployment protocol.
pub struct Evaluator<D: Digest, I, const N: usize> {
    records: I,
    gate: u64,
    marker: PhantomData<D>,
}

impl<D: Digest, I, const N: usize> Evaluator<D, I, N>
where
    I: Iterator<Item = GarblingRecord<N>>,
{
    /// Construct an evaluator that pulls ordered three-row records.
    pub fn new(records: I) -> Self {
        assert!(N > 0, "wire labels must not be empty");
        assert!(
            N <= D::digest([]).len(),
            "wire labels exceed the digest output"
        );
        Self {
            records,
            gate: 0,
            marker: PhantomData,
        }
    }

    fn row_zero_label(&self, left: [u8; N], right: [u8; N], gate: u64) -> [u8; N] {
        let mut digest = D::new();
        digest.update(b"cirrus/gc/first-row-fixed/v1");
        digest.update(gate.to_le_bytes());
        digest.update(left);
        digest.update(right);
        let digest = digest.finalize();
        array::from_fn(|index| digest[index])
    }

    fn next_table(&mut self) -> Result<[[u8; N]; 3], EvaluationError> {
        match self.records.next() {
            Some(GarblingRecord::Table(table)) => Ok(table),
            None => Err(EvaluationError::Exhausted),
        }
    }
}

impl<D: Digest, I, const N: usize> HasError for Evaluator<D, I, N>
where
    I: Iterator<Item = GarblingRecord<N>>,
{
    type Error = EvaluationError;
}

impl<D: Digest, I, const N: usize> ContextWithValue<bool> for Evaluator<D, I, N>
where
    I: Iterator<Item = GarblingRecord<N>>,
{
    type Wrapped = [u8; N];
}

impl<D: Digest, I, const N: usize> ContextWithBitXor<bool> for Evaluator<D, I, N>
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

impl<D: Digest, I, const N: usize> ContextWithBitAnd<bool> for Evaluator<D, I, N>
where
    I: Iterator<Item = GarblingRecord<N>>,
{
    fn bitand(
        &mut self,
        left: <Self as ContextWithValue<bool>>::Wrapped,
        right: <Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<bool>>::Wrapped, Self::Error> {
        let gate = self.gate;
        self.gate = self.gate.wrapping_add(1);
        let row = usize::from(left[0] & 1 == 0) | (usize::from(right[0] & 1 == 0) << 1);
        let table = self.next_table()?;
        if row == 0 {
            Ok(self.row_zero_label(left, right, gate))
        } else {
            Ok(table[row - 1])
        }
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

impl<D: Digest, I, const N: usize> ContextWithBitOr<bool> for Evaluator<D, I, N>
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

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use core::{array, convert::Infallible};
    use std::vec::Vec;

    use cirrus_core::{ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, Pusher};
    use cirrus_ert::{DefaultHandler, RawMemory, RvDefaultHandler, ert_emit};
    use rv_asm::{Inst, Reg, Xlen};
    use sha2::Sha256;

    use super::{EvaluationError, Evaluator, GC, GarblingRecord, Label};

    #[derive(Default)]
    struct Records<const N: usize>(Vec<[[u8; N]; 3]>);

    impl<const N: usize> Pusher<[[u8; N]; 3]> for Records<N> {
        fn push(&mut self, table: [[u8; N]; 3]) {
            self.0.push(table);
        }
    }

    fn label(zero: [u8; 16], value: bool) -> [u8; 16] {
        array::from_fn(|index| zero[index] ^ if value && index == 0 { 1 } else { 0 })
    }

    fn program(instructions: impl IntoIterator<Item = Inst>) -> Vec<u8> {
        instructions
            .into_iter()
            .flat_map(|instruction| instruction.encode_normal(Xlen::Rv32).to_le_bytes())
            .collect()
    }

    fn garbler_hash<C>(_: &mut C, _: &[[Label<16>; 32]]) -> Result<[u8; 32], Infallible> {
        Ok([0; 32])
    }

    fn evaluator_hash<C>(_: &mut C, _: &[[[u8; 16]; 32]]) -> Result<[u8; 32], EvaluationError> {
        Ok([0; 32])
    }

    #[test]
    fn evaluator_replays_all_and_truth_table_rows_from_three_row_records() {
        for left_zero in [[0; 16], [1; 16]] {
            for right_zero in [[2; 16], [3; 16]] {
                let mut records = Records::default();
                let delta = array::from_fn(|index| (index == 0) as u8);
                let mut garbler = GC::<Sha256, 16>::new(&mut records, delta.into());
                let result_zero = garbler
                    .bitand(Label::new(left_zero), Label::new(right_zero))
                    .expect("garbling cannot fail");
                drop(garbler);
                let table = records.0[0];
                for (left, right) in [(false, false), (false, true), (true, false), (true, true)] {
                    let mut evaluator = Evaluator::<Sha256, _, 16>::new(
                        [table].into_iter().map(GarblingRecord::Table),
                    );
                    assert_eq!(
                        evaluator
                            .bitand(label(left_zero, left), label(right_zero, right))
                            .expect("one record is available for each replay"),
                        label(result_zero.zero_label(), left & right),
                        "left_zero={left_zero:?}, right_zero={right_zero:?}, left={left}, right={right}",
                    );
                }
            }
        }
    }

    #[test]
    fn evaluator_replays_an_and_after_an_affine_inversion() {
        let left_zero = Label::new([0; 16]);
        let right_zero = Label::new([2; 16]);
        let mut records = Records::default();
        let delta = array::from_fn(|index| (index == 0) as u8);
        let mut garbler = GC::<Sha256, 16>::new(&mut records, delta.into());
        let result_zero = garbler
            .bitand(left_zero.not(), right_zero)
            .expect("garbling cannot fail");
        drop(garbler);
        let table = records.0[0];
        for (left, right) in [(false, false), (false, true), (true, false), (true, true)] {
            let mut evaluator =
                Evaluator::<Sha256, _, 16>::new([table].into_iter().map(GarblingRecord::Table));
            let inverted_left = evaluator
                .bitxor(
                    label(left_zero.zero_label(), left),
                    array::from_fn(|index| (index == 0) as u8),
                )
                .expect("XOR is free");
            assert_eq!(
                evaluator
                    .bitand(inverted_left, label(right_zero.zero_label(), right))
                    .expect("one table is available"),
                label(result_zero.zero_label(), !left & right),
                "left={left}, right={right}",
            );
        }
    }

    #[test]
    fn evaluator_replays_an_ert_add_from_three_row_records() {
        let instructions = program([
            Inst::Add {
                dest: Reg::T0,
                src1: Reg::A1,
                src2: Reg::A2,
            },
            Inst::Ecall,
        ]);
        let zero = [0; 16];
        let one = array::from_fn(|index| (index == 0) as u8);
        let garbling_zero = Label::new(zero);
        let garbling_one = garbling_zero.not();
        let delta = array::from_fn(|index| (index == 0) as u8).into();
        let left_value: u32 = 0x1020_3040;
        let right_value: u32 = 0x0102_0304;
        let right_zero = [2; 16];
        let left = array::from_fn(|bit| label(zero, (left_value >> bit) & 1 != 0));
        let right = array::from_fn(|bit| label(right_zero, (right_value >> bit) & 1 != 0));
        let mut garbled_registers = [[garbling_zero; 32]; 32];
        garbled_registers[Reg::A0.0 as usize] = [garbling_one; 32];
        garbled_registers[Reg::A1.0 as usize] = [garbling_zero; 32];
        garbled_registers[Reg::A2.0 as usize] = [Label::new(right_zero); 32];
        let mut garbled_constants = [None; 32];
        garbled_constants[Reg::A0.0 as usize] = Some(u32::MAX);
        let mut garbled_rstack = [0; 8];
        let mut garbled_vstack = [garbling_zero; 64];
        let mut records = Records::default();
        let garbler = GC::<Sha256, 16>::new(&mut records, delta);
        let mut handler = RvDefaultHandler {
            inner: DefaultHandler {
                context: garbler,
                hash: garbler_hash,
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
        let mut evaluator =
            Evaluator::<Sha256, _, 16>::new(records.0.into_iter().map(GarblingRecord::Table));
        let mut evaluated_registers = [[zero; 32]; 32];
        evaluated_registers[Reg::A0.0 as usize] = [one; 32];
        evaluated_registers[Reg::A1.0 as usize] = left;
        evaluated_registers[Reg::A2.0 as usize] = right;
        let mut evaluated_constants = [None; 32];
        evaluated_constants[Reg::A0.0 as usize] = Some(u32::MAX);
        let mut evaluated_rstack = [0; 8];
        let mut evaluated_vstack = [zero; 64];
        let mut handler = RvDefaultHandler {
            inner: DefaultHandler {
                context: evaluator,
                hash: evaluator_hash,
            },
        };

        let evaluated = ert_emit(
            &mut handler,
            RawMemory::from(instructions.as_slice()),
            &mut evaluated_rstack,
            &mut evaluated_vstack,
            0,
            &mut evaluated_registers,
            &mut evaluated_constants,
            zero,
            one,
        );
        assert!(evaluated.is_ok(), "evaluation consumes the ordered records");

        let result = left_value.wrapping_add(right_value);
        for bit in 0..32 {
            assert_eq!(
                evaluated_registers[Reg::T0.0 as usize][bit],
                label(garbled_result[bit].zero_label(), (result >> bit) & 1 != 0),
                "result bit {bit}",
            );
        }
        assert_eq!(
            handler.inner.context.bitand(zero, zero),
            Err(EvaluationError::Exhausted)
        );
    }

    #[test]
    fn evaluator_replays_a_long_mixed_boolean_stream() {
        let mut records = Records::default();
        let delta = array::from_fn(|index| (index == 0) as u8);
        let mut garbler = GC::<Sha256, 16>::new(&mut records, delta.into());
        let mut garbled_left = Label::new([0; 16]);
        let mut garbled_right = Label::new([2; 16]);
        for round in 0..512 {
            match round % 3 {
                0 => garbled_left = garbler.bitand(garbled_left, garbled_right).unwrap(),
                1 => garbled_right = garbler.bitor(garbled_left, garbled_right).unwrap(),
                _ => garbled_left = garbler.bitxor(garbled_left, garbled_right).unwrap(),
            }
        }
        drop(garbler);

        let mut evaluator =
            Evaluator::<Sha256, _, 16>::new(records.0.into_iter().map(GarblingRecord::Table));
        let mut left = label([0; 16], true);
        let mut right = label([2; 16], false);
        let mut left_value = true;
        let mut right_value = false;
        for round in 0..512 {
            match round % 3 {
                0 => {
                    left = evaluator.bitand(left, right).unwrap();
                    left_value &= right_value;
                }
                1 => {
                    right = evaluator.bitor(left, right).unwrap();
                    right_value = left_value | right_value;
                }
                _ => {
                    left = evaluator.bitxor(left, right).unwrap();
                    left_value ^= right_value;
                }
            }
        }
        assert_eq!(left, label(garbled_left.zero_label(), left_value));
        assert_eq!(right, label(garbled_right.zero_label(), right_value));
    }
}
