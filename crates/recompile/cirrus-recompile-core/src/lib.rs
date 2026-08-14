#![no_std]
#![warn(missing_docs)]

//! The shared "recompile" IR: a flat, loop-free trace of Boolean-circuit
//! operations recorded from a single run of an ERT interpreter, plus the
//! [`Recorder`] `Context` that produces it.
//!
//! ERT never branches on a symbolic value -- every conditional jump/call
//! target is resolved through the interpreter's concrete `u32` shadow, and an
//! attempt to branch on an unknown value is a hard error (see `cirrus-ert`'s
//! and `cirrus-armv8m-ert`'s `handlers::branch`). So for a workload whose
//! control flow does not depend on secret data, plugging [`Recorder`] into
//! `ert_func`/`ert_emit` in place of a live `Context` (`()`, `GC`,
//! `Evaluator`, ...) captures the exact, fully-linearized sequence of Boolean
//! operations that workload performs -- the same shape a garbler already
//! walks, just recorded instead of evaluated. That recorded [`Program`] is
//! the shared input every code-generating backend (assembly, LLVM, Rust)
//! lowers from.
//!
//! # The scratch-buffer design
//!
//! An [`Op`]'s own position in [`Program::ops`] is its output [`Idx`], the
//! same implicit-numbering trick an SSA basic block's instruction list uses.
//! That fixes the value size ahead of time: a backend's scratch buffer for
//! this `Program` is exactly `program.ops.len()` slots of whatever type the
//! target "source type" uses (`bool`, i.e. one byte per slot, for the v1
//! scope this crate covers). A backend never manipulates a slot's value
//! directly -- it only ever manipulates [`Idx`]es, and defers the actual
//! Boolean operation to a pinned runtime function (see `cirrus-recompile-rt`)
//! that reads and writes the buffer by index. That is also why explicit
//! register allocation is largely unnecessary for the LLVM and Rust
//! backends -- their own compilers already regalloc local values -- and why,
//! for the hand-emitted assembly backend, the regalloc that *is* needed is
//! over small integer indices rather than over full circuit values.

extern crate alloc;

use alloc::vec::Vec;
use core::convert::Infallible;

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux,
    ContextWithValue, HasError,
};

/// A slot index into a backend's scratch value buffer.
///
/// Every backend `Context`'s `Wrapped` type is `Idx`: recording never
/// computes a Boolean value, it only ever names the slot a future value will
/// occupy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Idx(pub u32);

impl Idx {
    /// This index as a `usize`, for indexing a host-side buffer.
    pub const fn get(self) -> usize {
        self.0 as usize
    }
}

/// One recorded Boolean-circuit operation.
///
/// Every variant names its operands by [`Idx`]; none carries its own output
/// slot; see the [module documentation](self) for why.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Op {
    /// Materialize a known Boolean constant into a fresh slot.
    Create(bool),
    /// Bitwise AND of two existing slots.
    BitAnd(Idx, Idx),
    /// Bitwise OR of two existing slots.
    BitOr(Idx, Idx),
    /// Bitwise XOR of two existing slots.
    BitXor(Idx, Idx),
    /// Select `then` when `cond` is set, else `r#else`.
    Mux {
        /// The Boolean condition slot.
        cond: Idx,
        /// The slot selected when `cond` holds.
        then: Idx,
        /// The slot selected when `cond` does not hold.
        r#else: Idx,
    },
}

/// A recorded, straight-line trace of Boolean-circuit operations.
///
/// `inputs` and `outputs` name the subset of slots a caller cares about after
/// recording; every other slot is scratch-only bookkeeping a backend is free
/// to discard once nothing downstream reads it.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Program {
    /// The recorded operation trace. `ops.len()` is the scratch buffer width.
    pub ops: Vec<Op>,
    /// The slots a caller supplied as this trace's inputs.
    pub inputs: Vec<Idx>,
    /// The slots a caller reads back as this trace's outputs.
    pub outputs: Vec<Idx>,
}

impl Program {
    /// The number of slots this program's scratch buffer needs.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether this program recorded no operations at all.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// A [`cirrus_core`] `Context` that records the shape of a Boolean circuit
/// instead of computing one.
///
/// `Recorder` implements exactly the operations `cirrus_ert_core`'s
/// `ContextWithErtOps` (`BitAnd`/`BitOr`/`BitXor`) and `cirrus_core`'s
/// `ContextWithCreate`/`ContextWithMux` need, with `Wrapped = Idx`. Plugging
/// it into `ert_func`/`ert_emit` in place of a live `Context` *is* the
/// "recompile" step described in the [module documentation](self); no
/// interpreter changes are needed, only this new `Context`.
#[derive(Clone, Debug, Default)]
pub struct Recorder {
    ops: Vec<Op>,
}

impl Recorder {
    /// Start recording an empty trace.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    fn push(&mut self, op: Op) -> Idx {
        let idx = Idx(self.ops.len() as u32);
        self.ops.push(op);
        idx
    }

    /// Finish recording, naming the slots a caller cares about as this
    /// trace's inputs and outputs.
    pub fn finish(self, inputs: Vec<Idx>, outputs: Vec<Idx>) -> Program {
        Program {
            ops: self.ops,
            inputs,
            outputs,
        }
    }
}

impl HasError for Recorder {
    type Error = Infallible;
}

impl ContextWithValue<bool> for Recorder {
    type Wrapped = Idx;
}

impl ContextWithCreate<bool> for Recorder {
    fn create(&mut self, val: bool) -> Result<Idx, Infallible> {
        Ok(self.push(Op::Create(val)))
    }
}

impl ContextWithBitAnd<bool> for Recorder {
    fn bitand(&mut self, a: Idx, b: Idx) -> Result<Idx, Infallible> {
        Ok(self.push(Op::BitAnd(a, b)))
    }

    fn bitand_assign(&mut self, a: &mut Idx, b: Idx) -> Result<(), Infallible> {
        *a = self.bitand(*a, b)?;
        Ok(())
    }
}

impl ContextWithBitOr<bool> for Recorder {
    fn bitor(&mut self, a: Idx, b: Idx) -> Result<Idx, Infallible> {
        Ok(self.push(Op::BitOr(a, b)))
    }

    fn bitor_assign(&mut self, a: &mut Idx, b: Idx) -> Result<(), Infallible> {
        *a = self.bitor(*a, b)?;
        Ok(())
    }
}

impl ContextWithBitXor<bool> for Recorder {
    fn bitxor(&mut self, a: Idx, b: Idx) -> Result<Idx, Infallible> {
        Ok(self.push(Op::BitXor(a, b)))
    }

    fn bitxor_assign(&mut self, a: &mut Idx, b: Idx) -> Result<(), Infallible> {
        *a = self.bitxor(*a, b)?;
        Ok(())
    }
}

impl ContextWithMux<bool> for Recorder {
    fn mux(&mut self, cond: Idx, then: Idx, r#else: Idx) -> Result<Idx, Infallible> {
        Ok(self.push(Op::Mux { cond, then, r#else }))
    }
}

/// Interpret a [`Program`] directly over `bool`, as a second, IR-level oracle
/// independent of any code-generating backend.
///
/// This is the "including with the ERT" cross-check described in the
/// project's test plan: it holds the recorded trace itself accountable
/// (catching a `Recorder` bug that no backend-specific bug would produce),
/// separately from each backend's own lowering.
pub fn interpret(program: &Program, inputs: &[bool]) -> Vec<bool> {
    assert_eq!(
        inputs.len(),
        program.inputs.len(),
        "input count must match the recorded program's input slots"
    );
    let mut slots: Vec<Option<bool>> = alloc::vec![None; program.ops.len()];
    for (&idx, &value) in program.inputs.iter().zip(inputs) {
        slots[idx.get()] = Some(value);
    }
    for (i, op) in program.ops.iter().enumerate() {
        if slots[i].is_some() {
            // Already supplied as a named input; the recorded `Create` (if
            // any) at this slot is redundant with the caller-supplied value.
            continue;
        }
        let value = match *op {
            Op::Create(v) => v,
            Op::BitAnd(a, b) => slots[a.get()].unwrap() & slots[b.get()].unwrap(),
            Op::BitOr(a, b) => slots[a.get()].unwrap() | slots[b.get()].unwrap(),
            Op::BitXor(a, b) => slots[a.get()].unwrap() ^ slots[b.get()].unwrap(),
            Op::Mux { cond, then, r#else } => {
                if slots[cond.get()].unwrap() {
                    slots[then.get()].unwrap()
                } else {
                    slots[r#else.get()].unwrap()
                }
            }
        };
        slots[i] = Some(value);
    }
    program
        .outputs
        .iter()
        .map(|idx| slots[idx.get()].unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorder_numbers_ops_by_position() {
        let mut recorder = Recorder::new();
        let a = recorder.create(true).unwrap();
        let b = recorder.create(false).unwrap();
        let c = ContextWithBitAnd::bitand(&mut recorder, a, b).unwrap();
        assert_eq!(a, Idx(0));
        assert_eq!(b, Idx(1));
        assert_eq!(c, Idx(2));
        let program = recorder.finish(alloc::vec![a, b], alloc::vec![c]);
        assert_eq!(program.len(), 3);
    }

    #[test]
    fn interpret_matches_recorded_bitwise_ops() {
        let mut recorder = Recorder::new();
        let a = recorder.create(false).unwrap();
        let b = recorder.create(false).unwrap();
        let and = ContextWithBitAnd::bitand(&mut recorder, a, b).unwrap();
        let or = ContextWithBitOr::bitor(&mut recorder, a, b).unwrap();
        let xor = ContextWithBitXor::bitxor(&mut recorder, a, b).unwrap();
        let mux = ContextWithMux::mux(&mut recorder, a, or, xor).unwrap();
        let program = recorder.finish(alloc::vec![a, b], alloc::vec![and, or, xor, mux]);

        for &(x, y) in &[(false, false), (false, true), (true, false), (true, true)] {
            let outputs = interpret(&program, &[x, y]);
            assert_eq!(outputs, [x & y, x | y, x ^ y, if x { x | y } else { x ^ y }]);
        }
    }
}
