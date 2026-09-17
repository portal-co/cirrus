#![no_std]
#![warn(missing_docs)]

//! Thumb-2 loop-adapter state for the shared ERT scheduler.
//!
//! This crate deliberately begins with only the fail-closed concrete-state
//! merge contract. The Arm interpreter owns instruction semantics; the shared
//! loop core owns candidate scheduling. [`ThumbAgreement`] is the bridge: at a
//! multi-candidate boundary, every host-only Arm state component must agree.
//! In particular that includes SP, private return-stack state, ITSTATE, and
//! virtual TrustZone state. A later wire-backed register/NZCV fold can only
//! run after this agreement check has succeeded.

use cirrus_armv8m_ert::{Flag, Machine, REG_COUNT, SecurityState};
use cirrus_ert_core::ContextWithErtOps;

mod body;
mod body_api;
mod generation;

pub use body::{ThumbBoundary, execute_body};
pub use body_api::execute_snapshot;
pub use generation::{ThumbGenerationDriver, run_generation};

/// A snapshot could not fit in the caller-provided fixed return-frame array.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThumbSnapshotError {
    /// The machine's return-stack depth exceeds the fixed snapshot capacity.
    ReturnStackCapacity,
}

/// The complete Thumb state that crosses a loop-body boundary.
///
/// Register wires and lazy NZCVQ flags remain symbolic values, while the
/// concrete metadata and virtual-machine fields are retained exactly. The
/// first loop-adapter implementation will select the wire-backed portions
/// under candidate activity and require agreement for the fields represented
/// by [`ThumbAgreement`].
#[derive(Clone)]
pub struct ThumbSnapshot<W, const FRAMES: usize> {
    /// General-purpose register wires, including `r13`'s mirrored SP word.
    pub regs: [[W; 32]; REG_COUNT],
    /// Per-register known concrete values.
    pub constants: [Option<u32>; REG_COUNT],
    /// Per-register stack-relative offset metadata.
    pub offsets: [Option<i32>; REG_COUNT],
    /// Lazy symbolic NZCVQ flags.
    pub flags: [Flag<W>; 5],
    /// Normalized even Thumb fetch PC.
    pub pc: u32,
    /// Concrete architectural stack pointer.
    pub sp: u32,
    /// Entry stack pointer required at an exit boundary.
    pub stack_top: u32,
    /// Private return-stack depth.
    pub rsp: usize,
    /// Private return addresses through `rsp`.
    pub rstack: [u32; FRAMES],
    /// Thumb predication state.
    pub itstate: u8,
    /// Virtual TrustZone-M state.
    pub security_state: SecurityState,
}

impl<W: Clone, const FRAMES: usize> ThumbSnapshot<W, FRAMES> {
    /// Capture a machine state using caller-owned fixed return-frame capacity.
    pub fn capture<E>(machine: &Machine<'_, W, E>) -> Result<Self, ThumbSnapshotError> {
        if machine.rsp > FRAMES || machine.rsp > machine.rstack.len() {
            return Err(ThumbSnapshotError::ReturnStackCapacity);
        }
        let mut rstack = [const { 0 }; FRAMES];
        rstack[..machine.rsp].copy_from_slice(&machine.rstack[..machine.rsp]);
        Ok(Self {
            regs: machine.regs.clone(),
            constants: *machine.constants,
            offsets: machine.offsets,
            flags: machine.flags.clone(),
            pc: machine.pc,
            sp: machine.sp,
            stack_top: machine.stack_top,
            rsp: machine.rsp,
            rstack,
            itstate: machine.itstate,
            security_state: machine.security_state,
        })
    }

    /// Restore this state into a machine using the same fixed return-frame
    /// capacity check as [`Self::capture`].
    pub fn restore<E>(&self, machine: &mut Machine<'_, W, E>) -> Result<(), ThumbSnapshotError> {
        if self.rsp > FRAMES || self.rsp > machine.rstack.len() {
            return Err(ThumbSnapshotError::ReturnStackCapacity);
        }
        *machine.regs = self.regs.clone();
        *machine.constants = self.constants;
        machine.offsets = self.offsets;
        machine.flags = self.flags.clone();
        machine.pc = self.pc;
        machine.sp = self.sp;
        machine.stack_top = self.stack_top;
        machine.rsp = self.rsp;
        machine.rstack[..self.rsp].copy_from_slice(&self.rstack[..self.rsp]);
        machine.itstate = self.itstate;
        machine.security_state = self.security_state;
        Ok(())
    }

    /// The concrete-only state that must agree before symbolic data folding.
    pub fn agreement(&self) -> ThumbAgreement<FRAMES> {
        ThumbAgreement {
            sp: self.sp,
            rstack_depth: self.rsp,
            rstack: self.rstack,
            itstate: self.itstate,
            security_state: self.security_state,
        }
    }
}

/// Select wire-backed state under `active`, leaving concrete metadata to the
/// explicit agreement check.
///
/// This is the same gate form as predicated storage writes:
/// `old ^ (active & (new ^ old))`.
pub fn select_wire<C, W>(context: &mut C, active: W, value: W, old: W) -> Result<W, C::Error>
where
    C: ContextWithErtOps<bool, Wrapped = W> + ?Sized,
    W: Clone,
{
    let difference = context.bitxor(value, old.clone())?;
    let gated = context.bitand(active, difference)?;
    context.bitxor(old, gated)
}

/// Fold the wire-backed register file from `candidate` into `accumulator`
/// under `active`.
///
/// The caller must first call [`merge_agreement`] for both snapshots. Concrete
/// Constants and stack offsets survive only where every folded body agrees;
/// divergent metadata becomes unknown. Lazy NZCVQ is retained in the snapshot
/// for now and is folded through the Arm materialization seam separately.
pub fn fold_registers<C, W, const FRAMES: usize>(
    context: &mut C,
    active: W,
    candidate: &ThumbSnapshot<W, FRAMES>,
    accumulator: &mut ThumbSnapshot<W, FRAMES>,
) -> Result<(), C::Error>
where
    C: ContextWithErtOps<bool, Wrapped = W> + ?Sized,
    W: Clone,
{
    for register in 0..REG_COUNT {
        for bit in 0..32 {
            accumulator.regs[register][bit] = select_wire(
                context,
                active.clone(),
                candidate.regs[register][bit].clone(),
                accumulator.regs[register][bit].clone(),
            )?;
        }
        if candidate.constants[register] != accumulator.constants[register] {
            accumulator.constants[register] = None;
        }
        if candidate.offsets[register] != accumulator.offsets[register] {
            accumulator.offsets[register] = None;
        }
    }
    Ok(())
}

/// Materialize and fold the five architectural NZCVQ wires.
///
/// Lazy flag recipes are materialized by a temporary Arm machine before the
/// fold, so the adapter never guesses their circuit shape. Concrete-only flag
/// facts must agree; symbolic flag wires are selected under `active`.
pub fn fold_flags<C, W, const FRAMES: usize>(
    context: &mut C,
    active: W,
    candidate: &ThumbSnapshot<W, FRAMES>,
    accumulator: &mut ThumbSnapshot<W, FRAMES>,
) -> Result<(), C::Error>
where
    C: cirrus_ert_core::ContextWithErtOps<bool, Wrapped = W> + ?Sized,
    W: Clone,
{
    for index in 0..5 {
        if candidate.flags[index].value != accumulator.flags[index].value {
            continue;
        }
        let (candidate_wire, accumulator_wire) =
            match (&candidate.flags[index].wire, &accumulator.flags[index].wire) {
                (
                    cirrus_armv8m_ert::FlagWire::Direct(candidate_wire),
                    cirrus_armv8m_ert::FlagWire::Direct(accumulator_wire),
                ) => (candidate_wire.clone(), accumulator_wire.clone()),
                _ => continue,
            };
        accumulator.flags[index].wire = cirrus_armv8m_ert::FlagWire::Direct(select_wire(
            context,
            active.clone(),
            candidate_wire,
            accumulator_wire,
        )?);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThumbAgreement<const FRAMES: usize> {
    /// Concrete architectural stack pointer.
    pub sp: u32,
    /// Depth of the interpreter's private return stack.
    pub rstack_depth: usize,
    /// In-flight private return addresses above the boundary's base depth.
    pub rstack: [u32; FRAMES],
    /// Thumb ITSTATE after the body.
    pub itstate: u8,
    /// Virtual TrustZone-M state after the body.
    pub security_state: SecurityState,
}

/// A candidate state did not satisfy the safe first-cut merge contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThumbMergeError {
    /// Candidate states disagree on concrete-only architectural state.
    ConcreteStateDivergence,
}

/// Merge `candidate` into the prior agreement.
///
/// The first candidate establishes the agreement. Subsequent candidates must
/// compare equal, otherwise the adapter fails closed rather than treating a
/// host-only state component as a symbolic value.
pub fn merge_agreement<const FRAMES: usize>(
    agreement: &mut Option<ThumbAgreement<FRAMES>>,
    candidate: ThumbAgreement<FRAMES>,
) -> Result<(), ThumbMergeError> {
    match agreement {
        Some(previous) if *previous != candidate => Err(ThumbMergeError::ConcreteStateDivergence),
        Some(_) => Ok(()),
        slot @ None => {
            *slot = Some(candidate);
            Ok(())
        }
    }
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::{ThumbAgreement, ThumbMergeError, merge_agreement, select_wire};
    use cirrus_armv8m_ert::SecurityState;

    fn state() -> ThumbAgreement<2> {
        ThumbAgreement {
            sp: 64,
            rstack_depth: 1,
            rstack: [4, 0],
            itstate: 0,
            security_state: SecurityState::Secure,
        }
    }

    #[test]
    fn active_wire_selection_is_predicated() {
        assert_eq!(select_wire(&mut (), false, true, false), Ok(false));
        assert_eq!(select_wire(&mut (), true, true, false), Ok(true));
    }

    #[test]
    fn equal_concrete_state_merges() {
        let mut agreement = None;
        assert_eq!(merge_agreement(&mut agreement, state()), Ok(()));
        assert_eq!(merge_agreement(&mut agreement, state()), Ok(()));
    }

    #[test]
    fn trustzone_or_itstate_divergence_fails_closed() {
        let mut agreement = Some(state());
        let mut non_secure = state();
        non_secure.security_state = SecurityState::NonSecure;
        assert_eq!(
            merge_agreement(&mut agreement, non_secure),
            Err(ThumbMergeError::ConcreteStateDivergence)
        );
        let mut it = state();
        it.itstate = 0x18;
        assert_eq!(
            merge_agreement(&mut agreement, it),
            Err(ThumbMergeError::ConcreteStateDivergence)
        );
    }
}
