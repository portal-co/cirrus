#![no_std]
#![warn(missing_docs)]

//! AArch64 adapter for the architecture-neutral ERT loop scheduler.
//!
//! The adapter keeps A64 decoding and state ownership in `cirrus-aarch64-ert`;
//! this crate provides boundary snapshots, concrete agreement, and predicated
//! folding for the shared candidate scheduler.

use cirrus_aarch64_ert::{Flow, RawMemory, State, step};
use cirrus_ert_loop_core::{
    CandidateDriver, CandidateTable, DriveError, drive_generation, predicated_value,
};

/// A symbolic A64 loop-boundary snapshot.
#[derive(Clone)]
pub struct Aarch64Snapshot<W> {
    /// Full facade state, including GPRs, SP, NZCV, and done.
    pub state: State<W>,
    /// Concrete fetch address at the boundary.
    pub pc: u64,
}

/// Capture a boundary snapshot.
pub fn capture<W: Clone>(state: &State<W>, pc: u64) -> Aarch64Snapshot<W> {
    Aarch64Snapshot {
        state: state.clone(),
        pc,
    }
}

/// Restore a boundary snapshot into `state`.
pub fn restore<W: Clone>(snapshot: &Aarch64Snapshot<W>, state: &mut State<W>) {
    state.clone_from(&snapshot.state);
}

/// Fold an executed candidate body into an accumulated state under `active`.
///
/// Concrete metadata survives only when both sides agree; disagreement makes
/// that fact unknown rather than choosing a host-side value.
pub fn fold_snapshot<C, W>(
    context: &mut C,
    active: &W,
    candidate: &Aarch64Snapshot<W>,
    accumulated: &mut Aarch64Snapshot<W>,
    zero: &W,
) -> Result<(), Aarch64FoldError<C::Error>>
where
    C: cirrus_core::ContextWithValue<bool, Wrapped = W>
        + cirrus_core::ContextWithBitAnd<bool, Wrapped = W>
        + cirrus_core::ContextWithBitOr<bool, Wrapped = W>
        + cirrus_core::ContextWithBitXor<bool, Wrapped = W>,
    W: Clone,
{
    if accumulated.state.sp != candidate.state.sp {
        return Err(Aarch64FoldError::ConcreteStateDivergence);
    }
    for register in 0..31 {
        fold_word(
            context,
            active,
            &candidate.state.regs[register],
            &mut accumulated.state.regs[register],
            zero,
        )?;
        if accumulated.state.constants[register] != candidate.state.constants[register] {
            accumulated.state.constants[register] = None;
        }
    }
    fold_word(
        context,
        active,
        &candidate.state.sp_word,
        &mut accumulated.state.sp_word,
        zero,
    )?;
    for flag in 0..4 {
        accumulated.state.nzcv[flag] = predicated_value(
            context,
            active.clone(),
            candidate.state.nzcv[flag].clone(),
            accumulated.state.nzcv[flag].clone(),
        )
        .map_err(Aarch64FoldError::Context)?;
        if accumulated.state.nzcv_constants[flag] != candidate.state.nzcv_constants[flag] {
            accumulated.state.nzcv_constants[flag] = None;
        }
    }
    accumulated.state.done = predicated_value(
        context,
        active.clone(),
        candidate.state.done.clone(),
        accumulated.state.done.clone(),
    )
    .map_err(Aarch64FoldError::Context)?;
    Ok(())
}

fn fold_word<C, W>(
    context: &mut C,
    active: &W,
    candidate: &[W; 64],
    accumulated: &mut [W; 64],
    _zero: &W,
) -> Result<(), Aarch64FoldError<C::Error>>
where
    C: cirrus_core::ContextWithValue<bool, Wrapped = W>
        + cirrus_core::ContextWithBitAnd<bool, Wrapped = W>
        + cirrus_core::ContextWithBitOr<bool, Wrapped = W>
        + cirrus_core::ContextWithBitXor<bool, Wrapped = W>,
    W: Clone,
{
    for (old, new) in accumulated.iter_mut().zip(candidate) {
        *old = predicated_value(context, active.clone(), new.clone(), old.clone())
            .map_err(Aarch64FoldError::Context)?;
    }
    Ok(())
}

/// Execute one candidate generation through the shared fixed-capacity
/// scheduler.
///
/// Each candidate body starts from `snapshot` and stops at the next symbolic
/// control-flow result or SVC exit. Concrete successors are appended in the
/// facade's stable order. Exited bodies leave no successor and predicate the
/// accumulated done wire.
pub fn run_generation<C, W>(
    context: &mut C,
    table: &mut CandidateTable<'_, u64>,
    snapshot: &Aarch64Snapshot<W>,
    virtual_ip: &[W; 64],
    memory: RawMemory<'_>,
    zero: &W,
    one: &W,
) -> Result<(Aarch64Snapshot<W>, [W; 64], W, bool), DriveError<Aarch64DriveError<C::Error>>>
where
    C: cirrus_core::ContextWithValue<bool, Wrapped = W>
        + cirrus_core::ContextWithBitAnd<bool, Wrapped = W>
        + cirrus_core::ContextWithBitOr<bool, Wrapped = W>
        + cirrus_core::ContextWithBitXor<bool, Wrapped = W>,
    W: WireValue + Clone,
{
    let mut driver = Aarch64GenerationDriver {
        context,
        initial: snapshot,
        accumulated: None,
        memory,
        zero,
        one,
        virtual_ip: virtual_ip.clone(),
        next_virtual_ip: virtual_ip.clone(),
        done: zero.clone(),
        exited: false,
    };
    drive_generation(table, &mut driver)?;
    Ok((
        driver
            .accumulated
            .ok_or(DriveError::Driver(Aarch64DriveError::NoSurvivingCandidates))?,
        driver.next_virtual_ip,
        driver.done,
        driver.exited,
    ))
}

struct Aarch64GenerationDriver<'a, C, W> {
    context: &'a mut C,
    initial: &'a Aarch64Snapshot<W>,
    accumulated: Option<Aarch64Snapshot<W>>,
    memory: RawMemory<'a>,
    zero: &'a W,
    one: &'a W,
    virtual_ip: [W; 64],
    next_virtual_ip: [W; 64],
    done: W,
    exited: bool,
}

impl<C, W> CandidateDriver<u64> for Aarch64GenerationDriver<'_, C, W>
where
    C: cirrus_core::ContextWithValue<bool, Wrapped = W>
        + cirrus_core::ContextWithBitAnd<bool, Wrapped = W>
        + cirrus_core::ContextWithBitOr<bool, Wrapped = W>
        + cirrus_core::ContextWithBitXor<bool, Wrapped = W>,
    W: WireValue + Clone,
{
    type Error = Aarch64DriveError<C::Error>;

    fn execute(
        &mut self,
        candidate: u64,
        successors: &mut dyn FnMut(u64) -> Result<(), DriveError<Self::Error>>,
    ) -> Result<(), DriveError<Self::Error>> {
        let mut state = self.initial.state.clone();
        let raw = u32::from_le_bytes(
            self.memory
                .read64::<4>(candidate)
                .ok_or(Aarch64DriveError::Memory(candidate))
                .map_err(DriveError::Driver)?,
        );
        let flow = step(
            self.context,
            &mut state,
            candidate,
            raw,
            self.zero,
            self.one,
        )
        .map_err(Aarch64DriveError::Decode)
        .map_err(DriveError::Driver)?;
        let mut snapshot = Aarch64Snapshot {
            state,
            pc: candidate,
        };
        let active = cirrus_ert_core::compare_word(
            self.context,
            &self.virtual_ip,
            &word64(candidate, self.zero, self.one),
            cirrus_ert_core::ComparePredicate::Eq,
            self.one,
        )
        .map_err(Aarch64DriveError::Context)
        .map_err(DriveError::Driver)?;
        match flow {
            Flow::Next(next_word) => {
                let next = Aarch64Snapshot::<W>::concrete_next(&next_word)
                    .ok_or(Aarch64DriveError::SymbolicNextPc)
                    .map_err(DriveError::Driver)?;
                snapshot.pc = next;
                if let Some(accumulated) = &mut self.accumulated {
                    fold_snapshot(self.context, &active, &snapshot, accumulated, self.zero)
                        .map_err(Aarch64DriveError::Fold)
                        .map_err(DriveError::Driver)?;
                } else {
                    self.accumulated = Some(snapshot);
                }
                for bit in 0..64 {
                    self.next_virtual_ip[bit] = predicated_value(
                        self.context,
                        active.clone(),
                        next_word[bit].clone(),
                        self.next_virtual_ip[bit].clone(),
                    )
                    .map_err(Aarch64DriveError::Context)
                    .map_err(DriveError::Driver)?;
                }
                successors(next)?;
            }
            Flow::Exit => {
                self.exited = true;
                self.done =
                    predicated_value(self.context, active, self.one.clone(), self.done.clone())
                        .map_err(Aarch64DriveError::Context)
                        .map_err(DriveError::Driver)?;
            }
        }
        Ok(())
    }
}

/// A Boolean wire that may carry a host-known concrete value.
pub trait WireValue {
    /// Return the host-known Boolean value, if this wire is concrete.
    fn concrete(self) -> Option<bool>;
}

fn word64<W: Clone>(value: u64, zero: &W, one: &W) -> [W; 64] {
    core::array::from_fn(|bit| {
        if (value >> bit) & 1 == 0 {
            zero.clone()
        } else {
            one.clone()
        }
    })
}

impl WireValue for bool {
    fn concrete(self) -> Option<bool> {
        Some(self)
    }
}

impl<W: WireValue + Clone> Aarch64Snapshot<W> {
    fn concrete_next(word: &[W; 64]) -> Option<u64> {
        let mut value = 0u64;
        for (bit, wire) in word.iter().enumerate() {
            value |= u64::from(wire.clone().concrete()?) << bit;
        }
        Some(value)
    }
}

/// A candidate body could not execute or fold under A64 loop rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Aarch64DriveError<E> {
    /// Instruction decode/execution failed.
    Decode(cirrus_aarch64_ert::DecodeError),
    /// The candidate instruction was unmapped.
    Memory(u64),
    /// The facade produced a symbolic next-PC outside a higher-level dispatch
    /// mechanism; the first adapter cut is concrete-control only.
    SymbolicNextPc,
    /// A Boolean folding/dispatch gate failed.
    Context(E),
    /// Folding candidate state failed.
    Fold(Aarch64FoldError<E>),
    /// Every candidate in the generation exited; there is no live state to fold.
    NoSurvivingCandidates,
}

/// A fold failed, either because a context gate failed or concrete A64 state
/// diverged across candidates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Aarch64FoldError<E> {
    /// The Boolean context rejected a folding gate.
    Context(E),
    /// Concrete SP diverged between live candidate bodies.
    ConcreteStateDivergence,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folding_merges_symbolic_bits_and_forgets_divergent_metadata() {
        let mut accumulated = capture(
            &cirrus_aarch64_ert::initial_state_with_arguments(false, &true, 64, &[Some(1)])
                .unwrap(),
            0,
        );
        let mut candidate = accumulated.clone();
        candidate.state.regs[0][0] = true;
        candidate.state.constants[0] = Some(3);
        accumulated.state.constants[0] = Some(1);
        fold_snapshot(&mut (), &true, &candidate, &mut accumulated, &false).unwrap();
        assert_eq!(accumulated.state.regs[0][0], true);
        assert_eq!(accumulated.state.constants[0], None);
        let mut inactive = accumulated.clone();
        inactive.state.constants[1] = Some(7);
        fold_snapshot(&mut (), &false, &inactive, &mut accumulated, &false).unwrap();
        assert_eq!(accumulated.state.regs[0][0], true);
        assert_eq!(accumulated.state.constants[0], None);
    }

    #[test]
    fn generation_runs_concrete_bodies_and_collects_successors() {
        let bytes = [0x01, 0x00, 0x00, 0x14]; // b +4
        let memory = RawMemory::from_slice(&bytes);
        let snapshot = capture(&cirrus_aarch64_ert::initial_state(false), 0);
        let mut entries = [0; 4];
        let mut table = CandidateTable::new(&mut entries, 0).unwrap();
        let (folded, next_vip, done, exited) = run_generation(
            &mut (),
            &mut table,
            &snapshot,
            &word64(0, &false, &true),
            memory,
            &false,
            &true,
        )
        .unwrap();
        assert_eq!(folded.pc, 4);
        assert_eq!(next_vip, word64(4, &false, &true));
        assert!(!done);
        assert!(!exited);
        assert_eq!(table.current(), &[4]);
    }

    #[test]
    fn symbolic_compare_branch_selects_virtual_ip_and_successors() {
        let bytes = 0xb500_0041u32.to_le_bytes(); // cbnz x1, +8
        let memory = RawMemory::from_slice(&bytes);
        let mut state = cirrus_aarch64_ert::initial_state(false);
        state.regs[1][0] = true;
        let snapshot = capture(&state, 0);
        let mut entries = [0; 8];
        let mut table = CandidateTable::new(&mut entries, 0).unwrap();
        let (folded, next_vip, done, exited) = run_generation(
            &mut (),
            &mut table,
            &snapshot,
            &word64(0, &false, &true),
            memory,
            &false,
            &true,
        )
        .unwrap();
        assert_eq!(folded.pc, 8);
        assert_eq!(next_vip, word64(8, &false, &true));
        assert!(!done);
        assert!(!exited);
        assert_eq!(table.current(), &[8]);
    }

    #[test]
    fn generation_fails_closed_when_all_candidates_exit() {
        let bytes = 0xd400_0001u32.to_le_bytes(); // svc #0
        let memory = RawMemory::from_slice(&bytes);
        let mut state = cirrus_aarch64_ert::initial_state(false);
        state.constants[0] = Some(u64::MAX);
        state.regs[0] = core::array::from_fn(|_| true);
        let snapshot = capture(&state, 0);
        let mut entries = [0; 4];
        let mut table = CandidateTable::new(&mut entries, 0).unwrap();
        assert!(matches!(
            run_generation(
                &mut (),
                &mut table,
                &snapshot,
                &word64(0, &false, &true),
                memory,
                &false,
                &true,
            ),
            Err(DriveError::Driver(Aarch64DriveError::NoSurvivingCandidates))
        ));
        assert!(table.is_empty());
    }

    #[test]
    fn exit_done_is_predicated_by_candidate_activity() {
        let bytes = 0xd400_0001u32.to_le_bytes(); // svc #0
        let memory = RawMemory::from_slice(&bytes);
        let mut state = cirrus_aarch64_ert::initial_state(false);
        state.constants[0] = Some(u64::MAX);
        state.regs[0] = core::array::from_fn(|_| true);
        let snapshot = capture(&state, 0);
        let mut entries = [0; 4];
        let mut table = CandidateTable::new(&mut entries, 0).unwrap();
        let result = run_generation(
            &mut (),
            &mut table,
            &snapshot,
            &word64(8, &false, &true),
            memory,
            &false,
            &true,
        );
        assert!(matches!(
            result,
            Err(DriveError::Driver(Aarch64DriveError::NoSurvivingCandidates))
        ));
    }

    #[test]
    fn folding_rejects_divergent_sp() {
        let mut accumulated = capture(
            &cirrus_aarch64_ert::initial_state_with_arguments(false, &true, 64, &[]).unwrap(),
            0,
        );
        let mut candidate = accumulated.clone();
        candidate.state.sp = Some(128);
        assert_eq!(
            fold_snapshot(&mut (), &true, &candidate, &mut accumulated, &false),
            Err(Aarch64FoldError::ConcreteStateDivergence)
        );
    }
}
