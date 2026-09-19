#![no_std]
#![warn(missing_docs)]

//! AArch64 adapter for the architecture-neutral ERT loop scheduler.
//!
//! The adapter keeps A64 decoding and state ownership in `cirrus-aarch64-ert`;
//! this crate provides boundary snapshots, concrete agreement, and predicated
//! folding for the shared candidate scheduler.

use cirrus_aarch64_ert::{DecodeError, Flow, RawMemory, State, step as execute_instruction};
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

/// The result of one shared-scheduler AArch64 generation.
#[derive(Clone)]
pub struct Aarch64Step<W> {
    /// Folded architectural state for the next generation.
    pub snapshot: Aarch64Snapshot<W>,
    /// Symbolic virtual instruction pointer for the next generation.
    pub virtual_ip: [W; 64],
    /// Accumulated symbolic completion wire.
    pub done: W,
    /// Whether any represented body exited in this generation.
    pub exited: bool,
}

/// Build the initial A64 loop snapshot and virtual-IP word.
pub fn initial_step<W: Clone>(
    state: State<W>,
    entry: u64,
    zero: &W,
    one: &W,
) -> Result<Aarch64Step<W>, DecodeError> {
    if entry & 3 != 0 {
        return Err(DecodeError::Malformed(0));
    }
    Ok(Aarch64Step {
        snapshot: capture(&state, entry),
        virtual_ip: word64(entry, zero, one),
        done: zero.clone(),
        exited: false,
    })
}

/// Execute one no-alloc AArch64 loop generation through the shared scheduler.
///
/// The caller owns the candidate table. A prior `done` wire is carried into
/// the generation and predicated with exits from active candidate bodies.
pub fn step<C, W>(
    context: &mut C,
    table: &mut CandidateTable<'_, u64>,
    previous: Aarch64Step<W>,
    memory: RawMemory<'_>,
    zero: &W,
    one: &W,
) -> Result<Aarch64Step<W>, DriveError<Aarch64DriveError<C::Error>>>
where
    C: cirrus_core::ContextWithValue<bool, Wrapped = W>
        + cirrus_core::ContextWithBitAnd<bool, Wrapped = W>
        + cirrus_core::ContextWithBitOr<bool, Wrapped = W>
        + cirrus_core::ContextWithBitXor<bool, Wrapped = W>,
    W: WireValue + Clone,
{
    let (snapshot, virtual_ip, generation_done, exited) = run_generation(
        context,
        table,
        &previous.snapshot,
        &previous.virtual_ip,
        memory,
        zero,
        one,
    )?;
    let done = predicated_value(context, previous.done.clone(), one.clone(), generation_done)
        .map_err(Aarch64DriveError::Context)
        .map_err(DriveError::Driver)?;
    Ok(Aarch64Step {
        snapshot,
        virtual_ip,
        done,
        exited: exited || previous.exited,
    })
}

/// The boundary reached by an A64 candidate body.
#[derive(Clone)]
pub enum Aarch64Boundary<W> {
    /// The body stopped at a symbolic next-PC word.
    SymbolicBranch {
        /// Symbolic selected next-PC word.
        next: [W; 64],
        /// Concrete taken target.
        taken: u64,
        /// Concrete fallthrough target.
        fallthrough: u64,
    },
    /// The body reached the declared SVC exit.
    Exit,
}

/// Detect and resolve a symbolic A64 conditional-branch boundary.
///
/// The facade's decoded branch form remains private, so the adapter audits
/// the same raw CBZ/CBNZ and TBZ/TBNZ masks before executing the instruction.
fn symbolic_branch_boundary<C, W>(
    context: &mut C,
    state: &State<W>,
    pc: u64,
    raw: u32,
    zero: &W,
    one: &W,
) -> Result<Option<Aarch64Boundary<W>>, C::Error>
where
    C: cirrus_core::ContextWithValue<bool, Wrapped = W>
        + cirrus_core::ContextWithBitAnd<bool, Wrapped = W>
        + cirrus_core::ContextWithBitOr<bool, Wrapped = W>
        + cirrus_core::ContextWithBitXor<bool, Wrapped = W>,
    W: WireValue + Clone,
{
    let fallthrough = pc.wrapping_add(4);
    if raw & 0x7e00_0000 == 0x3400_0000 {
        let register = (raw & 31) as usize;
        let nonzero = raw & (1 << 24) != 0;
        if register == 31 || state.constants[register].is_some() {
            return Ok(None);
        }
        let mut wire = state.regs[register].clone();
        let zero_word = word64(0, zero, one);
        if raw & (1 << 31) == 0 {
            for bit in 32..64 {
                wire[bit] = zero.clone();
            }
        }
        let condition = cirrus_ert_core::compare_word(
            context,
            &wire,
            &zero_word,
            if nonzero {
                cirrus_ert_core::ComparePredicate::Ne
            } else {
                cirrus_ert_core::ComparePredicate::Eq
            },
            one,
        )?;
        let taken = pc.wrapping_add_signed(cirrus_aarch64_ert::compare_branch_offset(raw));
        let next = cirrus_ert_core::select_word(
            context,
            condition,
            &word64(taken, zero, one),
            &word64(fallthrough, zero, one),
        )?;
        return Ok(Some(Aarch64Boundary::SymbolicBranch {
            next,
            taken,
            fallthrough,
        }));
    }
    if raw & 0x7e00_0000 == 0x3600_0000 {
        let register = (raw & 31) as usize;
        let bit = (((raw >> 31) & 1) << 5 | ((raw >> 19) & 31)) as usize;
        let tested = if register == 31 {
            zero.clone()
        } else {
            state.regs[register][bit].clone()
        };
        if register == 31 || tested.clone().concrete().is_some() {
            return Ok(None);
        }
        let condition = if raw & (1 << 24) != 0 {
            tested
        } else {
            context.bitxor(tested, one.clone())?
        };
        let taken = pc.wrapping_add_signed(cirrus_aarch64_ert::test_branch_offset(raw));
        let next = cirrus_ert_core::select_word(
            context,
            condition,
            &word64(taken, zero, one),
            &word64(fallthrough, zero, one),
        )?;
        return Ok(Some(Aarch64Boundary::SymbolicBranch {
            next,
            taken,
            fallthrough,
        }));
    }
    Ok(None)
}

/// Execute a straight-line A64 candidate body until a symbolic branch or exit.
///
/// The adapter owns the body-runner distinction that the facade intentionally
/// does not expose: ordinary flow advances to a concrete next PC, while a
/// symbolic `Flow::Next` word stops the body. The symbolic branch currently
/// has the A64 conditional shape, so the boundary reports concrete taken and
/// fallthrough targets in stable order.
pub fn execute_body<C, W>(
    context: &mut C,
    state: &mut State<W>,
    entry: u64,
    memory: RawMemory<'_>,
    zero: &W,
    one: &W,
) -> Result<Aarch64Boundary<W>, Aarch64DriveError<C::Error>>
where
    C: cirrus_core::ContextWithValue<bool, Wrapped = W>
        + cirrus_core::ContextWithBitAnd<bool, Wrapped = W>
        + cirrus_core::ContextWithBitOr<bool, Wrapped = W>
        + cirrus_core::ContextWithBitXor<bool, Wrapped = W>,
    W: WireValue + Clone,
{
    let mut pc = entry;
    loop {
        let raw = u32::from_le_bytes(
            memory
                .read64::<4>(pc)
                .ok_or(Aarch64DriveError::Memory(pc))?,
        );
        if let Some(boundary) = symbolic_branch_boundary(context, state, pc, raw, zero, one)
            .map_err(Aarch64DriveError::Context)?
        {
            return Ok(boundary);
        }
        let flow = execute_instruction(context, state, pc, raw, zero, one)
            .map_err(Aarch64DriveError::Decode)?;
        match flow {
            Flow::Next(next_word) => {
                let Some(next) = Aarch64Snapshot::<W>::concrete_next(&next_word) else {
                    return Err(Aarch64DriveError::SymbolicNextPc);
                };
                pc = next;
                if raw & 0x7e00_0000 == 0x3600_0000 {
                    return Ok(Aarch64Boundary::SymbolicBranch {
                        next: next_word,
                        taken: pc.wrapping_add_signed(cirrus_aarch64_ert::test_branch_offset(raw)),
                        fallthrough: pc.wrapping_add(4),
                    });
                }
                if raw & 0x7e00_0000 == 0x3400_0000 {
                    return Ok(Aarch64Boundary::SymbolicBranch {
                        next: next_word,
                        taken: pc
                            .wrapping_add_signed(cirrus_aarch64_ert::compare_branch_offset(raw)),
                        fallthrough: pc.wrapping_add(4),
                    });
                }
                if raw & 0x7c00_0000 == 0x1400_0000 {
                    return Ok(Aarch64Boundary::SymbolicBranch {
                        next: next_word,
                        taken: next,
                        fallthrough: next,
                    });
                }
            }
            Flow::Exit => return Ok(Aarch64Boundary::Exit),
        }
    }
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
        let boundary = execute_body(
            self.context,
            &mut state,
            candidate,
            self.memory,
            self.zero,
            self.one,
        )
        .map_err(DriveError::Driver)?;
        let active = cirrus_ert_core::compare_word(
            self.context,
            &self.virtual_ip,
            &word64(candidate, self.zero, self.one),
            cirrus_ert_core::ComparePredicate::Eq,
            self.one,
        )
        .map_err(Aarch64DriveError::Context)
        .map_err(DriveError::Driver)?;
        let snapshot_pc = match &boundary {
            Aarch64Boundary::SymbolicBranch {
                taken, fallthrough, ..
            } => {
                if taken == fallthrough {
                    *taken
                } else {
                    candidate
                }
            }
            Aarch64Boundary::Exit => candidate,
        };
        let snapshot = Aarch64Snapshot {
            state,
            pc: snapshot_pc,
        };
        match boundary {
            Aarch64Boundary::SymbolicBranch {
                next: next_word,
                taken,
                fallthrough,
            } => {
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
                successors(taken)?;
                if fallthrough != taken {
                    successors(fallthrough)?;
                }
            }
            Aarch64Boundary::Exit => {
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
        // b +8; nop; nop (the branch body terminates at the symbolic boundary)
        let bytes = [
            0x02, 0x00, 0x00, 0x14, 0x1f, 0x20, 0x03, 0xd5, 0x1f, 0x20, 0x03, 0xd5,
        ];
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
        assert_eq!(folded.pc, 8);
        assert_eq!(next_vip, word64(8, &false, &true));
        assert!(!done);
        assert!(!exited);
        assert_eq!(table.current(), &[8]);
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
        assert_eq!(folded.pc, 0);
        assert_eq!(next_vip, word64(8, &false, &true));
        assert!(!done);
        assert!(!exited);
        assert_eq!(table.current(), &[8, 4]);
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
    fn symbolic_test_branch_advances_through_public_step_generations() {
        // tbnz x1, #0, +8; movz x0, #0; svc #0; nop; svc #0
        let code = [
            0x3728_0041u32,
            0x5280_0000,
            0xd400_0001,
            0xd503_201f,
            0xd400_0001,
        ];
        let mut state = cirrus_aarch64_ert::initial_state(false);
        state.regs[1][0] = true;
        let mut initial = initial_step(state, 0, &false, &true).unwrap();
        initial.snapshot.state.constants[0] = Some(u64::MAX);
        initial.snapshot.state.regs[0] = core::array::from_fn(|_| true);
        // The plaintext host treats symbolic bit 0 of x1 as true, so the
        // generation queues candidates 12 and 8; the 12 body exits through the
        // second SVC, then candidate 8 exits through the first SVC.
        let mut bytes = [0u8; 20];
        for (index, word) in code.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let mut backing = [0u64; 8];
        let mut candidates = CandidateTable::new(&mut backing, 0).unwrap();
        let first = step(
            &mut (),
            &mut candidates,
            initial,
            RawMemory::from_slice(&bytes),
            &false,
            &true,
        )
        .unwrap();
        assert_eq!(candidates.current(), &[12, 8]);
        assert_eq!(first.virtual_ip, word64(4, &false, &true));
        assert!(!first.done);
        assert!(!first.exited);

        let mut first = first;
        first.snapshot.state.constants[0] = Some(u64::MAX);
        first.snapshot.state.regs[0] = core::array::from_fn(|_| true);
        let second = step(
            &mut (),
            &mut candidates,
            first,
            RawMemory::from_slice(&bytes),
            &false,
            &true,
        );
        assert!(matches!(
            second,
            Err(DriveError::Driver(Aarch64DriveError::NoSurvivingCandidates))
        ));
        assert!(candidates.is_empty());
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
    fn public_step_carries_done_and_virtual_ip_across_generations() {
        let bytes = 0xd400_0001u32.to_le_bytes(); // svc #0
        let memory = RawMemory::from_slice(&bytes);
        let mut state = cirrus_aarch64_ert::initial_state(false);
        state.constants[0] = Some(u64::MAX);
        state.regs[0] = core::array::from_fn(|_| true);
        let initial = initial_step(state, 0, &false, &true).unwrap();
        let mut entries = [0; 4];
        let mut table = CandidateTable::new(&mut entries, 0).unwrap();
        let stepped = step(&mut (), &mut table, initial, memory, &false, &true);
        assert!(matches!(
            stepped,
            Err(DriveError::Driver(Aarch64DriveError::NoSurvivingCandidates))
        ));
        assert!(table.is_empty());
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
