#![no_std]
#![warn(missing_docs)]

//! Looped-circuit emulator for the cirrus ERT.
//!
//! This crate shares the ERT's RV32/RV64 instruction semantics — the same
//! data-path handlers, the same symbolic-word circuits — but relaxes the
//! concrete-control-flow rule: when execution meets a branch whose condition
//! is symbolic, the emulator does not fail and does not unroll. It closes the
//! current step, makes the branch outcome a `select_word` on a **virtual
//! IP**, and transfers control by looping. A loop with a secret trip count
//! emits its body once per executed iteration, while the *driver* stays
//! `O(regions seen)`.
//!
//! Paths are explored **live**: instructions decode straight from
//! [`RawMemory`] exactly like the single-pass interpreter, and symbolic
//! operations are emitted into whatever [`ContextWithRvOps`] context the
//! caller drives — no allocation, no precomputed program representation.
//! Everything lives in caller-owned buffers: the register file, the return
//! stack, and a fixed-capacity candidate table (`&mut [u64]`) naming the
//! branch-target PCs the virtual IP may currently point at.
//!
//! ## The step contract
//!
//! Each [`LoopedMachine::step`] executes the straight-line code reachable
//! from every live candidate until a control-flow point:
//!
//! - one live candidate: the body executes directly on the live state —
//!   fallthrough is literal fallthrough, zero dispatch gates;
//! - several live candidates: each body executes on a snapshot of the shared
//!   state, its stack writes predicated by the candidate's activity wire
//!   (`write(old ^ (active & (new ^ old)))` — a no-op when inactive), and the
//!   resulting register files fold back through one select per register.
//!   Concrete stack pointers and return-stack depths must agree across
//!   continued bodies; divergence fails closed.
//!
//! Terminators are specialized: an unconditional target is baked into the
//! virtual IP as a constant (zero gates); a symbolic conditional branch emits
//! exactly one select; a declared bounded indirect `JALR`
//! ([`IndirectTargets`]) emits a constant-mux over its declared target set.
//! The accumulated `done` wire ([`LoopedMachine::done_wire`]) is the OR of
//! every exited path's activity; hosts with a concrete context read it to
//! decide termination, mirroring the `run_step_loop` contract of the Volar
//! one-step circuits.
//!
//! ## What stays fail-closed
//!
//! Symbolic `sp`-relative forms and symbolic non-stack addresses are rejected
//! exactly as in the single-pass interpreter (the virtual stack is the only
//! symbolic memory). Undeclared indirect targets (without a resolving
//! [`RvHandler::call_hook`]), candidate-table overflow, divergent concrete
//! stack state at a fold, unbalanced stack at exit, and unsupported
//! instruction encodings all report [`ErtError::Unexpected`]. Compressed
//! encodings of the supported subset are accepted.

use core::array;

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithStorage, ContextWithValue,
    HasError, StorageAddressBit,
};
use cirrus_ert::{
    CallAction, CallEvent, EcallOutcome, ErtError, RawMemory, RvHandler, handlers,
    machine::{ABI_REGS, Machine, RstackWord, Runtime},
};
use cirrus_ert_core::{ComparePredicate, compare_word, select_word};
use cirrus_ert_loop_core::CandidateTable;
use rv_asm::{Imm, Inst, Reg};

/// The declared, exhaustive target set of one bounded indirect `JALR`.
///
/// When the emulator meets a `JALR` (not the conventional return form) whose
/// base register is symbolic, it looks the instruction's address up in the
/// caller's declarations. A hit emits a constant-mux over `targets` — the
/// virtual IP's next candidates — instead of failing closed. The declaration
/// must be exhaustive: the mux's last leaf is the default, so an undeclared
/// reachable target would silently resolve to it. Targets are the base
/// addresses *before* the instruction's immediate offset is applied.
#[derive(Clone, Copy, Debug)]
pub struct IndirectTargets<'a> {
    /// The `JALR` instruction's address.
    pub pc: u64,
    /// Every base value the instruction may resolve to.
    pub targets: &'a [u64],
}

/// One step of the emulator's progress.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Step {
    /// Live paths remain; call [`LoopedMachine::step`] again.
    Continue,
    /// Every path reached the exit `ECALL` (structurally known).
    Done,
}

/// Couples an ERT handler with its caller-owned storage for one looped
/// execution, adding per-candidate write predication to the storage seam.
struct LoopedRuntime<'a, H: ContextWithStorage<bool> + ?Sized> {
    handler: &'a mut H,
    storage: &'a mut H::Storage,
    zero: H::Wrapped,
    one: H::Wrapped,
    predicate: Option<H::Wrapped>,
}

impl<H> HasError for LoopedRuntime<'_, H>
where
    H: ContextWithStorage<bool> + ?Sized,
{
    type Error = H::Error;
}

impl<H> ContextWithValue<bool> for LoopedRuntime<'_, H>
where
    H: ContextWithStorage<bool> + ?Sized,
{
    type Wrapped = H::Wrapped;
}

impl<H> ContextWithBitAnd<bool> for LoopedRuntime<'_, H>
where
    H: ContextWithStorage<bool> + ContextWithBitAnd<bool> + ?Sized,
{
    fn bitand(&mut self, a: H::Wrapped, b: H::Wrapped) -> Result<H::Wrapped, H::Error> {
        self.handler.bitand(a, b)
    }

    fn bitand_assign(&mut self, a: &mut H::Wrapped, b: H::Wrapped) -> Result<(), H::Error> {
        self.handler.bitand_assign(a, b)
    }
}

impl<H> ContextWithBitOr<bool> for LoopedRuntime<'_, H>
where
    H: ContextWithStorage<bool> + ContextWithBitOr<bool> + ?Sized,
{
    fn bitor(&mut self, a: H::Wrapped, b: H::Wrapped) -> Result<H::Wrapped, H::Error> {
        self.handler.bitor(a, b)
    }

    fn bitor_assign(&mut self, a: &mut H::Wrapped, b: H::Wrapped) -> Result<(), H::Error> {
        self.handler.bitor_assign(a, b)
    }
}

impl<H> ContextWithBitXor<bool> for LoopedRuntime<'_, H>
where
    H: ContextWithStorage<bool> + ContextWithBitXor<bool> + ?Sized,
{
    fn bitxor(&mut self, a: H::Wrapped, b: H::Wrapped) -> Result<H::Wrapped, H::Error> {
        self.handler.bitxor(a, b)
    }

    fn bitxor_assign(&mut self, a: &mut H::Wrapped, b: H::Wrapped) -> Result<(), H::Error> {
        self.handler.bitxor_assign(a, b)
    }
}

impl<H, W: Clone, E: core::error::Error, const BITS: usize> Runtime<W, BITS>
    for LoopedRuntime<'_, H>
where
    H: RvHandler<bool, BITS, Wrapped = W, Error = E> + ?Sized,
{
    fn ecall(
        &mut self,
        regs: &mut [[W; BITS]],
        reg_consts: &mut [Option<u64>],
        offsets: &mut [Option<i64>],
        zero: &W,
        one: &W,
    ) -> Result<EcallOutcome, E> {
        self.handler.ecall(regs, reg_consts, offsets, zero, one)
    }

    fn call_hook(
        &mut self,
        event: CallEvent,
        regs: &mut [[W; BITS]],
        reg_consts: &mut [Option<u64>],
        offsets: &mut [Option<i64>],
        zero: &W,
        one: &W,
    ) -> Result<CallAction, E> {
        self.handler
            .call_hook(event, regs, reg_consts, offsets, zero, one)
    }

    fn storage_read_bit(&mut self, bit: usize) -> Result<W, E> {
        let address = self.storage_address(bit);
        self.handler.storage_read(self.storage, &address)
    }

    fn storage_write_bit(&mut self, bit: usize, value: W) -> Result<(), E> {
        let address = self.storage_address(bit);
        match &self.predicate {
            None => self.handler.storage_write(self.storage, &address, value),
            Some(predicate) => {
                // `write(old ^ (active & (new ^ old)))`: an inactive
                // candidate's store is a no-op, keeping folded stack state
                // sound when sibling bodies run on the shared storage.
                let old = self.handler.storage_read(self.storage, &address)?;
                let difference = self.handler.bitxor(value, old.clone())?;
                let gated = self.handler.bitand(predicate.clone(), difference)?;
                let selected = self.handler.bitxor(old, gated)?;
                self.handler.storage_write(self.storage, &address, selected)
            }
        }
    }

    #[cfg(feature = "early-exit-loops")]
    fn early_exit_loop_options(&self) -> cirrus_ert_core::EarlyExitLoopOptions {
        self.handler.early_exit_loop_options()
    }
}

impl<H> LoopedRuntime<'_, H>
where
    H: ContextWithStorage<bool> + ?Sized,
    H::Wrapped: Clone,
{
    fn storage_address(
        &self,
        value: usize,
    ) -> [StorageAddressBit<H::Wrapped>; usize::BITS as usize] {
        array::from_fn(|bit| {
            let known = (value >> bit) & 1 != 0;
            StorageAddressBit {
                wire: if known {
                    self.one.clone()
                } else {
                    self.zero.clone()
                },
                known: Some(known),
            }
        })
    }
}

/// What one executed body reached at its control-flow point.
enum BodyKind<W: Clone, const BITS: usize> {
    /// A symbolic conditional branch: the virtual IP becomes
    /// `select(condition, taken, fallthrough)`.
    Branch {
        /// The selected next virtual IP wires.
        next_vip: [W; BITS],
        /// The branch-taken target.
        taken: u64,
        /// The not-taken (fallthrough) target.
        fallthrough: u64,
    },
    /// A declared bounded indirect `JALR`: the virtual IP becomes a
    /// constant-mux over the declared target set.
    Indirect {
        /// The muxed next virtual IP wires.
        next_vip: [W; BITS],
        /// Index into the machine's [`IndirectTargets`] declaration slice.
        declaration: usize,
    },
    /// The body reached the exit `ECALL` with a balanced stack.
    Exited,
}

/// The deepest a body may be inside unreturned calls at a control-flow
/// point, keeping the fold's return-stack agreement check free of
/// allocation.
const MAX_INFLIGHT: usize = 8;

/// One body's full end state.
struct BodyOutcome<W: Clone, const BITS: usize> {
    /// Concrete stack pointer at the control-flow point.
    sp: u64,
    /// Concrete return-stack depth at the control-flow point.
    rsp: u64,
    /// Stack-offset metadata at the control-flow point.
    offs: [Option<i64>; 32],
    /// Return-stack contents pushed above the step's starting depth
    /// (in-flight call frames); compared across bodies at a fold.
    inflight: ([u64; MAX_INFLIGHT], usize),
    /// What the body reached.
    kind: BodyKind<W, BITS>,
}

/// Capture the body's in-flight return-stack contents into its outcome.
fn capture_inflight<W: Clone, E, const BITS: usize, R: RstackWord>(
    machine: &Machine<'_, W, E, BITS, R>,
    base_rsp: u64,
    outcome: &mut BodyOutcome<W, BITS>,
) -> Result<(), ErtError<E>> {
    let depth = machine.rsp.saturating_sub(base_rsp) as usize;
    if depth > MAX_INFLIGHT {
        return Err(ErtError::Unexpected);
    }
    let mut inflight = [0u64; MAX_INFLIGHT];
    for (slot, entry) in inflight[..depth].iter_mut().enumerate() {
        *entry = machine.rstack[base_rsp as usize + slot].into_u64();
    }
    outcome.inflight = (inflight, depth);
    Ok(())
}

/// A constant word in the caller's wire type.
fn word_from_constant<W: Clone, const BITS: usize>(value: u64, zero: &W, one: &W) -> [W; BITS] {
    let mask = if BITS == 64 {
        u64::MAX
    } else {
        (1u64 << BITS) - 1
    };
    let value = value & mask;
    array::from_fn(|bit| {
        if (value >> bit) & 1 == 0 {
            zero.clone()
        } else {
            one.clone()
        }
    })
}

/// Execute one body from `pc` until a control-flow point, intercepting
/// symbolic branches and declared indirect calls; everything else runs
/// through the shared ERT handlers.
#[allow(clippy::too_many_arguments)]
fn execute_body<'a, H, W, E, const BITS: usize, R>(
    runtime: &mut LoopedRuntime<'a, H>,
    mem: RawMemory<'_>,
    rstack: &mut [R],
    storage_bits: usize,
    stack_top: u64,
    rsp: u64,
    indirect: &[IndirectTargets<'_>],
    pc: u64,
    regs: &mut [[W; BITS]; 32],
    reg_consts: &mut [Option<u64>; 32],
    offs: &mut [Option<i64>; 32],
    sp: u64,
    predicate: Option<W>,
    #[cfg(feature = "precompute")] program: Option<&LoopedProgram>,
) -> Result<BodyOutcome<W, BITS>, ErtError<E>>
where
    H: RvHandler<bool, BITS, Wrapped = W, Error = E> + ?Sized,
    W: Clone,
    E: core::error::Error,
    R: RstackWord,
{
    runtime.predicate = predicate;
    let zero = runtime.zero.clone();
    let one = runtime.one.clone();
    let mut machine = Machine::new(
        runtime,
        mem,
        rstack,
        storage_bits,
        pc,
        regs,
        reg_consts,
        zero,
        one,
        sp,
    );
    machine.offs = *offs;
    machine.rsp = rsp;
    machine.stack_top = stack_top;
    let base_rsp = rsp;
    let outcome = loop {
        let (instruction, len) = machine.decode()?;
        machine.inst_len = len;
        machine.reset_fixed_registers();
        if let Some(mut outcome) = intercept(
            &mut machine,
            instruction,
            indirect,
            #[cfg(feature = "precompute")]
            program,
        )? {
            capture_inflight(&machine, base_rsp, &mut outcome)?;
            break outcome;
        }
        match handlers::execute(&mut machine, instruction)? {
            handlers::Flow::Next(next_pc) => machine.pc = next_pc,
            handlers::Flow::Exit => {
                let mut outcome = BodyOutcome {
                    sp: machine.sp,
                    rsp: machine.rsp,
                    offs: machine.offs,
                    inflight: ([0; MAX_INFLIGHT], 0),
                    kind: BodyKind::Exited,
                };
                capture_inflight(&machine, base_rsp, &mut outcome)?;
                break outcome;
            }
        }
    };
    runtime.predicate = None;
    Ok(outcome)
}

/// Intercept a decoded instruction when it is a symbolic control-flow point
/// the looped emulator handles itself. Returns `Ok(None)` to delegate to the
/// shared handlers.
fn intercept<W, E, const BITS: usize, R>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    instruction: Inst,
    indirect: &[IndirectTargets<'_>],
    #[cfg(feature = "precompute")] program: Option<&LoopedProgram>,
) -> Result<Option<BodyOutcome<W, BITS>>, ErtError<E>>
where
    W: Clone,
    E: core::error::Error,
    R: RstackWord,
{
    match instruction {
        Inst::Beq { offset, src1, src2 } => intercept_branch(
            machine,
            offset,
            src1,
            src2,
            ComparePredicate::Eq,
            #[cfg(feature = "precompute")]
            program,
        ),
        Inst::Bne { offset, src1, src2 } => intercept_branch(
            machine,
            offset,
            src1,
            src2,
            ComparePredicate::Ne,
            #[cfg(feature = "precompute")]
            program,
        ),
        Inst::Bgeu { offset, src1, src2 } => intercept_branch(
            machine,
            offset,
            src1,
            src2,
            ComparePredicate::GeU,
            #[cfg(feature = "precompute")]
            program,
        ),
        Inst::Bltu { offset, src1, src2 } => intercept_branch(
            machine,
            offset,
            src1,
            src2,
            ComparePredicate::LtU,
            #[cfg(feature = "precompute")]
            program,
        ),
        Inst::Bge { offset, src1, src2 } => intercept_branch(
            machine,
            offset,
            src1,
            src2,
            ComparePredicate::GeS,
            #[cfg(feature = "precompute")]
            program,
        ),
        Inst::Blt { offset, src1, src2 } => intercept_branch(
            machine,
            offset,
            src1,
            src2,
            ComparePredicate::LtS,
            #[cfg(feature = "precompute")]
            program,
        ),
        Inst::Jalr { offset, base, dest }
            if !(dest == Reg::ZERO && base == Reg::RA && offset == Imm::ZERO)
                && machine.reg_consts[base.0 as usize].is_none() =>
        {
            // A symbolic-target `JALR`. A declared target set emits a
            // constant-mux; otherwise the shared handlers consult the call
            // hook and keep the historical fail-closed behavior.
            let Some(declaration) = indirect.iter().position(|d| d.pc == machine.pc) else {
                return Ok(None);
            };
            #[cfg(feature = "precompute")]
            if let Some(program) = program {
                if !program.indirect_sites.contains(&machine.pc) {
                    return Err(ErtError::Unexpected);
                }
            }
            let targets = indirect[declaration].targets;
            if targets.is_empty() {
                return Err(ErtError::Unexpected);
            }
            // The call's link semantics still apply: the callee's
            // conventional return pops the frame this pushes.
            if dest != Reg::ZERO {
                *machine
                    .rstack
                    .get_mut(machine.rsp as usize)
                    .ok_or(ErtError::Unexpected)? = R::from_u64(machine.pc + machine.inst_len);
                machine.rsp += 1;
                machine.write_constant(dest, machine.pc + machine.inst_len);
            }
            let offset64 = i64::from(offset.as_i32());
            let mut next_vip = machine
                .word_from_constant(targets[targets.len() - 1].wrapping_add_signed(offset64) & !1);
            for target in targets[..targets.len() - 1].iter().rev() {
                let resolved = target.wrapping_add_signed(offset64) & !1;
                let resolved_word = machine.word_from_constant(resolved);
                let one = machine.one.clone();
                let active = compare_word(
                    &mut *machine.t,
                    &machine.regs[base.0 as usize],
                    &resolved_word,
                    ComparePredicate::Eq,
                    &one,
                )
                .map_err(ErtError::Emitted)?;
                let target_word = resolved_word;
                next_vip = select_word(&mut *machine.t, active, &target_word, &next_vip)
                    .map_err(ErtError::Emitted)?;
            }
            Ok(Some(BodyOutcome {
                sp: machine.sp,
                rsp: machine.rsp,
                offs: machine.offs,
                inflight: ([0; MAX_INFLIGHT], 0),
                kind: BodyKind::Indirect {
                    next_vip,
                    declaration,
                },
            }))
        }
        _ => Ok(None),
    }
}

/// Intercept a conditional branch whose condition is symbolic: emit the
/// virtual-IP select and close the step.
fn intercept_branch<W, E, const BITS: usize, R>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    offset: Imm,
    src1: Reg,
    src2: Reg,
    predicate: ComparePredicate,
    #[cfg(feature = "precompute")] program: Option<&LoopedProgram>,
) -> Result<Option<BodyOutcome<W, BITS>>, ErtError<E>>
where
    W: Clone,
    E: core::error::Error,
    R: RstackWord,
{
    if machine.reg_consts[src1.0 as usize].is_some()
        && machine.reg_consts[src2.0 as usize].is_some()
    {
        return Ok(None);
    }
    #[cfg(feature = "precompute")]
    if let Some(program) = program {
        // The precomputed boundary map must agree with the live walker.
        let Some(&(taken, fallthrough)) = program.boundaries.get(&machine.pc) else {
            return Err(ErtError::Unexpected);
        };
        if taken != machine.pc.wrapping_add_signed(i64::from(offset.as_i32()))
            || fallthrough != machine.pc + machine.inst_len
        {
            return Err(ErtError::Unexpected);
        }
    }
    let condition = compare_word(
        &mut *machine.t,
        &machine.regs[src1.0 as usize],
        &machine.regs[src2.0 as usize],
        predicate,
        &machine.one,
    )
    .map_err(ErtError::Emitted)?;
    let taken = machine.pc.wrapping_add_signed(i64::from(offset.as_i32()));
    let fallthrough = machine.pc + machine.inst_len;
    let taken_word = machine.word_from_constant(taken);
    let fallthrough_word = machine.word_from_constant(fallthrough);
    let next_vip = select_word(&mut *machine.t, condition, &taken_word, &fallthrough_word)
        .map_err(ErtError::Emitted)?;
    Ok(Some(BodyOutcome {
        sp: machine.sp,
        rsp: machine.rsp,
        offs: machine.offs,
        inflight: ([0; MAX_INFLIGHT], 0),
        kind: BodyKind::Branch {
            next_vip,
            taken,
            fallthrough,
        },
    }))
}

/// The looped-circuit emulator: an ERT machine plus a virtual IP, a `done`
/// wire, and a caller-sized candidate table.
///
/// Construct with [`LoopedMachine::new`], drive with
/// [`LoopedMachine::step`], and read results with
/// [`LoopedMachine::results`] once finished. The generic `W` is the Boolean
/// wire type of the caller's context (`bool` for plaintext execution, an
/// index for recording, a garbled label, ...).
pub struct LoopedMachine<'a, H, W, E, const BITS: usize, R>
where
    H: RvHandler<bool, BITS, Wrapped = W, Error = E> + ?Sized,
    W: Clone,
    E: core::error::Error,
    R: RstackWord,
{
    runtime: LoopedRuntime<'a, H>,
    mem: RawMemory<'a>,
    rstack: &'a mut [R],
    storage_bits: usize,
    regs: [[W; BITS]; 32],
    reg_consts: [Option<u64>; 32],
    offs: [Option<i64>; 32],
    sp: u64,
    stack_top: u64,
    rsp: u64,
    vip: [W; BITS],
    done: W,
    done_const: Option<bool>,
    candidates: CandidateTable<'a, u64>,
    indirect: &'a [IndirectTargets<'a>],
    #[cfg(feature = "precompute")]
    program: Option<&'a LoopedProgram>,
}

impl<'a, H, W, E, const BITS: usize, R> LoopedMachine<'a, H, W, E, BITS, R>
where
    H: RvHandler<bool, BITS, Wrapped = W, Error = E> + ?Sized,
    W: Clone,
    E: core::error::Error,
    R: RstackWord,
{
    /// Set up a looped execution with the RISC-V argument ABI: the first
    /// eight `args` go to `a0`..`a7`, further values to caller stack slots,
    /// exactly as in `ert64_func`. `candidates` is the fixed-capacity
    /// candidate table: it must hold at least twice the maximum number of
    /// simultaneously-live branch targets (two per step for ordinary
    /// branches), and overflow fails closed.
    #[allow(clippy::too_many_arguments)]
    pub fn new<const N: usize>(
        t: &'a mut H,
        storage: &'a mut H::Storage,
        storage_bits: usize,
        mem: RawMemory<'a>,
        rstack: &'a mut [R],
        pc: u64,
        args: [([W; BITS], Option<u64>); N],
        candidates: &'a mut [u64],
        indirect: &'a [IndirectTargets<'a>],
        zero: W,
        one: W,
    ) -> Result<Self, ErtError<E>> {
        if storage_bits % 8 != 0 || candidates.is_empty() {
            return Err(ErtError::Unexpected);
        }
        let stack_bytes = u64::try_from(storage_bits / 8).map_err(|_| ErtError::Unexpected)?;
        let extra_values = N.saturating_sub(ABI_REGS.len());
        let slot_bytes = BITS as u64 / 8;
        let stack_pointer = stack_bytes
            .checked_sub(extra_values as u64 * slot_bytes)
            .ok_or(ErtError::Unexpected)?;
        let mut runtime = LoopedRuntime {
            handler: t,
            storage,
            zero: zero.clone(),
            one: one.clone(),
            predicate: None,
        };
        let mut regs: [[W; BITS]; 32] = array::from_fn(|_| array::from_fn(|_| zero.clone()));
        let mut reg_consts = [None; 32];
        cirrus_ert::machine::write_abi_args(
            &mut runtime,
            &mut regs,
            &mut reg_consts,
            storage_bits,
            stack_pointer,
            args,
        )
        .map_err(ErtError::Emitted)?;
        let candidates = CandidateTable::new(candidates, pc).map_err(|_| ErtError::Unexpected)?;
        Ok(Self {
            runtime,
            mem,
            rstack,
            storage_bits,
            regs,
            reg_consts,
            offs: [const { None }; 32],
            sp: stack_pointer,
            stack_top: stack_pointer,
            rsp: 0,
            vip: word_from_constant::<W, BITS>(pc, &zero, &one),
            done: zero,
            done_const: Some(false),
            candidates,
            indirect,
            #[cfg(feature = "precompute")]
            program: None,
        })
    }

    /// Attach a precomputed boundary map ([`LoopedProgram::compile`]). The
    /// live walker re-derives everything it needs; the map only
    /// consistency-checks each symbolic control-flow point, failing closed
    /// on any drift. This is a cache, never a semantic change.
    #[cfg(feature = "precompute")]
    pub fn set_program(&mut self, program: &'a LoopedProgram) {
        self.program = Some(program);
    }

    /// The accumulated `done` wire: the OR of every exited path's activity.
    /// Hosts with a concrete context (e.g. `bool`) read it to decide
    /// termination; symbolic-context hosts replay the recorded step on a
    /// plaintext backend and check it there.
    pub fn done_wire(&self) -> &W {
        &self.done
    }

    /// The candidate table's live count.
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    /// The current register file (wires), for hosts reading state between
    /// steps.
    pub fn regs(&self) -> &[[W; BITS]; 32] {
        &self.regs
    }

    /// Overwrite a register's wires and concrete metadata (ABI setup beyond
    /// `a0`..`a7`, and test fixtures).
    pub fn set_register(&mut self, reg: Reg, wires: [W; BITS], constant: Option<u64>) {
        self.regs[reg.0 as usize] = wires;
        self.reg_consts[reg.0 as usize] = constant;
        self.offs[reg.0 as usize] = None;
    }

    /// Execute one step: every live candidate's body runs to its next
    /// control-flow point and the results fold. See the crate-level step
    /// contract.
    pub fn step(&mut self) -> Result<Step, ErtError<E>> {
        if self.done_const == Some(true) || self.candidates.len() == 0 {
            return Ok(Step::Done);
        }
        if self.candidates.len() == 1 {
            let pc = self.candidates.current()[0];
            let outcome = execute_body(
                &mut self.runtime,
                self.mem,
                &mut self.rstack[..],
                self.storage_bits,
                self.stack_top,
                self.rsp,
                self.indirect,
                pc,
                &mut self.regs,
                &mut self.reg_consts,
                &mut self.offs,
                self.sp,
                None,
                #[cfg(feature = "precompute")]
                self.program,
            )?;
            self.sp = outcome.sp;
            self.rsp = outcome.rsp;
            self.offs = outcome.offs;
            match outcome.kind {
                BodyKind::Branch {
                    next_vip,
                    taken,
                    fallthrough,
                } => {
                    self.vip = next_vip;
                    let successors = if taken == fallthrough {
                        [taken, taken]
                    } else {
                        [taken, fallthrough]
                    };
                    let successor_count = if taken == fallthrough { 1 } else { 2 };
                    self.candidates
                        .replace(&successors[..successor_count])
                        .map_err(|_| ErtError::Unexpected)?;
                    self.done_const = Some(false);
                }
                BodyKind::Indirect {
                    next_vip,
                    declaration,
                } => {
                    self.vip = next_vip;
                    let targets = self.indirect[declaration].targets;
                    self.candidates
                        .replace(targets)
                        .map_err(|_| ErtError::Unexpected)?;
                    self.done_const = Some(false);
                }
                BodyKind::Exited => {
                    self.done = self.runtime.one.clone();
                    self.done_const = Some(true);
                    self.candidates.clear();
                    return Ok(Step::Done);
                }
            }
            return Ok(Step::Continue);
        }

        // Several live candidates: each body runs on a snapshot with its
        // stack writes predicated, and the register files fold back through
        // one select per register. The next candidate set accumulates in the
        // table's tail (`count..`) and is compacted down at the end.
        let base_regs = self.regs.clone();
        let base_consts = self.reg_consts;
        let base_offs = self.offs;
        let base_sp = self.sp;
        let base_rsp = self.rsp;
        let base_vip = self.vip.clone();
        let count = self.candidates.len();
        let mut next_vip = base_vip.clone();
        let mut next_done = self.done.clone();
        let mut merged_sp: Option<u64> = None;
        let mut merged_rsp: Option<u64> = None;
        let mut merged_inflight: Option<([u64; MAX_INFLIGHT], usize)> = None;
        let mut any_exited = false;
        let mut any_continued = false;
        let mut new_count = 0usize;

        for index in 0..count {
            let candidate = self.candidates.current()[index];
            let candidate_word =
                word_from_constant::<W, BITS>(candidate, &self.runtime.zero, &self.runtime.one);
            let one = self.runtime.one.clone();
            let active = compare_word(
                &mut self.runtime,
                &base_vip,
                &candidate_word,
                ComparePredicate::Eq,
                &one,
            )
            .map_err(ErtError::Emitted)?;

            let mut body_regs = base_regs.clone();
            let mut body_consts = base_consts;
            let mut body_offs = base_offs;
            let outcome = execute_body(
                &mut self.runtime,
                self.mem,
                &mut self.rstack[..],
                self.storage_bits,
                self.stack_top,
                base_rsp,
                self.indirect,
                candidate,
                &mut body_regs,
                &mut body_consts,
                &mut body_offs,
                base_sp,
                Some(active.clone()),
                #[cfg(feature = "precompute")]
                self.program,
            )?;

            // Fold the register file: one select per register.
            for register in 0..32 {
                self.regs[register] = select_word(
                    &mut self.runtime,
                    active.clone(),
                    &body_regs[register],
                    &self.regs[register],
                )
                .map_err(ErtError::Emitted)?;
                if body_consts[register] != self.reg_consts[register] {
                    self.reg_consts[register] = None;
                }
                if body_offs[register] != self.offs[register] {
                    self.offs[register] = None;
                }
            }

            let body_done;
            match &outcome.kind {
                BodyKind::Branch {
                    next_vip: body_vip,
                    taken,
                    fallthrough,
                } => {
                    body_done = self.runtime.zero.clone();
                    next_vip = select_word(&mut self.runtime, active.clone(), body_vip, &next_vip)
                        .map_err(ErtError::Emitted)?;
                    any_continued = true;
                    new_count = self.append_candidate(count, new_count, *taken)?;
                    new_count = self.append_candidate(count, new_count, *fallthrough)?;
                }
                BodyKind::Indirect {
                    next_vip: body_vip,
                    declaration,
                } => {
                    body_done = self.runtime.zero.clone();
                    next_vip = select_word(&mut self.runtime, active.clone(), body_vip, &next_vip)
                        .map_err(ErtError::Emitted)?;
                    any_continued = true;
                    for target in self.indirect[*declaration].targets {
                        new_count = self.append_candidate(count, new_count, *target)?;
                    }
                }
                BodyKind::Exited => {
                    body_done = self.runtime.one.clone();
                    any_exited = true;
                }
            }

            // done' = select(active, body_done, done).
            let difference = self
                .runtime
                .bitxor(body_done, next_done.clone())
                .map_err(ErtError::Emitted)?;
            let gated = self
                .runtime
                .bitand(active.clone(), difference)
                .map_err(ErtError::Emitted)?;
            next_done = self
                .runtime
                .bitxor(next_done, gated)
                .map_err(ErtError::Emitted)?;

            // Concrete stack state must agree across continued bodies;
            // exited bodies end at the (uniform) stack top and do not fold.
            if !matches!(outcome.kind, BodyKind::Exited) {
                match merged_sp {
                    Some(sp) if sp != outcome.sp => return Err(ErtError::Unexpected),
                    _ => merged_sp = Some(outcome.sp),
                }
                match merged_rsp {
                    Some(rsp) if rsp != outcome.rsp => return Err(ErtError::Unexpected),
                    _ => merged_rsp = Some(outcome.rsp),
                }
                if let Some(previous) = merged_inflight {
                    if previous != outcome.inflight {
                        return Err(ErtError::Unexpected);
                    }
                } else {
                    merged_inflight = Some(outcome.inflight);
                }
            }
        }

        if any_continued {
            self.sp = merged_sp.ok_or(ErtError::Unexpected)?;
            self.rsp = merged_rsp.ok_or(ErtError::Unexpected)?;
        }
        self.vip = next_vip;
        self.done = next_done;
        // Compact the tail-accumulated candidates into the table prefix.
        self.candidates
            .finish_next(new_count)
            .map_err(|_| ErtError::Unexpected)?;
        self.done_const = if any_exited && !any_continued {
            Some(true)
        } else if !any_exited {
            Some(false)
        } else {
            None
        };
        Ok(Step::Continue)
    }

    /// Append `candidate` to the next-candidate set accumulating in the
    /// table's tail region (`base + count`), deduplicated. The tail must not
    /// overlap the live prefix `0..base`, so the table must hold at least
    /// twice the maximum live count.
    fn append_candidate(
        &mut self,
        base: usize,
        count: usize,
        candidate: u64,
    ) -> Result<usize, ErtError<E>> {
        if base != self.candidates.len() {
            return Err(ErtError::Unexpected);
        }
        self.candidates
            .append_next(count, candidate)
            .map_err(|_| ErtError::Unexpected)
    }

    /// Keep stepping until done (structurally or by the caller's wire
    /// predicate) or the step budget runs out. Returns `true` when finished;
    /// `false` means the budget was exhausted and the caller must decide.
    pub fn run(
        &mut self,
        max_steps: u64,
        mut is_done: impl FnMut(&W) -> bool,
    ) -> Result<bool, ErtError<E>> {
        for _ in 0..max_steps {
            self.step()?;
            if self.done_const == Some(true) || self.candidates.len() == 0 || is_done(&self.done) {
                // The host's `is_done` is the done wire faithfully read (the
                // concrete-context contract); it authorizes `results()`.
                self.done_const = Some(true);
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Read ABI results (`a0`..`a7`, then caller stack slots) after the run
    /// finished. Fails closed when execution has not completed.
    pub fn results<const M: usize>(
        &mut self,
    ) -> Result<[([W; BITS], Option<u64>); M], ErtError<E>> {
        if self.done_const != Some(true) {
            return Err(ErtError::Unexpected);
        }
        cirrus_ert::machine::read_abi_results(
            &mut self.runtime,
            &self.regs,
            &self.reg_consts,
            self.storage_bits,
            self.stack_top,
        )
        .map_err(ErtError::Emitted)
    }
}

/// Invoke a symbolic RV64 function under the looped-circuit emulator, using
/// the RISC-V argument and result ABI. The run continues until done or the
/// step budget runs out; an exhausted budget fails closed.
///
/// `is_done` reads the `done` wire on concrete contexts (`|wire| *wire` for
/// `bool`); pass `|_| false` on opaque contexts to rely on structural
/// termination only.
#[allow(clippy::too_many_arguments)]
pub fn looped_ert64_func<W: Clone, E: core::error::Error, const N: usize, const M: usize, H>(
    t: &mut H,
    storage: &mut H::Storage,
    storage_bits: usize,
    mem: RawMemory<'_>,
    rstack: &mut [u64],
    pc: u64,
    args: [([W; 64], Option<u64>); N],
    candidates: &mut [u64],
    indirect: &[IndirectTargets<'_>],
    zero: W,
    one: W,
    max_steps: u64,
    is_done: impl FnMut(&W) -> bool,
) -> Result<[([W; 64], Option<u64>); M], ErtError<E>>
where
    H: RvHandler<bool, 64, Wrapped = W, Error = E> + ?Sized,
{
    let mut machine = LoopedMachine::<H, W, E, 64, u64>::new(
        t,
        storage,
        storage_bits,
        mem,
        rstack,
        pc,
        args,
        candidates,
        indirect,
        zero,
        one,
    )?;
    if !machine.run(max_steps, is_done)? {
        return Err(ErtError::Unexpected);
    }
    machine.results()
}

/// The RV32 spelling of [`looped_ert64_func`]: 32-bit wires, `u32` concrete
/// metadata, and a `u32` return stack.
#[allow(clippy::too_many_arguments)]
pub fn looped_ert_func<W: Clone, E: core::error::Error, const N: usize, const M: usize, H>(
    t: &mut H,
    storage: &mut H::Storage,
    storage_bits: usize,
    mem: RawMemory<'_>,
    rstack: &mut [u32],
    pc: u32,
    args: [([W; 32], Option<u32>); N],
    candidates: &mut [u64],
    indirect: &[IndirectTargets<'_>],
    zero: W,
    one: W,
    max_steps: u64,
    is_done: impl FnMut(&W) -> bool,
) -> Result<[([W; 32], Option<u32>); M], ErtError<E>>
where
    H: RvHandler<bool, Wrapped = W, Error = E> + ?Sized,
{
    let args64: [([W; 32], Option<u64>); N] =
        args.map(|(word, constant)| (word, constant.map(u64::from)));
    let mut machine = LoopedMachine::<H, W, E, 32, u32>::new(
        t,
        storage,
        storage_bits,
        mem,
        rstack,
        u64::from(pc),
        args64,
        candidates,
        indirect,
        zero,
        one,
    )?;
    if !machine.run(max_steps, is_done)? {
        return Err(ErtError::Unexpected);
    }
    let results = machine.results::<M>()?;
    Ok(results.map(|(word, constant)| (word, constant.map(|value| value as u32))))
}

#[cfg(feature = "precompute")]
extern crate alloc;

/// A precomputed, validated map of a guest image's control-flow boundaries
/// (the `precompute` feature's cache). Execution through
/// [`LoopedMachine::set_program`] runs the identical live path — this is a
/// cache and consistency check, never a semantic change.
#[cfg(feature = "precompute")]
pub struct LoopedProgram {
    /// Every conditional-branch boundary: pc -> (taken, fallthrough).
    boundaries: alloc::collections::BTreeMap<u64, (u64, u64)>,
    /// Every indirect `JALR` site that required a declaration.
    indirect_sites: alloc::vec::Vec<u64>,
}

#[cfg(feature = "precompute")]
impl LoopedProgram {
    /// Walk the image from `entry`, treating every conditional branch as a
    /// boundary with both successors live, every direct jump/call as
    /// fallthrough-plus-target, and every non-return `JALR` as requiring a
    /// declaration. Stops at returns and the exit `ECALL`. Fails closed on
    /// unsupported encodings, undeclared indirects, or more than
    /// `max_boundaries` boundaries.
    pub fn compile(
        mem: &RawMemory<'_>,
        entry: u64,
        indirect: &[IndirectTargets<'_>],
        max_boundaries: usize,
    ) -> Result<Self, ErtError<core::convert::Infallible>> {
        use alloc::collections::{BTreeMap, BTreeSet};
        let mut boundaries = BTreeMap::new();
        let mut indirect_sites = alloc::vec::Vec::new();
        let mut visited = BTreeSet::new();
        let mut worklist = alloc::vec![entry];
        while let Some(start) = worklist.pop() {
            if !visited.insert(start) {
                continue;
            }
            let mut pc = start;
            loop {
                let half = u16::from_le_bytes(mem.read64::<2>(pc).ok_or(ErtError::Unexpected)?);
                let xlen = rv_asm::Xlen::Rv64;
                let (instruction, len) = if Inst::first_byte_is_compressed(half as u8) {
                    (
                        Inst::decode_compressed(half, xlen).map_err(ErtError::Decode)?,
                        2u64,
                    )
                } else {
                    let word = u32::from_le_bytes(mem.read64::<4>(pc).ok_or(ErtError::Unexpected)?);
                    (
                        Inst::decode(word, xlen)
                            .map(|(instruction, _)| instruction)
                            .map_err(ErtError::Decode)?,
                        4u64,
                    )
                };
                match instruction {
                    Inst::Beq { offset, .. }
                    | Inst::Bne { offset, .. }
                    | Inst::Bgeu { offset, .. }
                    | Inst::Bltu { offset, .. }
                    | Inst::Bge { offset, .. }
                    | Inst::Blt { offset, .. } => {
                        if boundaries.len() >= max_boundaries {
                            return Err(ErtError::Unexpected);
                        }
                        let taken = pc.wrapping_add_signed(i64::from(offset.as_i32()));
                        let fallthrough = pc + len;
                        boundaries.insert(pc, (taken, fallthrough));
                        worklist.push(taken);
                        worklist.push(fallthrough);
                        break;
                    }
                    Inst::Jal { offset, dest } => {
                        let target = pc.wrapping_add_signed(i64::from(offset.as_i32()));
                        if dest == Reg::ZERO {
                            worklist.push(target);
                            break;
                        }
                        // A call: explore the callee and the return landing.
                        worklist.push(target);
                        pc += len;
                    }
                    Inst::Jalr { offset, base, dest } => {
                        if dest == Reg::ZERO && base == Reg::RA && offset == Imm::ZERO {
                            break;
                        }
                        let Some(declaration) = indirect.iter().find(|d| d.pc == pc) else {
                            return Err(ErtError::Unexpected);
                        };
                        if declaration.targets.is_empty() {
                            return Err(ErtError::Unexpected);
                        }
                        indirect_sites.push(pc);
                        for target in declaration.targets {
                            worklist
                                .push(target.wrapping_add_signed(i64::from(offset.as_i32())) & !1);
                        }
                        if dest != Reg::ZERO {
                            // Also explore the return landing.
                            pc += len;
                            continue;
                        }
                        break;
                    }
                    Inst::Ecall => break,
                    _ => pc += len,
                }
            }
        }
        Ok(Self {
            boundaries,
            indirect_sites,
        })
    }

    /// The branch-boundary map this program validated.
    pub fn boundaries(&self) -> &alloc::collections::BTreeMap<u64, (u64, u64)> {
        &self.boundaries
    }
}

#[cfg(test)]
mod tests;
