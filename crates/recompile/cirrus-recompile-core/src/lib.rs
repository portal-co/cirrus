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

    /// Reabstract profitable adjacent repetitions into table-driven loops.
    ///
    /// The returned artifact writes the exact same absolute scratch-buffer
    /// slots as this raw trace.  That lets a caller retain `Program` as the
    /// portable circuit interchange format while a backend chooses a smaller
    /// executable representation.
    pub fn prepare(&self, options: &OptimizationOptions) -> PreparedProgram {
        PreparedProgram::from_program(self, options)
    }
}

/// Tuning knobs for [`Program::prepare`].
///
/// The cost model is deliberately about static artifact size and compilation
/// work, not an unproven runtime-speed estimate.  A caller can turn the pass
/// off to inspect or debug the original trace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OptimizationOptions {
    /// Whether reabstraction is enabled at all.
    pub enabled: bool,
    /// The smallest number of adjacent occurrences eligible for a loop.
    pub min_repetitions: usize,
    /// The minimum estimated static-byte reduction, including index tables.
    pub min_static_savings_bytes: usize,
    /// The largest operation template the deterministic search considers.
    pub max_template_ops: usize,
}

impl Default for OptimizationOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            min_repetitions: 4,
            min_static_savings_bytes: 64,
            max_template_ops: 128,
        }
    }
}

/// A [`Program`] arranged for execution by an optimizing backend.
///
/// It deliberately keeps the raw program's slot numbering, input order, and
/// output order.  The only change is how operations are grouped for execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedProgram {
    /// Scratch-buffer width, identical to the source [`Program::len`].
    pub slots: usize,
    /// Input slots in the source program's deterministic order.
    pub inputs: Vec<Idx>,
    /// Output slots in the source program's deterministic order.
    pub outputs: Vec<Idx>,
    /// Flat operation groups and table-driven repeated groups in program order.
    pub steps: Vec<PreparedStep>,
    /// Estimated bytes for the source's fully unrolled call sequence.
    pub estimated_unrolled_bytes: usize,
    /// Estimated bytes for this prepared sequence, including loop tables.
    pub estimated_prepared_bytes: usize,
}

impl PreparedProgram {
    fn from_program(program: &Program, options: &OptimizationOptions) -> Self {
        let mut selected = select_candidates(program, options);
        selected.sort_by_key(|candidate| candidate.start);

        let mut steps = Vec::new();
        let mut cursor = 0usize;
        for candidate in selected {
            if cursor < candidate.start {
                steps.push(PreparedStep::Flat(flat_ops(
                    program,
                    cursor,
                    candidate.start,
                )));
            }
            steps.push(PreparedStep::Loop(candidate.loop_step));
            cursor = candidate.end;
        }
        if cursor < program.ops.len() {
            steps.push(PreparedStep::Flat(flat_ops(
                program,
                cursor,
                program.ops.len(),
            )));
        }

        let estimated_unrolled_bytes = program.ops.len().saturating_mul(UNROLLED_OP_BYTES);
        let estimated_prepared_bytes = steps.iter().map(PreparedStep::estimated_bytes).sum();
        Self {
            slots: program.ops.len(),
            inputs: program.inputs.clone(),
            outputs: program.outputs.clone(),
            steps,
            estimated_unrolled_bytes,
            estimated_prepared_bytes,
        }
    }

    /// Whether the prepared artifact has no operations.
    pub fn is_empty(&self) -> bool {
        self.slots == 0
    }

    /// Whether at least one table-driven loop was selected.
    pub fn has_loops(&self) -> bool {
        self.steps
            .iter()
            .any(|step| matches!(step, PreparedStep::Loop(_)))
    }
}

/// One ordered execution group in a [`PreparedProgram`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedStep {
    /// Operations which remain directly represented.
    Flat(Vec<ScheduledOp>),
    /// A repeated template whose varying absolute slot numbers live in a table.
    Loop(TableLoop),
}

impl PreparedStep {
    fn estimated_bytes(&self) -> usize {
        match self {
            Self::Flat(ops) => ops.len().saturating_mul(UNROLLED_OP_BYTES),
            Self::Loop(loop_step) => LOOP_OVERHEAD_BYTES
                .saturating_add(loop_step.ops.len().saturating_mul(UNROLLED_OP_BYTES))
                .saturating_add(
                    loop_step
                        .table
                        .len()
                        .saturating_mul(core::mem::size_of::<u32>()),
                ),
        }
    }
}

/// An operation with its otherwise implicit output slot made explicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScheduledOp {
    /// The absolute scratch-buffer slot written by `op`.
    pub out: Idx,
    /// The circuit operation to execute.
    pub op: Op,
}

/// A counted table-driven group of structurally identical operations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TableLoop {
    /// Number of template executions.
    pub iterations: u32,
    /// Number of `u32` slot entries in one row of [`Self::table`].
    pub fields_per_iteration: u16,
    /// Packed iteration-major absolute slot numbers.
    pub table: Vec<u32>,
    /// The operation template executed once per table row.
    pub ops: Vec<LoopOp>,
}

impl TableLoop {
    /// Resolve a template slot for `iteration`.
    pub fn resolve(&self, iteration: usize, slot: LoopSlot) -> Idx {
        match slot {
            LoopSlot::Static(slot) => slot,
            LoopSlot::Table(field) => {
                let offset = iteration
                    .checked_mul(self.fields_per_iteration as usize)
                    .and_then(|offset| offset.checked_add(field as usize))
                    .expect("table-loop index must fit usize");
                Idx(self.table[offset])
            }
        }
    }

    /// Materialize one template operation with its absolute slots resolved for
    /// `iteration`.
    pub fn scheduled_op(&self, iteration: usize, op: LoopOp) -> ScheduledOp {
        match op {
            LoopOp::Create { val, out } => ScheduledOp {
                out: self.resolve(iteration, out),
                op: Op::Create(val),
            },
            LoopOp::BitAnd { a, b, out } => ScheduledOp {
                out: self.resolve(iteration, out),
                op: Op::BitAnd(self.resolve(iteration, a), self.resolve(iteration, b)),
            },
            LoopOp::BitOr { a, b, out } => ScheduledOp {
                out: self.resolve(iteration, out),
                op: Op::BitOr(self.resolve(iteration, a), self.resolve(iteration, b)),
            },
            LoopOp::BitXor { a, b, out } => ScheduledOp {
                out: self.resolve(iteration, out),
                op: Op::BitXor(self.resolve(iteration, a), self.resolve(iteration, b)),
            },
            LoopOp::Mux {
                cond,
                then,
                r#else,
                out,
            } => ScheduledOp {
                out: self.resolve(iteration, out),
                op: Op::Mux {
                    cond: self.resolve(iteration, cond),
                    then: self.resolve(iteration, then),
                    r#else: self.resolve(iteration, r#else),
                },
            },
        }
    }
}

/// A slot reference in a [`TableLoop`] template.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoopSlot {
    /// An invariant absolute slot, hoisted out of the table.
    Static(Idx),
    /// A per-iteration absolute slot in the table row.
    Table(u16),
}

/// One circuit operation in a [`TableLoop`] template.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoopOp {
    /// Materialize a known Boolean constant.
    Create {
        /// Boolean value to materialize.
        val: bool,
        /// Destination scratch slot.
        out: LoopSlot,
    },
    /// Bitwise AND.
    BitAnd {
        /// First operand.
        a: LoopSlot,
        /// Second operand.
        b: LoopSlot,
        /// Destination scratch slot.
        out: LoopSlot,
    },
    /// Bitwise OR.
    BitOr {
        /// First operand.
        a: LoopSlot,
        /// Second operand.
        b: LoopSlot,
        /// Destination scratch slot.
        out: LoopSlot,
    },
    /// Bitwise XOR.
    BitXor {
        /// First operand.
        a: LoopSlot,
        /// Second operand.
        b: LoopSlot,
        /// Destination scratch slot.
        out: LoopSlot,
    },
    /// Boolean select.
    Mux {
        /// Condition slot.
        cond: LoopSlot,
        /// True branch slot.
        then: LoopSlot,
        /// False branch slot.
        r#else: LoopSlot,
        /// Destination slot.
        out: LoopSlot,
    },
}

const UNROLLED_OP_BYTES: usize = 20;
const LOOP_OVERHEAD_BYTES: usize = 16;

#[derive(Clone)]
struct Candidate {
    start: usize,
    end: usize,
    savings: usize,
    loop_step: TableLoop,
}

fn flat_ops(program: &Program, start: usize, end: usize) -> Vec<ScheduledOp> {
    (start..end)
        .map(|out| ScheduledOp {
            out: Idx(out as u32),
            op: program.ops[out],
        })
        .collect()
}

fn select_candidates(program: &Program, options: &OptimizationOptions) -> Vec<Candidate> {
    if !options.enabled || options.min_repetitions < 2 || options.max_template_ops == 0 {
        return Vec::new();
    }

    let mut input_slots = alloc::vec![false; program.ops.len()];
    for input in &program.inputs {
        if let Some(slot) = input_slots.get_mut(input.get()) {
            *slot = true;
        }
    }

    let mut candidates = Vec::new();
    for start in 0..program.ops.len() {
        let remaining = program.ops.len() - start;
        let maximum_width = core::cmp::min(
            options.max_template_ops,
            remaining / options.min_repetitions,
        );
        for width in 1..=maximum_width {
            let mut repetitions = 1usize;
            while start + (repetitions + 1) * width <= program.ops.len()
                && blocks_match(&program.ops, start, width, repetitions)
            {
                repetitions += 1;
            }
            if repetitions < options.min_repetitions {
                continue;
            }
            let end = start + repetitions * width;
            if input_slots[start..end].iter().any(|input| *input) {
                continue;
            }
            let loop_step = build_loop(program, start, width, repetitions);
            let unrolled = repetitions
                .saturating_mul(width)
                .saturating_mul(UNROLLED_OP_BYTES);
            let prepared = LOOP_OVERHEAD_BYTES
                .saturating_add(width.saturating_mul(UNROLLED_OP_BYTES))
                .saturating_add(
                    loop_step
                        .table
                        .len()
                        .saturating_mul(core::mem::size_of::<u32>()),
                );
            let savings = unrolled.saturating_sub(prepared);
            if savings >= options.min_static_savings_bytes {
                candidates.push(Candidate {
                    start,
                    end,
                    savings,
                    loop_step,
                });
            }
        }
    }

    candidates.sort_by(|left, right| {
        right
            .savings
            .cmp(&left.savings)
            .then_with(|| right.loop_step.ops.len().cmp(&left.loop_step.ops.len()))
            .then_with(|| left.start.cmp(&right.start))
    });
    let mut taken = alloc::vec![false; program.ops.len()];
    let mut selected = Vec::new();
    for candidate in candidates {
        if taken[candidate.start..candidate.end]
            .iter()
            .any(|taken| *taken)
        {
            continue;
        }
        taken[candidate.start..candidate.end].fill(true);
        selected.push(candidate);
    }
    selected
}

fn blocks_match(ops: &[Op], start: usize, width: usize, repetition: usize) -> bool {
    let later = start + repetition * width;
    ops[start..start + width]
        .iter()
        .zip(&ops[later..later + width])
        .all(|(&left, &right)| same_shape(left, right))
}

fn same_shape(left: Op, right: Op) -> bool {
    match (left, right) {
        (Op::Create(left), Op::Create(right)) => left == right,
        (Op::BitAnd(..), Op::BitAnd(..))
        | (Op::BitOr(..), Op::BitOr(..))
        | (Op::BitXor(..), Op::BitXor(..))
        | (Op::Mux { .. }, Op::Mux { .. }) => true,
        _ => false,
    }
}

#[derive(Clone, Copy)]
enum FieldSource {
    Output(usize),
    Operand { body_op: usize, operand: usize },
}

fn build_loop(program: &Program, start: usize, width: usize, repetitions: usize) -> TableLoop {
    let mut fields = Vec::<FieldSource>::new();
    let mut ops = Vec::with_capacity(width);
    for body_op in 0..width {
        let out = LoopSlot::Table(next_field(&mut fields, FieldSource::Output(body_op)));
        let op = program.ops[start + body_op];
        let operand = |operand: usize, fields: &mut Vec<FieldSource>| {
            classify_slot(program, start, width, repetitions, body_op, operand, fields)
        };
        let op = match op {
            Op::Create(val) => LoopOp::Create { val, out },
            Op::BitAnd(_, _) => LoopOp::BitAnd {
                a: operand(0, &mut fields),
                b: operand(1, &mut fields),
                out,
            },
            Op::BitOr(_, _) => LoopOp::BitOr {
                a: operand(0, &mut fields),
                b: operand(1, &mut fields),
                out,
            },
            Op::BitXor(_, _) => LoopOp::BitXor {
                a: operand(0, &mut fields),
                b: operand(1, &mut fields),
                out,
            },
            Op::Mux { .. } => LoopOp::Mux {
                cond: operand(0, &mut fields),
                then: operand(1, &mut fields),
                r#else: operand(2, &mut fields),
                out,
            },
        };
        ops.push(op);
    }

    let mut table = Vec::with_capacity(repetitions.saturating_mul(fields.len()));
    for repetition in 0..repetitions {
        for source in &fields {
            let slot = match *source {
                FieldSource::Output(body_op) => start + repetition * width + body_op,
                FieldSource::Operand { body_op, operand } => {
                    operand_at(program.ops[start + repetition * width + body_op], operand).get()
                }
            };
            table.push(slot as u32);
        }
    }
    TableLoop {
        iterations: repetitions as u32,
        fields_per_iteration: fields.len() as u16,
        table,
        ops,
    }
}

fn next_field(fields: &mut Vec<FieldSource>, source: FieldSource) -> u16 {
    let field = u16::try_from(fields.len()).expect("table loop cannot exceed u16 fields");
    fields.push(source);
    field
}

fn classify_slot(
    program: &Program,
    start: usize,
    width: usize,
    repetitions: usize,
    body_op: usize,
    operand: usize,
    fields: &mut Vec<FieldSource>,
) -> LoopSlot {
    let first = operand_at(program.ops[start + body_op], operand).get();
    let relative = first
        .checked_sub(start)
        .filter(|relative| *relative < width);
    if let Some(relative) = relative {
        if (0..repetitions).all(|repetition| {
            operand_at(program.ops[start + repetition * width + body_op], operand).get()
                == start + repetition * width + relative
        }) {
            return LoopSlot::Table(next_field(
                fields,
                FieldSource::Operand { body_op, operand },
            ));
        }
    }
    if (1..repetitions).all(|repetition| {
        operand_at(program.ops[start + repetition * width + body_op], operand).get() == first
    }) {
        LoopSlot::Static(Idx(first as u32))
    } else {
        LoopSlot::Table(next_field(
            fields,
            FieldSource::Operand { body_op, operand },
        ))
    }
}

fn operand_at(op: Op, operand: usize) -> Idx {
    match (op, operand) {
        (Op::BitAnd(a, _b) | Op::BitOr(a, _b) | Op::BitXor(a, _b), 0) => a,
        (Op::BitAnd(_, b) | Op::BitOr(_, b) | Op::BitXor(_, b), 1) => b,
        (Op::Mux { cond, .. }, 0) => cond,
        (Op::Mux { then, .. }, 1) => then,
        (Op::Mux { r#else, .. }, 2) => r#else,
        _ => panic!("requested an invalid circuit operation operand"),
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

/// Interpret a [`PreparedProgram`] over `bool`.
///
/// This mirrors [`interpret`] while exercising table-loop execution.  The raw
/// interpreter remains the independent semantic oracle for preparation tests.
pub fn interpret_prepared(program: &PreparedProgram, inputs: &[bool]) -> Vec<bool> {
    assert_eq!(
        inputs.len(),
        program.inputs.len(),
        "input count must match the recorded program's input slots"
    );
    let mut slots: Vec<Option<bool>> = alloc::vec![None; program.slots];
    for (&idx, &value) in program.inputs.iter().zip(inputs) {
        slots[idx.get()] = Some(value);
    }
    for step in &program.steps {
        match step {
            PreparedStep::Flat(ops) => {
                for &op in ops {
                    interpret_scheduled_op(&mut slots, op);
                }
            }
            PreparedStep::Loop(loop_step) => {
                for iteration in 0..loop_step.iterations as usize {
                    for &op in &loop_step.ops {
                        interpret_scheduled_op(&mut slots, loop_step.scheduled_op(iteration, op));
                    }
                }
            }
        }
    }
    program
        .outputs
        .iter()
        .map(|idx| slots[idx.get()].unwrap())
        .collect()
}

fn interpret_scheduled_op(slots: &mut [Option<bool>], scheduled: ScheduledOp) {
    if slots[scheduled.out.get()].is_some() {
        return;
    }
    let value = match scheduled.op {
        Op::Create(value) => value,
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
    slots[scheduled.out.get()] = Some(value);
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
            assert_eq!(
                outputs,
                [x & y, x | y, x ^ y, if x { x | y } else { x ^ y }]
            );
        }
    }

    #[test]
    fn preparation_reabstracts_repeated_calls_without_changing_results() {
        let mut recorder = Recorder::new();
        let inputs = (0..32)
            .map(|_| recorder.create(false).unwrap())
            .collect::<Vec<_>>();
        let outputs = (0..16)
            .map(|index| {
                recorder
                    .bitand(inputs[index * 2], inputs[index * 2 + 1])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let program = recorder.finish(inputs, outputs);
        let prepared = program.prepare(&OptimizationOptions::default());

        assert!(prepared.has_loops());
        assert!(prepared.estimated_prepared_bytes < prepared.estimated_unrolled_bytes);
        assert_eq!(
            interpret(
                &program,
                &[
                    true, false, true, true, false, false, true, true, true, false, true, true,
                    false, true, false, false, true, true, false, true, true, true, false, false,
                    true, false, true, true, false, true, true, true
                ]
            ),
            interpret_prepared(
                &prepared,
                &[
                    true, false, true, true, false, false, true, true, true, false, true, true,
                    false, true, false, false, true, true, false, true, true, true, false, false,
                    true, false, true, true, false, true, true, true
                ]
            ),
        );

        let disabled = program.prepare(&OptimizationOptions {
            enabled: false,
            ..OptimizationOptions::default()
        });
        assert!(!disabled.has_loops());
        assert_eq!(prepared, program.prepare(&OptimizationOptions::default()));
    }
}
