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

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::convert::Infallible;

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux,
    ContextWithStorage, ContextWithValue, HasError, StorageAddressBit,
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
    /// Invoke one externally supplied Boolean primitive bit. The metadata and
    /// operand list live in [`Program::externals`] and are selected by this
    /// stable table index.
    External(u32),
}

/// External primitive class used by [`ExternalOp`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalKind {
    /// Deterministic oracle bit.
    Oracle,
    /// Replayable RNG bit.
    Rng,
    /// Action result bit. Storage effects remain owned by the caller's host.
    Action,
}

/// Stable metadata for one externally supplied recorded operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalOp {
    /// Primitive class.
    pub kind: ExternalKind,
    /// Caller-resolved logical source name.
    pub name: alloc::string::String,
    /// Input slots in flattened bit order.
    pub args: Vec<Idx>,
    /// Flattened output bit index.
    pub bit: usize,
    /// Stable occurrence token used for replay.
    pub occurrence: u64,
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
    /// Stable external metadata referenced by [`Op::External`].
    pub externals: Vec<ExternalOp>,
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

    /// Prepare this program, then compact its scratch-buffer slots.
    ///
    /// Shorthand for `self.prepare(options).compact_slots()`. `Program`
    /// itself can never represent slot reuse -- an op's output slot is
    /// implicitly its position in [`Program::ops`], so two ops can't share
    /// one output slot without abandoning that positional identity, which
    /// every raw-`Program` consumer (including the R1CS/Groth16 backends,
    /// which treat each [`Idx`] as a fixed-identity witness) depends on.
    /// [`PreparedProgram`] is the only representation with an explicit,
    /// position-independent output slot per operation, so this method
    /// routes through it.
    pub fn compact(&self, options: &OptimizationOptions) -> PreparedProgram {
        self.prepare(options).compact_slots()
    }

    /// Add one external bit operation and return the slot it defines.
    pub fn push_external(&mut self, external: ExternalOp) -> Idx {
        let id = u32::try_from(self.externals.len()).expect("external table exceeds u32");
        let slot = Idx(self.ops.len() as u32);
        self.externals.push(external);
        self.ops.push(Op::External(id));
        slot
    }
}

/// Tuning knobs for [`Program::prepare`] and [`PreparedProgram::reoptimize`].
///
/// The cost model is deliberately about static artifact size and compilation
/// work, not an unproven runtime-speed estimate. A caller can turn the pass
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

/// A half-open range into [`PreparedProgram::statements`].
///
/// The entry and every loop body are represented by this same range type.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct StatementRange {
    /// First statement in the range.
    pub start: u32,
    /// One past the last statement in the range.
    pub end: u32,
}

impl StatementRange {
    /// Make a range from pool indices.
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// Whether the range contains no statements.
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// One slot named by a prepared statement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedSlot {
    /// An absolute scratch-buffer slot.
    Static(Idx),
    /// A field in the active loop table. `depth = 0` names the innermost
    /// active loop; larger depths name lexical ancestors.
    Table {
        /// Number of lexical loop scopes between the statement and its owner.
        depth: u16,
        /// Field within the owner's table row.
        field: u32,
    },
}

/// An operation whose slots may be absolute or table-driven.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedOp {
    /// Materialize a known Boolean constant.
    Create {
        /// Constant value.
        value: bool,
        /// Destination slot.
        out: PreparedSlot,
    },
    /// Bitwise AND.
    BitAnd {
        /// First operand.
        a: PreparedSlot,
        /// Second operand.
        b: PreparedSlot,
        /// Destination slot.
        out: PreparedSlot,
    },
    /// Bitwise OR.
    BitOr {
        /// First operand.
        a: PreparedSlot,
        /// Second operand.
        b: PreparedSlot,
        /// Destination slot.
        out: PreparedSlot,
    },
    /// Bitwise XOR.
    BitXor {
        /// First operand.
        a: PreparedSlot,
        /// Second operand.
        b: PreparedSlot,
        /// Destination slot.
        out: PreparedSlot,
    },
    /// Select `then` when `cond` is set, else `r#else`.
    Mux {
        /// Condition slot.
        cond: PreparedSlot,
        /// True branch slot.
        then: PreparedSlot,
        /// False branch slot.
        r#else: PreparedSlot,
        /// Destination slot.
        out: PreparedSlot,
    },
    /// An external bit operation referenced through
    /// [`PreparedProgram::externals`].
    External {
        /// External metadata index.
        external: u32,
        /// Destination slot.
        out: PreparedSlot,
    },
}

/// One invocation of a [`PreparedLoop`].
///
/// A nested loop has one descriptor for each parent-body execution. Its trip
/// count may therefore vary with the parent while all of its circuit rows
/// retain one compact body template.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoopInvocation {
    /// First table row for this invocation.
    pub first_row: u32,
    /// Concrete number of body executions.
    pub iterations: u32,
}

/// A table-driven loop whose body is a range in the shared statement pool.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedLoop {
    /// Template body in [`PreparedProgram::statements`].
    pub body: StatementRange,
    /// One descriptor for every dynamic execution of this loop statement.
    pub invocations: Vec<LoopInvocation>,
    /// Number of `u32` slot entries in one table row.
    pub fields_per_iteration: u32,
    /// Packed, iteration-major absolute slot numbers for all invocations.
    pub table: Vec<u32>,
}

/// One statement in a [`PreparedProgram`] pool.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Statement {
    /// An operation executed once whenever its enclosing range is entered.
    Op(PreparedOp),
    /// A counted loop whose body refers back into the statement pool.
    Loop(PreparedLoop),
}

/// An operation with its otherwise implicit output slot made explicit.
///
/// This is the fully-resolved form consumed by the pinned-operation ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScheduledOp {
    /// The absolute scratch-buffer slot written by `op`.
    pub out: Idx,
    /// The circuit operation to execute.
    pub op: Op,
}

/// An invalid structural prepared-program artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedProgramError {
    /// A statement range is inverted or extends beyond the pool.
    InvalidRange,
    /// A loop's invocation descriptors or table length are inconsistent.
    InvalidLoopTable,
    /// A statement refers to a table scope not active at that point.
    InvalidSlotScope,
    /// A loop body recursively contains itself.
    RecursiveRange,
    /// A nested loop has a different number of invocation descriptors than
    /// its enclosing body can execute.
    InvalidInvocationCount,
    /// An absolute slot reference is outside the scratch-buffer width.
    InvalidSlot,
    /// An external operation refers to missing metadata or an out-of-range
    /// argument slot.
    InvalidExternal,
}

const UNROLLED_OP_BYTES: usize = 20;
const LOOP_OVERHEAD_BYTES: usize = 16;
const INVOCATION_BYTES: usize = core::mem::size_of::<u32>() * 2;

/// A [`Program`] arranged for execution by an optimizing backend.
///
/// It deliberately keeps the raw program's slot numbering, input order, and
/// output order. The entry and every loop body share one statement pool,
/// making loop nesting representable without changing the pinned-operation
/// ABI.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedProgram {
    /// Scratch-buffer width.
    ///
    /// Identical to the source [`Program::len`] for any artifact produced by
    /// [`Program::prepare`]/[`PreparedProgram::reoptimize`] alone --
    /// [`PreparedProgram::compact_slots`] may shrink it further.
    pub slots: usize,
    /// Input slots in the source program's deterministic order.
    pub inputs: Vec<Idx>,
    /// Output slots in the source program's deterministic order.
    pub outputs: Vec<Idx>,
    /// Stable external metadata referenced by prepared external operations.
    pub externals: Vec<ExternalOp>,
    /// Root statement range executed once.
    pub entry: StatementRange,
    /// Shared pool containing the entry and all loop bodies.
    pub statements: Vec<Statement>,
    /// Estimated bytes for the source's fully unrolled call sequence.
    pub estimated_unrolled_bytes: usize,
    /// Estimated bytes for this prepared sequence, including all tables.
    pub estimated_prepared_bytes: usize,
}

impl PreparedProgram {
    /// Build a structured prepared artifact from an explicit statement pool.
    ///
    /// This is the construction seam used by source-aware recorders.  It
    /// keeps the raw program's absolute slot numbering and declared I/O
    /// order, while allowing a frontend to preserve loop boundaries it knows
    /// from its source representation.
    pub fn new(
        slots: usize,
        inputs: Vec<Idx>,
        outputs: Vec<Idx>,
        entry: StatementRange,
        statements: Vec<Statement>,
    ) -> Result<Self, PreparedProgramError> {
        let mut program = Self {
            slots,
            inputs,
            outputs,
            externals: Vec::new(),
            entry,
            statements,
            estimated_unrolled_bytes: slots.saturating_mul(UNROLLED_OP_BYTES),
            estimated_prepared_bytes: 0,
        };
        program.validate()?;
        program.refresh_estimate();
        Ok(program)
    }

    fn from_program(program: &Program, options: &OptimizationOptions) -> Self {
        let statements = program
            .ops
            .iter()
            .cloned()
            .enumerate()
            .map(|(out, op)| {
                Statement::Op(PreparedOp::from_scheduled(ScheduledOp {
                    out: Idx(out as u32),
                    op,
                }))
            })
            .collect::<Vec<_>>();
        let mut prepared = Self {
            slots: program.ops.len(),
            inputs: program.inputs.clone(),
            outputs: program.outputs.clone(),
            externals: program.externals.clone(),
            entry: StatementRange::new(0, program.ops.len() as u32),
            statements,
            estimated_unrolled_bytes: program.ops.len().saturating_mul(UNROLLED_OP_BYTES),
            estimated_prepared_bytes: 0,
        };
        prepared.refresh_estimate();
        prepared.reoptimize(options)
    }

    /// Reoptimize this structured artifact to a deterministic fixed point.
    ///
    /// The pass visits nested bodies, discovers profitable raw repetitions,
    /// and merges adjacent compatible loops. Existing loop tables are never
    /// expanded unless a caller explicitly converts the artifact to raw form.
    pub fn reoptimize(&self, options: &OptimizationOptions) -> Self {
        let mut optimized = self.clone();
        if !options.enabled || optimized.validate().is_err() {
            optimized.refresh_estimate();
            return optimized;
        }
        loop {
            let mut changed = false;
            optimized.entry = optimized.optimize_range(optimized.entry, options, &mut changed);
            if !changed {
                break;
            }
        }
        optimized.refresh_estimate();
        optimized
    }

    /// Validate statement-pool and table invariants before execution.
    pub fn validate(&self) -> Result<(), PreparedProgramError> {
        if self
            .inputs
            .iter()
            .chain(&self.outputs)
            .any(|slot| slot.get() >= self.slots)
        {
            return Err(PreparedProgramError::InvalidSlot);
        }
        self.validate_range(self.entry, 0, 1, &mut Vec::new())?;
        if self
            .externals
            .iter()
            .any(|external| external.args.iter().any(|slot| slot.get() >= self.slots))
        {
            return Err(PreparedProgramError::InvalidExternal);
        }
        Ok(())
    }

    /// Whether the prepared artifact has no operations.
    pub fn is_empty(&self) -> bool {
        self.slots == 0
    }

    /// Whether the reachable entry tree contains a table-driven loop.
    pub fn has_loops(&self) -> bool {
        self.range_has_loops(self.entry)
    }

    fn append(&mut self, statements: Vec<Statement>) -> StatementRange {
        let start = u32::try_from(self.statements.len()).expect("statement pool exceeds u32");
        self.statements.extend(statements);
        let end = u32::try_from(self.statements.len()).expect("statement pool exceeds u32");
        StatementRange::new(start, end)
    }

    fn range_slice(&self, range: StatementRange) -> Option<&[Statement]> {
        let start = usize::try_from(range.start).ok()?;
        let end = usize::try_from(range.end).ok()?;
        self.statements.get(start..end)
    }

    fn optimize_range(
        &mut self,
        range: StatementRange,
        options: &OptimizationOptions,
        changed: &mut bool,
    ) -> StatementRange {
        let original = self
            .range_slice(range)
            .expect("validated range remains valid during optimization")
            .to_vec();
        let mut statements = original.clone();
        for statement in &mut statements {
            if let Statement::Loop(loop_step) = statement {
                let body = self.optimize_range(loop_step.body, options, changed);
                loop_step.body = body;
            }
        }
        let candidates = select_candidates(&statements, &self.inputs, options);
        if !candidates.is_empty() {
            let mut selected = candidates;
            selected.sort_by_key(|candidate| candidate.start);
            let mut rebuilt = Vec::new();
            let mut cursor = 0usize;
            for candidate in selected {
                rebuilt.extend_from_slice(&statements[cursor..candidate.start]);
                let body = self.append(candidate.draft.body);
                rebuilt.push(Statement::Loop(PreparedLoop {
                    body,
                    invocations: alloc::vec![LoopInvocation {
                        first_row: 0,
                        iterations: candidate.repetitions as u32,
                    }],
                    fields_per_iteration: candidate.draft.fields_per_iteration,
                    table: candidate.draft.table,
                }));
                cursor = candidate.end;
            }
            rebuilt.extend_from_slice(&statements[cursor..]);
            statements = rebuilt;
            *changed = true;
        }
        let (merged, merged_any) = merge_adjacent_loops(self, statements);
        statements = merged;
        if merged_any {
            *changed = true;
        }
        if statements == original {
            range
        } else {
            self.append(statements)
        }
    }

    fn validate_range(
        &self,
        range: StatementRange,
        loop_depth: u16,
        expected_invocations: usize,
        active_ranges: &mut Vec<StatementRange>,
    ) -> Result<(), PreparedProgramError> {
        let statements = self
            .range_slice(range)
            .ok_or(PreparedProgramError::InvalidRange)?;
        if active_ranges.contains(&range) {
            return Err(PreparedProgramError::RecursiveRange);
        }
        active_ranges.push(range);
        for statement in statements {
            match statement {
                Statement::Op(op) => {
                    op.validate_scopes(loop_depth, self.slots)?;
                    if let PreparedOp::External { external, .. } = op {
                        if *external as usize >= self.externals.len() {
                            return Err(PreparedProgramError::InvalidExternal);
                        }
                    }
                }
                Statement::Loop(loop_step) => {
                    if loop_step.invocations.len() != expected_invocations {
                        return Err(PreparedProgramError::InvalidInvocationCount);
                    }
                    let mut rows = 0u32;
                    for invocation in &loop_step.invocations {
                        if invocation.first_row != rows {
                            return Err(PreparedProgramError::InvalidLoopTable);
                        }
                        rows = rows
                            .checked_add(invocation.iterations)
                            .ok_or(PreparedProgramError::InvalidLoopTable)?;
                    }
                    let expected = usize::try_from(rows)
                        .ok()
                        .and_then(|rows| rows.checked_mul(loop_step.fields_per_iteration as usize))
                        .ok_or(PreparedProgramError::InvalidLoopTable)?;
                    if expected != loop_step.table.len() {
                        return Err(PreparedProgramError::InvalidLoopTable);
                    }
                    if loop_step
                        .table
                        .iter()
                        .any(|&slot| usize::try_from(slot).map_or(true, |slot| slot >= self.slots))
                    {
                        return Err(PreparedProgramError::InvalidSlot);
                    }
                    self.validate_range(
                        loop_step.body,
                        loop_depth.saturating_add(1),
                        rows as usize,
                        active_ranges,
                    )?;
                }
            }
        }
        active_ranges.pop();
        Ok(())
    }

    fn range_has_loops(&self, range: StatementRange) -> bool {
        self.range_slice(range).is_some_and(|statements| {
            statements.iter().any(|statement| match statement {
                Statement::Op(_) => false,
                Statement::Loop(_) => true,
            })
        })
    }

    fn refresh_estimate(&mut self) {
        self.estimated_unrolled_bytes = self.slots.saturating_mul(UNROLLED_OP_BYTES);
        self.estimated_prepared_bytes = self.estimate_range(self.entry);
    }

    fn estimate_range(&self, range: StatementRange) -> usize {
        self.range_slice(range)
            .unwrap_or_default()
            .iter()
            .map(|statement| match statement {
                Statement::Op(_) => UNROLLED_OP_BYTES,
                Statement::Loop(loop_step) => LOOP_OVERHEAD_BYTES
                    .saturating_add(loop_step.invocations.len().saturating_mul(INVOCATION_BYTES))
                    .saturating_add(
                        loop_step
                            .table
                            .len()
                            .saturating_mul(core::mem::size_of::<u32>()),
                    )
                    .saturating_add(self.estimate_range(loop_step.body)),
            })
            .sum()
    }
}

impl PreparedOp {
    fn from_scheduled(scheduled: ScheduledOp) -> Self {
        let out = PreparedSlot::Static(scheduled.out);
        match scheduled.op {
            Op::Create(value) => Self::Create { value, out },
            Op::BitAnd(a, b) => Self::BitAnd {
                a: PreparedSlot::Static(a),
                b: PreparedSlot::Static(b),
                out,
            },
            Op::BitOr(a, b) => Self::BitOr {
                a: PreparedSlot::Static(a),
                b: PreparedSlot::Static(b),
                out,
            },
            Op::BitXor(a, b) => Self::BitXor {
                a: PreparedSlot::Static(a),
                b: PreparedSlot::Static(b),
                out,
            },
            Op::Mux { cond, then, r#else } => Self::Mux {
                cond: PreparedSlot::Static(cond),
                then: PreparedSlot::Static(then),
                r#else: PreparedSlot::Static(r#else),
                out,
            },
            Op::External(external) => Self::External { external, out },
        }
    }

    fn validate_scopes(self, loop_depth: u16, slots: usize) -> Result<(), PreparedProgramError> {
        self.slots().iter().try_for_each(|slot| match *slot {
            PreparedSlot::Static(slot) if slot.get() < slots => Ok(()),
            PreparedSlot::Static(_) => Err(PreparedProgramError::InvalidSlot),
            PreparedSlot::Table { depth, .. } if depth < loop_depth => Ok(()),
            PreparedSlot::Table { .. } => Err(PreparedProgramError::InvalidSlotScope),
        })
    }

    fn slots(&self) -> [PreparedSlot; 4] {
        match *self {
            Self::Create { out, .. } => [out, out, out, out],
            Self::BitAnd { a, b, out } | Self::BitOr { a, b, out } | Self::BitXor { a, b, out } => {
                [a, b, out, out]
            }
            Self::Mux {
                cond,
                then,
                r#else,
                out,
            } => [cond, then, r#else, out],
            Self::External { out, .. } => [out, out, out, out],
        }
    }

    /// Resolve this template operation through its active loop scopes.
    pub fn resolve(&self, mut slot: impl FnMut(PreparedSlot) -> Idx) -> ScheduledOp {
        match *self {
            Self::Create { value, out } => ScheduledOp {
                out: slot(out),
                op: Op::Create(value),
            },
            Self::BitAnd { a, b, out } => ScheduledOp {
                out: slot(out),
                op: Op::BitAnd(slot(a), slot(b)),
            },
            Self::BitOr { a, b, out } => ScheduledOp {
                out: slot(out),
                op: Op::BitOr(slot(a), slot(b)),
            },
            Self::BitXor { a, b, out } => ScheduledOp {
                out: slot(out),
                op: Op::BitXor(slot(a), slot(b)),
            },
            Self::Mux {
                cond,
                then,
                r#else,
                out,
            } => ScheduledOp {
                out: slot(out),
                op: Op::Mux {
                    cond: slot(cond),
                    then: slot(then),
                    r#else: slot(r#else),
                },
            },
            Self::External { external, out } => ScheduledOp {
                out: slot(out),
                op: Op::External(external),
            },
        }
    }

    fn as_scheduled(self) -> Option<ScheduledOp> {
        let static_slot = |slot| match slot {
            PreparedSlot::Static(slot) => Some(slot),
            PreparedSlot::Table { .. } => None,
        };
        match self {
            Self::Create { value, out } => Some(ScheduledOp {
                out: static_slot(out)?,
                op: Op::Create(value),
            }),
            Self::BitAnd { a, b, out } => Some(ScheduledOp {
                out: static_slot(out)?,
                op: Op::BitAnd(static_slot(a)?, static_slot(b)?),
            }),
            Self::BitOr { a, b, out } => Some(ScheduledOp {
                out: static_slot(out)?,
                op: Op::BitOr(static_slot(a)?, static_slot(b)?),
            }),
            Self::BitXor { a, b, out } => Some(ScheduledOp {
                out: static_slot(out)?,
                op: Op::BitXor(static_slot(a)?, static_slot(b)?),
            }),
            Self::Mux {
                cond,
                then,
                r#else,
                out,
            } => Some(ScheduledOp {
                out: static_slot(out)?,
                op: Op::Mux {
                    cond: static_slot(cond)?,
                    then: static_slot(then)?,
                    r#else: static_slot(r#else)?,
                },
            }),
            Self::External { external, out } => Some(ScheduledOp {
                out: static_slot(out)?,
                op: Op::External(external),
            }),
        }
    }
}

#[derive(Clone)]
struct Candidate {
    start: usize,
    end: usize,
    savings: usize,
    repetitions: usize,
    draft: LoopDraft,
}

#[derive(Clone)]
struct LoopDraft {
    body: Vec<Statement>,
    fields_per_iteration: u32,
    table: Vec<u32>,
}

fn select_candidates(
    statements: &[Statement],
    inputs: &[Idx],
    options: &OptimizationOptions,
) -> Vec<Candidate> {
    if !options.enabled || options.min_repetitions < 2 || options.max_template_ops == 0 {
        return Vec::new();
    }
    let scheduled = statements
        .iter()
        .map(|statement| match statement {
            Statement::Op(op) => op.as_scheduled(),
            Statement::Loop(_) => None,
        })
        .collect::<Vec<_>>();
    let mut candidates = Vec::new();
    for start in 0..scheduled.len() {
        let remaining = scheduled.len() - start;
        let maximum_width = core::cmp::min(
            options.max_template_ops,
            remaining / options.min_repetitions,
        );
        for width in 1..=maximum_width {
            if scheduled[start..start + width].iter().any(Option::is_none) {
                continue;
            }
            let mut repetitions = 1usize;
            while start + (repetitions + 1) * width <= scheduled.len()
                && blocks_match(&scheduled, start, width, repetitions)
            {
                repetitions += 1;
            }
            if repetitions < options.min_repetitions {
                continue;
            }
            let end = start + repetitions * width;
            if scheduled[start..end].iter().any(Option::is_none)
                || scheduled[start..end]
                    .iter()
                    .flatten()
                    .any(|op| inputs.contains(&op.out))
            {
                continue;
            }
            let draft = build_loop(&scheduled[start..end], width, repetitions);
            let unrolled = repetitions
                .saturating_mul(width)
                .saturating_mul(UNROLLED_OP_BYTES);
            let prepared = LOOP_OVERHEAD_BYTES
                .saturating_add(INVOCATION_BYTES)
                .saturating_add(width.saturating_mul(UNROLLED_OP_BYTES))
                .saturating_add(
                    draft
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
                    repetitions,
                    draft,
                });
            }
        }
    }
    candidates.sort_by(|left, right| {
        right
            .savings
            .cmp(&left.savings)
            .then_with(|| right.draft.body.len().cmp(&left.draft.body.len()))
            .then_with(|| left.start.cmp(&right.start))
    });
    let mut taken = alloc::vec![false; statements.len()];
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

fn blocks_match(
    ops: &[Option<ScheduledOp>],
    start: usize,
    width: usize,
    repetition: usize,
) -> bool {
    let later = start + repetition * width;
    ops[start..start + width]
        .iter()
        .zip(&ops[later..later + width])
        .all(|(left, right)| match (left, right) {
            (Some(left), Some(right)) => same_shape(left.op, right.op),
            _ => false,
        })
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

fn build_loop(ops: &[Option<ScheduledOp>], width: usize, repetitions: usize) -> LoopDraft {
    let mut fields = Vec::<FieldSource>::new();
    let mut body = Vec::with_capacity(width);
    for body_op in 0..width {
        let op = ops[body_op].expect("selected candidate has scheduled operations");
        let out = PreparedSlot::Table {
            depth: 0,
            field: next_field(&mut fields, FieldSource::Output(body_op)),
        };
        let operand = |operand: usize, fields: &mut Vec<FieldSource>| {
            classify_slot(ops, width, repetitions, body_op, operand, fields)
        };
        let op = match op.op {
            Op::Create(value) => PreparedOp::Create { value, out },
            Op::BitAnd(_, _) => PreparedOp::BitAnd {
                a: operand(0, &mut fields),
                b: operand(1, &mut fields),
                out,
            },
            Op::BitOr(_, _) => PreparedOp::BitOr {
                a: operand(0, &mut fields),
                b: operand(1, &mut fields),
                out,
            },
            Op::BitXor(_, _) => PreparedOp::BitXor {
                a: operand(0, &mut fields),
                b: operand(1, &mut fields),
                out,
            },
            Op::Mux { .. } => PreparedOp::Mux {
                cond: operand(0, &mut fields),
                then: operand(1, &mut fields),
                r#else: operand(2, &mut fields),
                out,
            },
            Op::External(_) => unreachable!("external operations are never loop-abstracted"),
        };
        body.push(Statement::Op(op));
    }
    let mut table = Vec::with_capacity(repetitions.saturating_mul(fields.len()));
    for repetition in 0..repetitions {
        for source in &fields {
            let op = ops[repetition * width
                + match *source {
                    FieldSource::Output(body_op) => body_op,
                    FieldSource::Operand { body_op, .. } => body_op,
                }]
            .expect("selected candidate has scheduled operations");
            let slot = match *source {
                FieldSource::Output(_) => op.out,
                FieldSource::Operand { operand, .. } => operand_at(op.op, operand),
            };
            table.push(slot.0);
        }
    }
    LoopDraft {
        body,
        fields_per_iteration: fields.len() as u32,
        table,
    }
}

fn next_field(fields: &mut Vec<FieldSource>, source: FieldSource) -> u32 {
    let field = u32::try_from(fields.len()).expect("table loop cannot exceed u32 fields");
    fields.push(source);
    field
}

fn classify_slot(
    ops: &[Option<ScheduledOp>],
    width: usize,
    repetitions: usize,
    body_op: usize,
    operand: usize,
    fields: &mut Vec<FieldSource>,
) -> PreparedSlot {
    let first = operand_at(
        ops[body_op]
            .expect("selected candidate has scheduled operations")
            .op,
        operand,
    );
    let base = ops[0]
        .expect("selected candidate has scheduled operations")
        .out
        .get();
    let relative = first
        .get()
        .checked_sub(base)
        .filter(|relative| *relative < width);
    if let Some(relative) = relative {
        if (0..repetitions).all(|repetition| {
            operand_at(
                ops[repetition * width + body_op]
                    .expect("selected candidate has scheduled operations")
                    .op,
                operand,
            )
            .get()
                == ops[repetition * width]
                    .expect("selected candidate has scheduled operations")
                    .out
                    .get()
                    + relative
        }) {
            return PreparedSlot::Table {
                depth: 0,
                field: next_field(fields, FieldSource::Operand { body_op, operand }),
            };
        }
    }
    if (1..repetitions).all(|repetition| {
        operand_at(
            ops[repetition * width + body_op]
                .expect("selected candidate has scheduled operations")
                .op,
            operand,
        ) == first
    }) {
        PreparedSlot::Static(first)
    } else {
        PreparedSlot::Table {
            depth: 0,
            field: next_field(fields, FieldSource::Operand { body_op, operand }),
        }
    }
}

fn operand_at(op: Op, operand: usize) -> Idx {
    match (op, operand) {
        (Op::BitAnd(a, _) | Op::BitOr(a, _) | Op::BitXor(a, _), 0) => a,
        (Op::BitAnd(_, b) | Op::BitOr(_, b) | Op::BitXor(_, b), 1) => b,
        (Op::Mux { cond, .. }, 0) => cond,
        (Op::Mux { then, .. }, 1) => then,
        (Op::Mux { r#else, .. }, 2) => r#else,
        _ => panic!("requested an invalid circuit operation operand"),
    }
}

fn merge_adjacent_loops(
    program: &mut PreparedProgram,
    statements: Vec<Statement>,
) -> (Vec<Statement>, bool) {
    let mut merged = Vec::with_capacity(statements.len());
    let mut changed = false;
    for statement in statements {
        if let Statement::Loop(right) = &statement
            && let Some(Statement::Loop(left)) = merged.last()
            && let Some(loop_step) = merge_loops(program, left, right)
        {
            *merged.last_mut().expect("a loop was just inspected") = Statement::Loop(loop_step);
            changed = true;
            continue;
        }
        merged.push(statement);
    }
    (merged, changed)
}

fn merge_loops(
    program: &mut PreparedProgram,
    left: &PreparedLoop,
    right: &PreparedLoop,
) -> Option<PreparedLoop> {
    if left.fields_per_iteration != right.fields_per_iteration
        || left.invocations.len() != right.invocations.len()
    {
        return None;
    }
    let left_body = program.range_slice(left.body)?.to_vec();
    let right_body = program.range_slice(right.body)?.to_vec();
    let body = if left_body
        .iter()
        .all(|statement| matches!(statement, Statement::Op(_)))
        && left_body == right_body
    {
        left.body
    } else if left.invocations.len() == 1 {
        merge_body_sequence(program, &left_body, &right_body)?
    } else {
        return None;
    };
    let fields = left.fields_per_iteration as usize;
    let mut table = Vec::new();
    let mut invocations = Vec::with_capacity(left.invocations.len());
    let mut next_row = 0u32;
    for (left_invocation, right_invocation) in left.invocations.iter().zip(&right.invocations) {
        append_rows(&mut table, left, *left_invocation, fields)?;
        append_rows(&mut table, right, *right_invocation, fields)?;
        let iterations = left_invocation
            .iterations
            .checked_add(right_invocation.iterations)?;
        invocations.push(LoopInvocation {
            first_row: next_row,
            iterations,
        });
        next_row = next_row.checked_add(iterations)?;
    }
    Some(PreparedLoop {
        body,
        invocations,
        fields_per_iteration: left.fields_per_iteration,
        table,
    })
}

/// Merge one loop-body template after its parent joins a left execution
/// segment followed by a right execution segment.  Nested loop descriptors
/// are concatenated in that same order, so their concrete trip counts retain
/// their association with the active parent row.
fn merge_body_sequence(
    program: &mut PreparedProgram,
    left: &[Statement],
    right: &[Statement],
) -> Option<StatementRange> {
    if left.len() != right.len() {
        return None;
    }
    let mut merged = Vec::with_capacity(left.len());
    for (left, right) in left.iter().zip(right) {
        match (left, right) {
            (Statement::Op(left), Statement::Op(right)) if left == right => {
                merged.push(Statement::Op(*left));
            }
            (Statement::Loop(left), Statement::Loop(right)) => {
                merged.push(Statement::Loop(concatenate_loop_executions(
                    program, left, right,
                )?));
            }
            _ => return None,
        }
    }
    Some(program.append(merged))
}

/// Concatenate all executions of two equivalent nested loop statements.
///
/// Unlike [`merge_loops`], this preserves each descriptor as a separate
/// execution because the newly merged parent body is entered once for each
/// old parent row.  It is the recursive half of nested-loop merging.
fn concatenate_loop_executions(
    program: &mut PreparedProgram,
    left: &PreparedLoop,
    right: &PreparedLoop,
) -> Option<PreparedLoop> {
    if left.fields_per_iteration != right.fields_per_iteration {
        return None;
    }
    let left_body = program.range_slice(left.body)?.to_vec();
    let right_body = program.range_slice(right.body)?.to_vec();
    let body = merge_body_sequence(program, &left_body, &right_body)?;
    let fields = left.fields_per_iteration as usize;
    let mut table = Vec::new();
    let mut invocations = Vec::with_capacity(left.invocations.len() + right.invocations.len());
    let mut next_row = 0u32;
    for invocation in left.invocations.iter().chain(&right.invocations) {
        let source = if invocations.len() < left.invocations.len() {
            left
        } else {
            right
        };
        append_rows(&mut table, source, *invocation, fields)?;
        invocations.push(LoopInvocation {
            first_row: next_row,
            iterations: invocation.iterations,
        });
        next_row = next_row.checked_add(invocation.iterations)?;
    }
    Some(PreparedLoop {
        body,
        invocations,
        fields_per_iteration: left.fields_per_iteration,
        table,
    })
}

fn append_rows(
    destination: &mut Vec<u32>,
    loop_step: &PreparedLoop,
    invocation: LoopInvocation,
    fields: usize,
) -> Option<()> {
    let start = usize::try_from(invocation.first_row)
        .ok()?
        .checked_mul(fields)?;
    let end = usize::try_from(invocation.iterations)
        .ok()?
        .checked_mul(fields)?
        .checked_add(start)?;
    destination.extend_from_slice(loop_step.table.get(start..end)?);
    Some(())
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
    externals: Vec<ExternalOp>,
}

impl Recorder {
    /// Start recording an empty trace.
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            externals: Vec::new(),
        }
    }

    fn push(&mut self, op: Op) -> Idx {
        let idx = Idx(self.ops.len() as u32);
        self.ops.push(op);
        idx
    }

    /// Record one externally dispatched Boolean bit.
    pub fn external_bit(&mut self, external: ExternalOp) -> Idx {
        let id = u32::try_from(self.externals.len()).expect("external table exceeds u32");
        self.externals.push(external);
        self.push(Op::External(id))
    }

    /// Finish recording, naming the slots a caller cares about as this
    /// trace's inputs and outputs.
    pub fn finish(self, inputs: Vec<Idx>, outputs: Vec<Idx>) -> Program {
        Program {
            ops: self.ops,
            inputs,
            outputs,
            externals: self.externals,
        }
    }

    /// Finish through a zero-sized preparation mode.
    ///
    /// `NoPreparation` returns the raw [`Program`], while
    /// `CountdownPreparation` returns a [`PreparedProgram`].  The mode has
    /// no runtime state, so normal recording does not carry loop-recognition
    /// state or branches merely because prepared recording is available.
    #[inline]
    pub fn finish_with<Mode: PreparationMode>(
        self,
        inputs: Vec<Idx>,
        outputs: Vec<Idx>,
        options: &OptimizationOptions,
    ) -> Mode::Output {
        Mode::finish(self.finish(inputs, outputs), options)
    }
}

/// A zero-sized choice of how a completed recording is materialized.
///
/// Source frontends select this at compile time.  It deliberately has no
/// recording callbacks: source-specific adapters may preserve recognized
/// loops using [`PreparedProgram::new`], while its conservative fallback is
/// the same deterministic [`Program::prepare`] pass available for any raw
/// trace.
pub trait PreparationMode {
    /// The completed artifact type.
    type Output;

    /// Materialize a completed raw trace.
    fn finish(program: Program, options: &OptimizationOptions) -> Self::Output;
}

/// Leave a recording as its portable raw [`Program`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NoPreparation;

impl PreparationMode for NoPreparation {
    type Output = Program;

    #[inline]
    fn finish(program: Program, _: &OptimizationOptions) -> Self::Output {
        program
    }
}

/// Materialize a recording as a reoptimizable prepared program.
///
/// The name reflects the current canonical source adapters: a frontend may
/// retain a validated countdown-loop tree through [`PreparedProgram::new`]
/// and use this mode as the safe, flat fallback whenever recognition fails.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CountdownPreparation;

impl PreparationMode for CountdownPreparation {
    type Output = PreparedProgram;

    #[inline]
    fn finish(program: Program, options: &OptimizationOptions) -> Self::Output {
        program.prepare(options)
    }
}

/// An opt-in [`Recorder`] facade that completes directly to a
/// [`PreparedProgram`].
///
/// It has the exact same [`Idx`] wire ABI as [`Recorder`], so ERT entry
/// points can use it without changing their machine or direct-execution ABI.
#[derive(Clone, Debug)]
pub struct PreparedRecorder {
    recorder: Recorder,
    options: OptimizationOptions,
}

impl Default for PreparedRecorder {
    fn default() -> Self {
        Self::new(OptimizationOptions::default())
    }
}

impl PreparedRecorder {
    /// Start prepared recording with explicit reoptimization options.
    pub fn new(options: OptimizationOptions) -> Self {
        Self {
            recorder: Recorder::new(),
            options,
        }
    }

    /// Complete this recording using its configured preparation mode.
    pub fn finish(self, inputs: Vec<Idx>, outputs: Vec<Idx>) -> PreparedProgram {
        self.recorder
            .finish_with::<CountdownPreparation>(inputs, outputs, &self.options)
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

/// Dense symbolic storage for the recording test host.
///
/// Concrete addresses update/read a cell directly, preserving the compact
/// trace shape expected from ABI setup. Secret addresses are represented by
/// ordinary Boolean MUX/demux gates, so a recorded program remains valid for
/// the same storage accesses on a symbolic backend.
impl ContextWithStorage<bool> for Recorder {
    type Storage = [Idx];

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Idx>],
    ) -> Result<Idx, Self::Error> {
        if let Some(index) = recorded_storage_index(address) {
            return Ok(storage[index]);
        }
        assert!(
            storage.len().is_power_of_two() && (1usize << address.len()) == storage.len(),
            "Recorder symbolic storage needs a power-of-two bank matching its address width"
        );
        let mut level = storage.to_vec();
        for address_bit in address {
            level = level
                .chunks_exact(2)
                .map(|pair| self.mux(address_bit.wire, pair[1], pair[0]))
                .collect::<Result<Vec<_>, _>>()?;
        }
        Ok(level[0])
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Idx>],
        value: Idx,
    ) -> Result<(), Self::Error> {
        if let Some(index) = recorded_storage_index(address) {
            storage[index] = value;
            return Ok(());
        }
        assert!(
            storage.len().is_power_of_two() && (1usize << address.len()) == storage.len(),
            "Recorder symbolic storage needs a power-of-two bank matching its address width"
        );
        for (index, cell) in storage.iter_mut().enumerate() {
            let selected = self.recorded_storage_selector(address, index)?;
            *cell = self.mux(selected, value, *cell)?;
        }
        Ok(())
    }
}

impl Recorder {
    fn recorded_storage_selector(
        &mut self,
        address: &[StorageAddressBit<Idx>],
        index: usize,
    ) -> Result<Idx, Infallible> {
        let mut selected = self.create(true)?;
        for (bit, address_bit) in address.iter().enumerate() {
            let expected = (index >> bit) & 1 != 0;
            let matches = match address_bit.known {
                Some(known) if known == expected => continue,
                Some(_) => return self.create(false),
                None if expected => address_bit.wire,
                None => {
                    let one = self.create(true)?;
                    self.bitxor(address_bit.wire, one)?
                }
            };
            selected = self.bitand(selected, matches)?;
        }
        Ok(selected)
    }
}

fn recorded_storage_index(address: &[StorageAddressBit<Idx>]) -> Option<usize> {
    let mut index = 0usize;
    for (bit, address_bit) in address.iter().enumerate() {
        match address_bit.known {
            Some(true) => index |= 1usize.checked_shl(bit as u32)?,
            Some(false) => {}
            None => return None,
        }
    }
    Some(index)
}

impl HasError for PreparedRecorder {
    type Error = Infallible;
}

impl ContextWithValue<bool> for PreparedRecorder {
    type Wrapped = Idx;
}

impl ContextWithCreate<bool> for PreparedRecorder {
    fn create(&mut self, val: bool) -> Result<Idx, Infallible> {
        self.recorder.create(val)
    }
}

impl ContextWithBitAnd<bool> for PreparedRecorder {
    fn bitand(&mut self, a: Idx, b: Idx) -> Result<Idx, Infallible> {
        self.recorder.bitand(a, b)
    }

    fn bitand_assign(&mut self, a: &mut Idx, b: Idx) -> Result<(), Infallible> {
        self.recorder.bitand_assign(a, b)
    }
}

impl ContextWithBitOr<bool> for PreparedRecorder {
    fn bitor(&mut self, a: Idx, b: Idx) -> Result<Idx, Infallible> {
        self.recorder.bitor(a, b)
    }

    fn bitor_assign(&mut self, a: &mut Idx, b: Idx) -> Result<(), Infallible> {
        self.recorder.bitor_assign(a, b)
    }
}

impl ContextWithBitXor<bool> for PreparedRecorder {
    fn bitxor(&mut self, a: Idx, b: Idx) -> Result<Idx, Infallible> {
        self.recorder.bitxor(a, b)
    }

    fn bitxor_assign(&mut self, a: &mut Idx, b: Idx) -> Result<(), Infallible> {
        self.recorder.bitxor_assign(a, b)
    }
}

impl ContextWithMux<bool> for PreparedRecorder {
    fn mux(&mut self, cond: Idx, then: Idx, r#else: Idx) -> Result<Idx, Infallible> {
        self.recorder.mux(cond, then, r#else)
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
            Op::External(_) => panic!("interpret: external program needs an external registry"),
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
    program
        .validate()
        .expect("prepared program must satisfy structural invariants");
    assert_eq!(
        inputs.len(),
        program.inputs.len(),
        "input count must match the recorded program's input slots"
    );
    let mut slots: Vec<Option<bool>> = alloc::vec![None; program.slots];
    let mut is_input = alloc::vec![false; program.slots];
    for &idx in &program.inputs {
        is_input[idx.get()] = true;
    }
    for (&idx, &value) in program.inputs.iter().zip(inputs) {
        slots[idx.get()] = Some(value);
    }
    interpret_range(
        program,
        program.entry,
        &mut slots,
        &is_input,
        &mut Vec::new(),
        0,
    );
    program
        .outputs
        .iter()
        .map(|idx| slots[idx.get()].unwrap())
        .collect()
}

struct ActiveLoop<'a> {
    loop_step: &'a PreparedLoop,
    row: usize,
}

fn interpret_range<'a>(
    program: &'a PreparedProgram,
    range: StatementRange,
    slots: &mut [Option<bool>],
    is_input: &[bool],
    active: &mut Vec<ActiveLoop<'a>>,
    invocation: usize,
) {
    for statement in program
        .range_slice(range)
        .expect("validated prepared range must be present")
    {
        match statement {
            Statement::Op(op) => interpret_scheduled_op(
                slots,
                is_input,
                op.resolve(|slot| resolve_slot(slot, active)),
            ),
            Statement::Loop(loop_step) => {
                let descriptor = loop_step.invocations[invocation];
                for iteration in 0..descriptor.iterations as usize {
                    let row = descriptor.first_row as usize + iteration;
                    active.push(ActiveLoop { loop_step, row });
                    interpret_range(program, loop_step.body, slots, is_input, active, row);
                    active.pop();
                }
            }
        }
    }
}

fn resolve_slot(slot: PreparedSlot, active: &[ActiveLoop<'_>]) -> Idx {
    match slot {
        PreparedSlot::Static(slot) => slot,
        PreparedSlot::Table { depth, field } => {
            let active = &active[active.len() - 1 - depth as usize];
            let offset = active
                .row
                .checked_mul(active.loop_step.fields_per_iteration as usize)
                .and_then(|offset| offset.checked_add(field as usize))
                .expect("validated prepared table offset must fit usize");
            Idx(active.loop_step.table[offset])
        }
    }
}

fn interpret_scheduled_op(slots: &mut [Option<bool>], is_input: &[bool], scheduled: ScheduledOp) {
    if is_input[scheduled.out.get()] {
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
        Op::External(_) => {
            panic!("interpret_prepared: external program needs an external registry")
        }
    };
    slots[scheduled.out.get()] = Some(value);
}

/// Per-slot liveness, in units of the virtual-time ticks [`compute_liveness`]
/// assigns while walking a [`PreparedProgram`] in its canonical execution
/// order (one tick per [`Statement::Op`] visited, including every loop
/// iteration/row).
struct SlotLiveness {
    /// First tick each absolute slot is defined, `None` if a slot's only
    /// defining statement sits in a loop body no invocation ever runs.
    def: Vec<Option<u32>>,
    /// Last tick each absolute slot is read or written. A declared input
    /// starts at tick `0`; a declared output is pinned to `u32::MAX` so it
    /// always outlives every computed value.
    last_ref: Vec<u32>,
}

/// A slot's physical-slot reassignment, from [`assign_compact_slots`].
struct SlotAssignment {
    /// `map[old.get()]` is the compacted absolute slot for original slot
    /// `old`. Every declared input maps into a permanently reserved prefix
    /// `0..inputs.len()`, in declared order; no other value is ever assigned
    /// into that prefix.
    map: Vec<u32>,
    /// The compacted program's new [`PreparedProgram::slots`] width.
    new_slots: usize,
}

impl PreparedProgram {
    /// Reassign scratch-buffer slots so that values with disjoint live
    /// ranges share one physical slot, shrinking [`PreparedProgram::slots`]
    /// where profitable.
    ///
    /// This is a single-shot pass, not a fixed point, and it never runs
    /// implicitly inside [`Program::prepare`] or
    /// [`PreparedProgram::reoptimize`] -- call it explicitly, and only after
    /// loop abstraction has already reached its own fixed point. Running it
    /// first would scramble the positional-arithmetic regularity
    /// `classify_slot` depends on to recognize loop-eligible repetition, so
    /// [`Program::compact`] always calls [`Program::prepare`] first.
    ///
    /// Every declared input keeps one physical slot reserved for the whole
    /// execution, renumbered into a tight `0..inputs.len()` prefix; only
    /// computed values participate in reuse. This is what lets every
    /// existing backend consume a compacted program with no changes: each
    /// backend's own "skip emitting a write to a declared input slot" check
    /// stays correct exactly because no computed value is ever assigned an
    /// input's slot.
    pub fn compact_slots(&self) -> Self {
        let liveness = self.compute_liveness();
        let assignment = self.assign_compact_slots(&liveness);
        self.rewrite_with(&assignment)
    }

    fn compute_liveness(&self) -> SlotLiveness {
        let mut liveness = SlotLiveness {
            def: alloc::vec![None; self.slots],
            last_ref: alloc::vec![0; self.slots],
        };
        for &input in &self.inputs {
            liveness.def[input.get()] = Some(0);
            liveness.last_ref[input.get()] = 0;
        }
        let mut tick = 0u32;
        self.liveness_range(self.entry, &mut Vec::new(), 0, &mut tick, &mut liveness);
        for &output in &self.outputs {
            liveness.last_ref[output.get()] = u32::MAX;
        }
        liveness
    }

    fn liveness_range<'a>(
        &'a self,
        range: StatementRange,
        active: &mut Vec<ActiveLoop<'a>>,
        invocation: usize,
        tick: &mut u32,
        liveness: &mut SlotLiveness,
    ) {
        for statement in self
            .range_slice(range)
            .expect("validated prepared range must be present")
        {
            match statement {
                Statement::Op(op) => {
                    *tick += 1;
                    let scheduled = op.resolve(|slot| resolve_slot(slot, active));
                    match scheduled.op {
                        Op::External(external) => {
                            for operand in &self.externals[external as usize].args {
                                liveness.last_ref[operand.get()] =
                                    liveness.last_ref[operand.get()].max(*tick);
                            }
                        }
                        op => {
                            for operand in op_operand_slots(op).into_iter().flatten() {
                                liveness.last_ref[operand.get()] =
                                    liveness.last_ref[operand.get()].max(*tick);
                            }
                        }
                    }
                    let out = scheduled.out.get();
                    if liveness.def[out].is_none() {
                        liveness.def[out] = Some(*tick);
                    }
                    liveness.last_ref[out] = liveness.last_ref[out].max(*tick);
                }
                Statement::Loop(loop_step) => {
                    let descriptor = loop_step.invocations[invocation];
                    for iteration in 0..descriptor.iterations as usize {
                        let row = descriptor.first_row as usize + iteration;
                        active.push(ActiveLoop { loop_step, row });
                        self.liveness_range(loop_step.body, active, row, tick, liveness);
                        active.pop();
                    }
                }
            }
        }
    }

    fn assign_compact_slots(&self, liveness: &SlotLiveness) -> SlotAssignment {
        let mut map = alloc::vec![u32::MAX; self.slots];
        for (position, &input) in self.inputs.iter().enumerate() {
            map[input.get()] = position as u32;
        }
        let mut next_fresh = self.inputs.len() as u32;
        let mut free = BinaryHeap::<Reverse<u32>>::new();
        let mut active = BinaryHeap::<Reverse<(u32, u32)>>::new();

        let mut computed = (0..self.slots)
            .filter(|&slot| map[slot] == u32::MAX)
            .map(|slot| {
                (
                    slot,
                    liveness.def[slot].unwrap_or(0),
                    liveness.last_ref[slot],
                )
            })
            .collect::<Vec<_>>();
        computed.sort_by_key(|&(_, def, _)| def);

        for (slot, def, last_ref) in computed {
            while let Some(&Reverse((earliest_last_ref, phys))) = active.peek() {
                if earliest_last_ref < def {
                    active.pop();
                    free.push(Reverse(phys));
                } else {
                    break;
                }
            }
            let phys = match free.pop() {
                Some(Reverse(phys)) => phys,
                None => {
                    let phys = next_fresh;
                    next_fresh += 1;
                    phys
                }
            };
            map[slot] = phys;
            active.push(Reverse((last_ref, phys)));
        }

        SlotAssignment {
            map,
            new_slots: next_fresh as usize,
        }
    }

    fn rewrite_with(&self, assignment: &SlotAssignment) -> Self {
        let mut statements = self.statements.clone();
        for statement in &mut statements {
            match statement {
                Statement::Op(op) => *op = remap_prepared_op(*op, &assignment.map),
                Statement::Loop(loop_step) => {
                    for slot in &mut loop_step.table {
                        *slot = assignment.map[*slot as usize];
                    }
                }
            }
        }
        let inputs = self
            .inputs
            .iter()
            .map(|idx| Idx(assignment.map[idx.get()]))
            .collect();
        let outputs = self
            .outputs
            .iter()
            .map(|idx| Idx(assignment.map[idx.get()]))
            .collect();
        let mut externals = self.externals.clone();
        for external in &mut externals {
            for arg in &mut external.args {
                *arg = Idx(assignment.map[arg.get()]);
            }
        }
        let mut compacted = Self {
            slots: assignment.new_slots,
            inputs,
            outputs,
            externals,
            entry: self.entry,
            statements,
            // Compaction never changes how many operations a fully unrolled
            // execution would perform -- only how many physical slots back
            // them -- so the unrolled-byte estimate carries over unchanged;
            // recomputing it from the new (smaller) `slots` via
            // `refresh_estimate` would understate it.
            estimated_unrolled_bytes: self.estimated_unrolled_bytes,
            estimated_prepared_bytes: 0,
        };
        compacted.estimated_prepared_bytes = compacted.estimate_range(compacted.entry);
        compacted
            .validate()
            .expect("slot compaction must preserve prepared-program structural invariants");
        compacted
    }
}

fn op_operand_slots(op: Op) -> [Option<Idx>; 3] {
    match op {
        Op::Create(_) => [None, None, None],
        Op::BitAnd(a, b) | Op::BitOr(a, b) | Op::BitXor(a, b) => [Some(a), Some(b), None],
        Op::Mux { cond, then, r#else } => [Some(cond), Some(then), Some(r#else)],
        Op::External(_) => [None, None, None],
    }
}

fn remap_prepared_slot(slot: PreparedSlot, map: &[u32]) -> PreparedSlot {
    match slot {
        PreparedSlot::Static(idx) => PreparedSlot::Static(Idx(map[idx.get()])),
        table @ PreparedSlot::Table { .. } => table,
    }
}

fn remap_prepared_op(op: PreparedOp, map: &[u32]) -> PreparedOp {
    match op {
        PreparedOp::Create { value, out } => PreparedOp::Create {
            value,
            out: remap_prepared_slot(out, map),
        },
        PreparedOp::BitAnd { a, b, out } => PreparedOp::BitAnd {
            a: remap_prepared_slot(a, map),
            b: remap_prepared_slot(b, map),
            out: remap_prepared_slot(out, map),
        },
        PreparedOp::BitOr { a, b, out } => PreparedOp::BitOr {
            a: remap_prepared_slot(a, map),
            b: remap_prepared_slot(b, map),
            out: remap_prepared_slot(out, map),
        },
        PreparedOp::BitXor { a, b, out } => PreparedOp::BitXor {
            a: remap_prepared_slot(a, map),
            b: remap_prepared_slot(b, map),
            out: remap_prepared_slot(out, map),
        },
        PreparedOp::Mux {
            cond,
            then,
            r#else,
            out,
        } => PreparedOp::Mux {
            cond: remap_prepared_slot(cond, map),
            then: remap_prepared_slot(then, map),
            r#else: remap_prepared_slot(r#else, map),
            out: remap_prepared_slot(out, map),
        },
        PreparedOp::External { external, out } => PreparedOp::External {
            external,
            out: remap_prepared_slot(out, map),
        },
    }
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

    /// Two nested loops where the inner loop's trip count varies per outer
    /// row (row 0 runs once, row 1 runs twice). Every computed slot (2..6)
    /// is declared as a program output, so nothing in this particular
    /// fixture is ever reusable -- it exists to exercise nested,
    /// variable-trip-count loop structure, not slot reuse.
    fn nested_variable_trip_count_fixture() -> (Program, PreparedProgram) {
        let raw = Program {
            ops: alloc::vec![
                Op::Create(false),
                Op::Create(false),
                Op::BitXor(Idx(0), Idx(1)),
                Op::BitXor(Idx(0), Idx(1)),
                Op::BitAnd(Idx(2), Idx(1)),
                Op::BitAnd(Idx(3), Idx(1)),
                Op::BitAnd(Idx(3), Idx(1)),
            ],
            inputs: alloc::vec![Idx(0), Idx(1)],
            outputs: alloc::vec![Idx(2), Idx(3), Idx(4), Idx(5), Idx(6)],
            externals: alloc::vec![],
        };
        let prepared = PreparedProgram::new(
            raw.len(),
            raw.inputs.clone(),
            raw.outputs.clone(),
            StatementRange::new(0, 1),
            alloc::vec![
                Statement::Loop(PreparedLoop {
                    body: StatementRange::new(1, 3),
                    invocations: alloc::vec![LoopInvocation {
                        first_row: 0,
                        iterations: 2,
                    }],
                    fields_per_iteration: 1,
                    table: alloc::vec![2, 3],
                }),
                Statement::Op(PreparedOp::BitXor {
                    a: PreparedSlot::Static(Idx(0)),
                    b: PreparedSlot::Static(Idx(1)),
                    out: PreparedSlot::Table { depth: 0, field: 0 },
                }),
                Statement::Loop(PreparedLoop {
                    body: StatementRange::new(3, 4),
                    // One descriptor for each outer loop row: the first
                    // inner execution runs once and the second runs twice.
                    invocations: alloc::vec![
                        LoopInvocation {
                            first_row: 0,
                            iterations: 1,
                        },
                        LoopInvocation {
                            first_row: 1,
                            iterations: 2,
                        },
                    ],
                    fields_per_iteration: 1,
                    table: alloc::vec![4, 5, 6],
                }),
                Statement::Op(PreparedOp::BitAnd {
                    a: PreparedSlot::Table { depth: 1, field: 0 },
                    b: PreparedSlot::Static(Idx(1)),
                    out: PreparedSlot::Table { depth: 0, field: 0 },
                }),
            ],
        )
        .unwrap();
        (raw, prepared)
    }

    #[test]
    fn nested_loops_support_a_different_inner_trip_count_per_outer_row() {
        let (raw, prepared) = nested_variable_trip_count_fixture();

        assert_eq!(
            interpret(&raw, &[true, false]),
            interpret_prepared(&prepared, &[true, false])
        );
        let reoptimized = prepared.reoptimize(&OptimizationOptions::default());
        assert_eq!(
            reoptimized,
            reoptimized.reoptimize(&OptimizationOptions::default())
        );
        assert_eq!(
            interpret_prepared(&prepared, &[false, true]),
            interpret_prepared(&reoptimized, &[false, true])
        );
    }

    #[test]
    fn prepared_validation_rejects_bad_slots_and_bad_nested_descriptor_counts() {
        let invalid_slot = PreparedProgram::new(
            1,
            Vec::new(),
            Vec::new(),
            StatementRange::new(0, 1),
            alloc::vec![Statement::Op(PreparedOp::Create {
                value: true,
                out: PreparedSlot::Static(Idx(1)),
            })],
        );
        assert_eq!(invalid_slot, Err(PreparedProgramError::InvalidSlot));

        let invalid_nested_count = PreparedProgram::new(
            1,
            Vec::new(),
            Vec::new(),
            StatementRange::new(0, 1),
            alloc::vec![
                Statement::Loop(PreparedLoop {
                    body: StatementRange::new(1, 2),
                    invocations: alloc::vec![LoopInvocation {
                        first_row: 0,
                        iterations: 2,
                    }],
                    fields_per_iteration: 0,
                    table: Vec::new(),
                }),
                Statement::Loop(PreparedLoop {
                    body: StatementRange::new(2, 2),
                    invocations: alloc::vec![LoopInvocation {
                        first_row: 0,
                        iterations: 0,
                    }],
                    fields_per_iteration: 0,
                    table: Vec::new(),
                }),
            ],
        );
        assert_eq!(
            invalid_nested_count,
            Err(PreparedProgramError::InvalidInvocationCount)
        );
    }

    #[test]
    fn reoptimization_merges_adjacent_nested_loop_segments() {
        let prepared = PreparedProgram::new(
            7,
            alloc::vec![Idx(0), Idx(1)],
            alloc::vec![Idx(2), Idx(3), Idx(4), Idx(5), Idx(6)],
            StatementRange::new(0, 2),
            alloc::vec![
                Statement::Loop(PreparedLoop {
                    body: StatementRange::new(2, 4),
                    invocations: alloc::vec![LoopInvocation {
                        first_row: 0,
                        iterations: 1,
                    }],
                    fields_per_iteration: 1,
                    table: alloc::vec![2],
                }),
                Statement::Loop(PreparedLoop {
                    body: StatementRange::new(4, 6),
                    invocations: alloc::vec![LoopInvocation {
                        first_row: 0,
                        iterations: 1,
                    }],
                    fields_per_iteration: 1,
                    table: alloc::vec![3],
                }),
                Statement::Op(PreparedOp::BitXor {
                    a: PreparedSlot::Static(Idx(0)),
                    b: PreparedSlot::Static(Idx(1)),
                    out: PreparedSlot::Table { depth: 0, field: 0 },
                }),
                Statement::Loop(PreparedLoop {
                    body: StatementRange::new(6, 7),
                    invocations: alloc::vec![LoopInvocation {
                        first_row: 0,
                        iterations: 1,
                    }],
                    fields_per_iteration: 1,
                    table: alloc::vec![4],
                }),
                Statement::Op(PreparedOp::BitXor {
                    a: PreparedSlot::Static(Idx(0)),
                    b: PreparedSlot::Static(Idx(1)),
                    out: PreparedSlot::Table { depth: 0, field: 0 },
                }),
                Statement::Loop(PreparedLoop {
                    body: StatementRange::new(7, 8),
                    invocations: alloc::vec![LoopInvocation {
                        first_row: 0,
                        iterations: 2,
                    }],
                    fields_per_iteration: 1,
                    table: alloc::vec![5, 6],
                }),
                Statement::Op(PreparedOp::BitAnd {
                    a: PreparedSlot::Table { depth: 1, field: 0 },
                    b: PreparedSlot::Static(Idx(1)),
                    out: PreparedSlot::Table { depth: 0, field: 0 },
                }),
                Statement::Op(PreparedOp::BitAnd {
                    a: PreparedSlot::Table { depth: 1, field: 0 },
                    b: PreparedSlot::Static(Idx(1)),
                    out: PreparedSlot::Table { depth: 0, field: 0 },
                }),
            ],
        )
        .unwrap();
        let optimized = prepared.reoptimize(&OptimizationOptions::default());

        assert_eq!(optimized.entry.end - optimized.entry.start, 1);
        let Statement::Loop(outer) = &optimized.statements[optimized.entry.start as usize] else {
            panic!("adjacent outer loops should merge");
        };
        assert_eq!(outer.invocations[0].iterations, 2);
        let Statement::Loop(inner) = &optimized.statements[outer.body.start as usize + 1] else {
            panic!("the merged body should retain its inner loop");
        };
        assert_eq!(inner.invocations.len(), 2);
        assert_eq!(inner.invocations[0].iterations, 1);
        assert_eq!(inner.invocations[1].iterations, 2);
        assert_eq!(
            interpret_prepared(&prepared, &[true, false]),
            interpret_prepared(&optimized, &[true, false])
        );
    }

    #[test]
    fn write_once_guard_still_protects_inputs_after_static_check_swap() {
        let raw = Program {
            ops: alloc::vec![Op::Create(false)],
            inputs: alloc::vec![Idx(0)],
            outputs: alloc::vec![Idx(0)],
            externals: alloc::vec![],
        };
        let prepared = raw.prepare(&OptimizationOptions::default());
        assert_eq!(interpret_prepared(&prepared, &[true]), [true]);
    }

    #[test]
    fn compact_slots_reduces_width_and_preserves_semantics_for_flat_program() {
        let mut recorder = Recorder::new();
        let a = recorder.create(false).unwrap();
        let b = recorder.create(false).unwrap();
        let c = recorder.create(false).unwrap();
        let d = recorder.create(false).unwrap();
        // x1 dies as soon as x2 reads it, before x3 is even computed, so x3
        // (an unrelated, independent chain) should be able to reuse x1's
        // physical slot after compaction.
        let x1 = ContextWithBitAnd::bitand(&mut recorder, a, b).unwrap();
        let x2 = ContextWithBitXor::bitxor(&mut recorder, x1, c).unwrap();
        let x3 = ContextWithBitAnd::bitand(&mut recorder, c, d).unwrap();
        let out = ContextWithBitOr::bitor(&mut recorder, x2, x3).unwrap();
        let program = recorder.finish(alloc::vec![a, b, c, d], alloc::vec![out]);
        let prepared = program.prepare(&OptimizationOptions::default());
        let compacted = prepared.compact_slots();

        assert!(compacted.slots < prepared.slots);
        for &(va, vb, vc, vd) in &[
            (false, false, false, false),
            (true, false, true, false),
            (true, true, false, true),
            (false, true, true, true),
        ] {
            let inputs = [va, vb, vc, vd];
            assert_eq!(
                interpret_prepared(&prepared, &inputs),
                interpret_prepared(&compacted, &inputs)
            );
        }
    }

    #[test]
    fn compact_slots_reuses_slots_inside_a_loop_body() {
        let mut recorder = Recorder::new();
        let c = recorder.create(false).unwrap();
        let mut a = Vec::new();
        let mut b = Vec::new();
        for _ in 0..8 {
            a.push(recorder.create(false).unwrap());
            b.push(recorder.create(false).unwrap());
        }
        let mut outputs = Vec::new();
        for i in 0..8 {
            // `t` never survives past its own iteration, so every
            // iteration's `t` should collapse onto one reused physical
            // slot; `out[i]` is a declared output and must not be reused.
            let t = ContextWithBitAnd::bitand(&mut recorder, a[i], b[i]).unwrap();
            let out = ContextWithBitXor::bitxor(&mut recorder, t, c).unwrap();
            outputs.push(out);
        }
        let mut inputs = alloc::vec![c];
        inputs.extend(a.iter().copied());
        inputs.extend(b.iter().copied());
        let program = recorder.finish(inputs, outputs);
        let prepared = program.prepare(&OptimizationOptions::default());
        assert!(prepared.has_loops());

        let compacted = prepared.compact_slots();
        assert!(compacted.has_loops());
        assert!(compacted.slots < prepared.slots);

        let sample = (0..17).map(|i| i % 3 == 0).collect::<Vec<_>>();
        assert_eq!(
            interpret_prepared(&prepared, &sample),
            interpret_prepared(&compacted, &sample)
        );
    }

    #[test]
    fn compact_slots_reuses_slots_across_nested_loop_rows_with_different_trip_counts() {
        let (_, prepared) = nested_variable_trip_count_fixture();
        let compacted = prepared.compact_slots();

        // Every computed slot in this fixture is a declared output (see the
        // fixture's own doc comment), so nothing here is actually
        // reusable -- this test's job is confirming compaction is a
        // correctness-preserving no-op under nested, variable-trip-count
        // loops. `compact_slots_reuses_slots_inside_a_loop_body` above
        // covers the case where reuse actually happens.
        assert_eq!(compacted.slots, prepared.slots);
        for &(a, b) in &[(true, false), (false, true), (true, true), (false, false)] {
            assert_eq!(
                interpret_prepared(&prepared, &[a, b]),
                interpret_prepared(&compacted, &[a, b])
            );
        }
    }

    #[test]
    fn compact_slots_preserves_input_and_output_identity_order() {
        let mut recorder = Recorder::new();
        let a = recorder.create(false).unwrap();
        let b = recorder.create(false).unwrap();
        let c = recorder.create(false).unwrap();
        let and = ContextWithBitAnd::bitand(&mut recorder, a, b).unwrap();
        let or = ContextWithBitOr::bitor(&mut recorder, a, b).unwrap();
        let xor = ContextWithBitXor::bitxor(&mut recorder, b, c).unwrap();
        // Outputs declared out of definition order, and not every computed
        // value is an output (`or` is scratch-only).
        let program = recorder.finish(alloc::vec![a, b, c], alloc::vec![xor, and, or]);
        let prepared = program.prepare(&OptimizationOptions::default());
        let compacted = prepared.compact_slots();

        for &(va, vb, vc) in &[
            (false, false, false),
            (true, false, true),
            (true, true, false),
            (false, true, true),
        ] {
            let inputs = [va, vb, vc];
            assert_eq!(
                interpret_prepared(&prepared, &inputs),
                interpret_prepared(&compacted, &inputs)
            );
        }
    }

    #[test]
    fn compact_slots_never_shares_an_input_slot_with_any_other_def() {
        let mut recorder = Recorder::new();
        let a = recorder.create(false).unwrap();
        let b = recorder.create(false).unwrap();
        // `a`'s real last use is early (only read once, right away), which
        // is exactly the case that could tempt an allocator into reusing
        // its slot for something else if inputs weren't pinned.
        let and = ContextWithBitAnd::bitand(&mut recorder, a, b).unwrap();
        let xor = ContextWithBitXor::bitxor(&mut recorder, and, b).unwrap();
        let program = recorder.finish(alloc::vec![a, b], alloc::vec![xor]);
        let prepared = program.prepare(&OptimizationOptions::default());

        let liveness = prepared.compute_liveness();
        let assignment = prepared.assign_compact_slots(&liveness);
        let input_phys_slots = prepared
            .inputs
            .iter()
            .map(|idx| assignment.map[idx.get()])
            .collect::<Vec<_>>();
        for slot in 0..prepared.slots {
            if prepared.inputs.iter().any(|idx| idx.get() == slot) {
                continue;
            }
            assert!(!input_phys_slots.contains(&assignment.map[slot]));
        }
    }

    #[test]
    fn compact_slots_is_a_no_op_shape_when_nothing_is_reusable() {
        let mut recorder = Recorder::new();
        let a = recorder.create(false).unwrap();
        let b = recorder.create(false).unwrap();
        let c = recorder.create(false).unwrap();
        // Three independent chains, all declared as outputs -- every
        // computed value is live through to the very end, so there is
        // nothing for compaction to reuse.
        let and = ContextWithBitAnd::bitand(&mut recorder, a, b).unwrap();
        let or = ContextWithBitOr::bitor(&mut recorder, b, c).unwrap();
        let xor = ContextWithBitXor::bitxor(&mut recorder, a, c).unwrap();
        let program = recorder.finish(alloc::vec![a, b, c], alloc::vec![and, or, xor]);
        let prepared = program.prepare(&OptimizationOptions::default());
        let compacted = prepared.compact_slots();

        assert_eq!(compacted.slots, prepared.slots);
    }

    #[test]
    fn compact_slots_handles_a_loop_with_zero_iterations_without_panicking() {
        let prepared = PreparedProgram::new(
            2,
            Vec::new(),
            Vec::new(),
            StatementRange::new(0, 1),
            alloc::vec![
                Statement::Loop(PreparedLoop {
                    body: StatementRange::new(1, 2),
                    invocations: alloc::vec![LoopInvocation {
                        first_row: 0,
                        iterations: 0,
                    }],
                    fields_per_iteration: 1,
                    table: Vec::new(),
                }),
                Statement::Op(PreparedOp::Create {
                    value: true,
                    out: PreparedSlot::Table { depth: 0, field: 0 },
                }),
            ],
        )
        .unwrap();

        let compacted = prepared.compact_slots();
        assert!(compacted.validate().is_ok());
    }

    #[test]
    fn preparation_modes_are_zero_sized_and_preserve_raw_compatibility() {
        assert_eq!(core::mem::size_of::<NoPreparation>(), 0);
        assert_eq!(core::mem::size_of::<CountdownPreparation>(), 0);

        let mut recorder = Recorder::new();
        let input = recorder.create(false).unwrap();
        let raw = recorder.finish_with::<NoPreparation>(
            alloc::vec![input],
            alloc::vec![input],
            &OptimizationOptions::default(),
        );
        assert_eq!(interpret(&raw, &[true]), [true]);
    }
}
