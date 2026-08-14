//! Lower a `cirrus_recompile_core::Program` to real AArch64 machine code.
//!
//! The emitted function has the shape `void {name}(void *backend, void
//! *buf)` under AAPCS64: given a pointer to a backend instance and a pointer
//! to a `program.ops.len()`-element scratch buffer, it performs the
//! recorded trace by calling out to that backend's pinned runtime functions
//! -- the exact same functions the LLVM and Rust backends call by name --
//! passing buffer-slot indices as arguments, exactly as
//! `cirrus-recompile-core`'s module documentation describes. The caller
//! supplies each pinned function's address ([`PinnedAddresses`]); this crate
//! does not depend on any particular `cirrus-recompile-rt` backend, so the
//! *same* emitted machine code calls a native `bool` context, a
//! garbled-circuit garbler, or an evaluator depending only on which
//! addresses `pinned` names.
//!
//! Every call site loads its pinned function's address into a scratch
//! register (`mov_imm` + indirect `bl`) rather than emitting a direct
//! PC-relative branch, so the emitted code has no relocations to resolve and
//! can be copied into freshly allocated memory and executed immediately.
//! `X19`/`X20` (callee-saved) hold `backend`/`buf` across calls, since a
//! pinned function is free to clobber its own first two argument registers
//! (`X0`/`X1`) like any other AAPCS64 callee.
//!
//! # Register allocation
//!
//! Every operand this backend's instructions touch is either the constant
//! buffer-slot index baked into a single call site (materialized fresh via
//! `mov_imm` every time -- there is no cross-call liveness to manage) or one
//! of the two long-lived `backend`/`buf` pointers kept in `X19`/`X20` for the
//! whole function. At this granularity -- one pinned-function call per
//! Boolean op -- there is genuinely nothing to spill: the "regalloc over
//! indices instead of values" goal is visible instead in
//! `portal-solutions-asm-regalloc`'s suitability for the natural follow-on
//! optimization this module intentionally leaves undone, caching a
//! recently-produced slot's *value* in a register across adjacent calls to
//! skip a `mov_imm`+call round trip. See `regalloc_over_indices` in this
//! crate's tests for a worked demonstration of that allocator doing exactly
//! this bookkeeping over a stream of pushed and popped
//! [`Idx`](cirrus_recompile_core::Idx) values.

extern crate alloc;

use alloc::vec::Vec;

use cirrus_recompile_core::{
    LoopOp, LoopSlot, OptimizationOptions, PreparedProgram, PreparedStep, Program, ScheduledOp,
    TableLoop,
};
use portal_pc_asm_common::types::{mem::MemorySize, reg::Reg};
use portal_solutions_asm_aarch64::{
    out::{
        arg::{AddressingMode, ArgKind, MemArgKind},
        bin::AArch64Writer,
        Writer, WriterCore,
    },
    AArch64Arch, ConditionCode, RegisterClass,
};

/// The address of each pinned runtime function this backend calls, by exact
/// name -- matching whichever `cirrus-recompile-rt` backend module (e.g.
/// `plaintext`, `gc`, `eval`) the caller targets.
pub struct PinnedAddresses {
    /// `create(backend, buf, val, out)`.
    pub create: usize,
    /// `bitand(backend, buf, a, b, out)`.
    pub bitand: usize,
    /// `bitor(backend, buf, a, b, out)`.
    pub bitor: usize,
    /// `bitxor(backend, buf, a, b, out)`.
    pub bitxor: usize,
    /// `mux(backend, buf, cond, then, r#else, out)`.
    pub mux: usize,
}

const BACKEND_ARG: u8 = 0; // X0: this function's own `backend` argument.
const BUF_ARG: u8 = 1; // X1: this function's own `buf` argument.
const BACKEND_SAVE: u8 = 19; // X19: callee-saved copy of `backend`.
const BUF_SAVE: u8 = 20; // X20: callee-saved copy of `buf`.
const TABLE_BASE_SAVE: u8 = 21; // X21: callee-saved current table base.
const ITERATION_SAVE: u8 = 22; // X22: callee-saved current table row.
const ADDR_SCRATCH: u8 = 9; // X9: scratch for a pinned function's address.
const TABLE_OFFSET_SCRATCH: u8 = 10; // X10: scratch while loading a table field.
const ARG_REGS: [u8; 4] = [2, 3, 4, 5]; // X2..X5: op-specific pinned-function arguments.

fn reg(index: u8, size: MemorySize) -> MemArgKind<ArgKind> {
    MemArgKind::NoMem(ArgKind::Reg {
        reg: Reg(index),
        size,
    })
}

fn pair_stack(disp: i32, mode: AddressingMode) -> MemArgKind<ArgKind> {
    MemArgKind::Mem {
        base: ArgKind::Reg {
            reg: Reg(31),
            size: MemorySize::_64,
        },
        offset: None,
        disp,
        size: MemorySize::_128,
        reg_class: RegisterClass::Gpr,
        mode,
    }
}

fn mem(base: u8, disp: i32, size: MemorySize) -> MemArgKind<ArgKind> {
    MemArgKind::Mem {
        base: ArgKind::Reg {
            reg: Reg(base),
            size: MemorySize::_64,
        },
        offset: None,
        disp,
        size,
        reg_class: RegisterClass::Gpr,
        mode: AddressingMode::Offset,
    }
}

fn literal(value: u64) -> MemArgKind<ArgKind> {
    MemArgKind::NoMem(ArgKind::Lit(value))
}

fn emit_call<L>(w: &mut AArch64Writer<L>, addr: usize, args: &[u64]) {
    let arch = AArch64Arch::default();
    let mut ctx = ();
    // Every pinned function is a normal AAPCS64 callee: it is free to
    // clobber X0/X1, so restore `backend`/`buf` from X19/X20 immediately
    // before each call.
    w.mov(
        &mut ctx,
        arch,
        &reg(BACKEND_ARG, MemorySize::_64),
        &reg(BACKEND_SAVE, MemorySize::_64),
    )
    .unwrap();
    w.mov(
        &mut ctx,
        arch,
        &reg(BUF_ARG, MemorySize::_64),
        &reg(BUF_SAVE, MemorySize::_64),
    )
    .unwrap();
    for (&argreg, &value) in ARG_REGS.iter().zip(args) {
        w.mov_imm(&mut ctx, arch, &reg(argreg, MemorySize::_64), value)
            .unwrap();
    }
    w.mov_imm(
        &mut ctx,
        arch,
        &reg(ADDR_SCRATCH, MemorySize::_64),
        addr as u64,
    )
    .unwrap();
    w.bl(&mut ctx, arch, &reg(ADDR_SCRATCH, MemorySize::_64))
        .unwrap();
}

/// Emit an executable AArch64 function performing `program`'s recorded
/// trace, calling out to `pinned`'s functions. Returns the encoded
/// instruction bytes (position-independent: no relocations remain).
pub fn compile_aarch64(program: &Program, pinned: &PinnedAddresses) -> Vec<u8> {
    compile_aarch64_with_options(program, pinned, &OptimizationOptions::default())
}

/// Prepare and lower a program with explicit reabstraction options.
pub fn compile_aarch64_with_options(
    program: &Program,
    pinned: &PinnedAddresses,
    options: &OptimizationOptions,
) -> Vec<u8> {
    let prepared = program.prepare(options);
    compile_prepared_aarch64(&prepared, pinned)
}

/// Lower a previously prepared program to AArch64 machine code.
///
/// Table-loop rows are appended after the return instruction as immutable
/// `u32` data.  Each loop uses `ADR` to obtain its table base and executes one
/// copy of the operation template per row, preserving the pinned-call ABI.
pub fn compile_prepared_aarch64(program: &PreparedProgram, pinned: &PinnedAddresses) -> Vec<u8> {
    let arch = AArch64Arch::default();
    let mut ctx = ();
    let mut w: AArch64Writer<u32> = AArch64Writer::new();
    let mut tables = Vec::new();
    let mut next_label = 0u32;

    // Prologue: save X29/X30 (frame pointer, link register) and stash
    // `backend`/`buf` (X0/X1) in the callee-saved X19/X20.
    w.stp(
        &mut ctx,
        arch,
        &reg(19, MemorySize::_64),
        &reg(20, MemorySize::_64),
        &pair_stack(-16, AddressingMode::PreIndex),
    )
    .unwrap();
    w.stp(
        &mut ctx,
        arch,
        &reg(TABLE_BASE_SAVE, MemorySize::_64),
        &reg(ITERATION_SAVE, MemorySize::_64),
        &pair_stack(-16, AddressingMode::PreIndex),
    )
    .unwrap();
    w.stp(
        &mut ctx,
        arch,
        &reg(29, MemorySize::_64),
        &reg(30, MemorySize::_64),
        &pair_stack(-16, AddressingMode::PreIndex),
    )
    .unwrap();
    w.mov(
        &mut ctx,
        arch,
        &reg(BACKEND_SAVE, MemorySize::_64),
        &reg(BACKEND_ARG, MemorySize::_64),
    )
    .unwrap();
    w.mov(
        &mut ctx,
        arch,
        &reg(BUF_SAVE, MemorySize::_64),
        &reg(BUF_ARG, MemorySize::_64),
    )
    .unwrap();

    for step in &program.steps {
        match step {
            PreparedStep::Flat(ops) => {
                for &op in ops {
                    if program.inputs.contains(&op.out) {
                        continue;
                    }
                    emit_scheduled_call(&mut w, pinned, op);
                }
            }
            PreparedStep::Loop(loop_step) => {
                let table_label = next_label;
                let body_label = next_label + 1;
                let exit_label = next_label + 2;
                next_label += 3;
                let adr_offset = w.offset();
                w.adr_label(
                    &mut ctx,
                    arch,
                    &reg(TABLE_BASE_SAVE, MemorySize::_64),
                    table_label,
                )
                .unwrap();
                w.mov_imm(&mut ctx, arch, &reg(ITERATION_SAVE, MemorySize::_64), 0)
                    .unwrap();
                w.set_label(&mut ctx, arch, body_label).unwrap();
                w.mov_imm(
                    &mut ctx,
                    arch,
                    &reg(TABLE_OFFSET_SCRATCH, MemorySize::_64),
                    loop_step.iterations as u64,
                )
                .unwrap();
                w.cmp(
                    &mut ctx,
                    arch,
                    &reg(ITERATION_SAVE, MemorySize::_64),
                    &reg(TABLE_OFFSET_SCRATCH, MemorySize::_64),
                )
                .unwrap();
                w.bcond_label(&mut ctx, arch, ConditionCode::HS, exit_label)
                    .unwrap();
                for &op in &loop_step.ops {
                    emit_loop_call(&mut w, pinned, loop_step, op);
                }
                w.add(
                    &mut ctx,
                    arch,
                    &reg(ITERATION_SAVE, MemorySize::_64),
                    &reg(ITERATION_SAVE, MemorySize::_64),
                    &literal(1),
                )
                .unwrap();
                w.b_label(&mut ctx, arch, body_label).unwrap();
                w.set_label(&mut ctx, arch, exit_label).unwrap();
                tables.push((adr_offset, loop_step.table.as_slice()));
            }
        }
    }

    // Epilogue: restore X29/X30, X19/X20, and return.
    w.ldp(
        &mut ctx,
        arch,
        &reg(29, MemorySize::_64),
        &reg(30, MemorySize::_64),
        &pair_stack(16, AddressingMode::PostIndex),
    )
    .unwrap();
    w.ldp(
        &mut ctx,
        arch,
        &reg(TABLE_BASE_SAVE, MemorySize::_64),
        &reg(ITERATION_SAVE, MemorySize::_64),
        &pair_stack(16, AddressingMode::PostIndex),
    )
    .unwrap();
    w.ldp(
        &mut ctx,
        arch,
        &reg(19, MemorySize::_64),
        &reg(20, MemorySize::_64),
        &pair_stack(16, AddressingMode::PostIndex),
    )
    .unwrap();
    w.ret(&mut ctx, arch).unwrap();

    let mut bytes = w.into_bytes();
    for (adr_offset, table) in tables {
        let table_offset = bytes.len();
        patch_adr(&mut bytes, adr_offset, table_offset);
        for &slot in table {
            bytes.extend_from_slice(&slot.to_le_bytes());
        }
    }
    bytes
}

fn patch_adr(bytes: &mut [u8], instruction_offset: usize, target_offset: usize) {
    let delta = target_offset as isize - instruction_offset as isize;
    assert!(
        (-(1 << 20)..(1 << 20)).contains(&delta),
        "AArch64 table is farther than ADR's +/-1 MiB range"
    );
    let encoded = delta as u32;
    let immlo = encoded & 0x3;
    let immhi = (encoded >> 2) & 0x7_FFFF;
    let mut word = u32::from_le_bytes(
        bytes[instruction_offset..instruction_offset + 4]
            .try_into()
            .expect("AArch64 ADR instruction is four bytes"),
    );
    word &= !((0x3 << 29) | (0x7_FFFF << 5));
    word |= (immlo << 29) | (immhi << 5);
    bytes[instruction_offset..instruction_offset + 4].copy_from_slice(&word.to_le_bytes());
}

fn emit_scheduled_call<L>(
    w: &mut AArch64Writer<L>,
    pinned: &PinnedAddresses,
    scheduled: ScheduledOp,
) {
    match scheduled.op {
        cirrus_recompile_core::Op::Create(value) => {
            emit_call(w, pinned.create, &[value as u64, scheduled.out.0 as u64])
        }
        cirrus_recompile_core::Op::BitAnd(a, b) => emit_call(
            w,
            pinned.bitand,
            &[a.0 as u64, b.0 as u64, scheduled.out.0 as u64],
        ),
        cirrus_recompile_core::Op::BitOr(a, b) => emit_call(
            w,
            pinned.bitor,
            &[a.0 as u64, b.0 as u64, scheduled.out.0 as u64],
        ),
        cirrus_recompile_core::Op::BitXor(a, b) => emit_call(
            w,
            pinned.bitxor,
            &[a.0 as u64, b.0 as u64, scheduled.out.0 as u64],
        ),
        cirrus_recompile_core::Op::Mux { cond, then, r#else } => emit_call(
            w,
            pinned.mux,
            &[
                cond.0 as u64,
                then.0 as u64,
                r#else.0 as u64,
                scheduled.out.0 as u64,
            ],
        ),
    }
}

#[derive(Clone, Copy)]
enum LoopArgument {
    Constant(u64),
    Slot(LoopSlot),
}

fn emit_loop_call(
    w: &mut AArch64Writer<u32>,
    pinned: &PinnedAddresses,
    loop_step: &TableLoop,
    op: LoopOp,
) {
    let (address, arguments) = match op {
        LoopOp::Create { val, out } => (
            pinned.create,
            [
                LoopArgument::Constant(val as u64),
                LoopArgument::Slot(out),
                LoopArgument::Constant(0),
                LoopArgument::Constant(0),
            ],
        ),
        LoopOp::BitAnd { a, b, out } => (
            pinned.bitand,
            [
                LoopArgument::Slot(a),
                LoopArgument::Slot(b),
                LoopArgument::Slot(out),
                LoopArgument::Constant(0),
            ],
        ),
        LoopOp::BitOr { a, b, out } => (
            pinned.bitor,
            [
                LoopArgument::Slot(a),
                LoopArgument::Slot(b),
                LoopArgument::Slot(out),
                LoopArgument::Constant(0),
            ],
        ),
        LoopOp::BitXor { a, b, out } => (
            pinned.bitxor,
            [
                LoopArgument::Slot(a),
                LoopArgument::Slot(b),
                LoopArgument::Slot(out),
                LoopArgument::Constant(0),
            ],
        ),
        LoopOp::Mux {
            cond,
            then,
            r#else,
            out,
        } => (
            pinned.mux,
            [
                LoopArgument::Slot(cond),
                LoopArgument::Slot(then),
                LoopArgument::Slot(r#else),
                LoopArgument::Slot(out),
            ],
        ),
    };
    let count = match op {
        LoopOp::Create { .. } => 2,
        LoopOp::BitAnd { .. } | LoopOp::BitOr { .. } | LoopOp::BitXor { .. } => 3,
        LoopOp::Mux { .. } => 4,
    };
    emit_pinned_arguments(w, address, loop_step, &arguments[..count]);
}

fn emit_pinned_arguments(
    w: &mut AArch64Writer<u32>,
    address: usize,
    loop_step: &TableLoop,
    arguments: &[LoopArgument],
) {
    let arch = AArch64Arch::default();
    let mut ctx = ();
    w.mov(
        &mut ctx,
        arch,
        &reg(BACKEND_ARG, MemorySize::_64),
        &reg(BACKEND_SAVE, MemorySize::_64),
    )
    .unwrap();
    w.mov(
        &mut ctx,
        arch,
        &reg(BUF_ARG, MemorySize::_64),
        &reg(BUF_SAVE, MemorySize::_64),
    )
    .unwrap();
    for (&argreg, argument) in ARG_REGS.iter().zip(arguments) {
        match argument {
            LoopArgument::Constant(value) => w
                .mov_imm(&mut ctx, arch, &reg(argreg, MemorySize::_64), *value)
                .unwrap(),
            LoopArgument::Slot(slot) => emit_loop_slot(w, loop_step, *slot, argreg),
        }
    }
    w.mov_imm(
        &mut ctx,
        arch,
        &reg(ADDR_SCRATCH, MemorySize::_64),
        address as u64,
    )
    .unwrap();
    w.bl(&mut ctx, arch, &reg(ADDR_SCRATCH, MemorySize::_64))
        .unwrap();
}

fn emit_loop_slot(w: &mut AArch64Writer<u32>, loop_step: &TableLoop, slot: LoopSlot, dest: u8) {
    let arch = AArch64Arch::default();
    let mut ctx = ();
    match slot {
        LoopSlot::Static(slot) => w
            .mov_imm(&mut ctx, arch, &reg(dest, MemorySize::_64), slot.0 as u64)
            .unwrap(),
        LoopSlot::Table(field) => {
            w.mov_imm(
                &mut ctx,
                arch,
                &reg(ADDR_SCRATCH, MemorySize::_64),
                loop_step.fields_per_iteration as u64,
            )
            .unwrap();
            w.mul(
                &mut ctx,
                arch,
                &reg(ADDR_SCRATCH, MemorySize::_64),
                &reg(ITERATION_SAVE, MemorySize::_64),
                &reg(ADDR_SCRATCH, MemorySize::_64),
            )
            .unwrap();
            // The table is a u32 array: convert its row number to a byte
            // offset without relying on an indexed-addressing form.
            w.add(
                &mut ctx,
                arch,
                &reg(ADDR_SCRATCH, MemorySize::_64),
                &reg(ADDR_SCRATCH, MemorySize::_64),
                &reg(ADDR_SCRATCH, MemorySize::_64),
            )
            .unwrap();
            w.add(
                &mut ctx,
                arch,
                &reg(ADDR_SCRATCH, MemorySize::_64),
                &reg(ADDR_SCRATCH, MemorySize::_64),
                &reg(ADDR_SCRATCH, MemorySize::_64),
            )
            .unwrap();
            w.add(
                &mut ctx,
                arch,
                &reg(ADDR_SCRATCH, MemorySize::_64),
                &reg(TABLE_BASE_SAVE, MemorySize::_64),
                &reg(ADDR_SCRATCH, MemorySize::_64),
            )
            .unwrap();
            if field != 0 {
                w.mov_imm(
                    &mut ctx,
                    arch,
                    &reg(TABLE_OFFSET_SCRATCH, MemorySize::_64),
                    u64::from(field) * 4,
                )
                .unwrap();
                w.add(
                    &mut ctx,
                    arch,
                    &reg(ADDR_SCRATCH, MemorySize::_64),
                    &reg(ADDR_SCRATCH, MemorySize::_64),
                    &reg(TABLE_OFFSET_SCRATCH, MemorySize::_64),
                )
                .unwrap();
            }
            w.ldr(
                &mut ctx,
                arch,
                &reg(dest, MemorySize::_32),
                &mem(ADDR_SCRATCH, 0, MemorySize::_32),
            )
            .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cirrus_recompile_core::{Idx, Op};

    #[test]
    fn compiled_bytes_are_word_aligned_and_nonempty() {
        let program = Program {
            ops: alloc::vec![
                Op::Create(true),
                Op::Create(false),
                Op::BitAnd(Idx(0), Idx(1))
            ],
            inputs: Vec::new(),
            outputs: alloc::vec![Idx(2)],
        };
        let pinned = PinnedAddresses {
            create: 0x1000,
            bitand: 0x2000,
            bitor: 0x3000,
            bitxor: 0x4000,
            mux: 0x5000,
        };
        let bytes = compile_aarch64(&program, &pinned);
        assert!(!bytes.is_empty());
        assert_eq!(bytes.len() % 4, 0, "every AArch64 instruction is 4 bytes");
    }

    #[test]
    fn prepared_table_loop_is_compact_and_keeps_its_rows_in_the_artifact() {
        let mut ops = alloc::vec![Op::Create(true)];
        for _ in 0..32 {
            ops.push(Op::BitAnd(Idx(0), Idx(0)));
        }
        let program = Program {
            ops,
            inputs: Vec::new(),
            outputs: alloc::vec![Idx(32)],
        };
        let pinned = PinnedAddresses {
            create: 0x1000,
            bitand: 0x2000,
            bitor: 0x3000,
            bitxor: 0x4000,
            mux: 0x5000,
        };
        let prepared = program.prepare(&OptimizationOptions::default());
        let table = prepared
            .steps
            .iter()
            .find_map(|step| match step {
                PreparedStep::Loop(loop_step) => Some(loop_step.table.clone()),
                PreparedStep::Flat(_) => None,
            })
            .expect("repeated calls should form a table loop");
        let prepared_bytes = compile_prepared_aarch64(&prepared, &pinned);
        let raw_bytes = compile_aarch64_with_options(
            &program,
            &pinned,
            &OptimizationOptions {
                enabled: false,
                ..OptimizationOptions::default()
            },
        );
        let table_bytes: Vec<u8> = table.iter().flat_map(|slot| slot.to_le_bytes()).collect();

        assert!(prepared_bytes.ends_with(&table_bytes));
        assert!(prepared_bytes.len() < raw_bytes.len());
    }

    /// A worked demonstration of driving `portal-solutions-asm-regalloc`
    /// over a stream of buffer-slot [`Idx`]es, as described in this module's
    /// documentation: pushing a slot's value asks the allocator which
    /// physical register it now lives in (spilling an older resident via the
    /// returned `Cmd`s if none is free), and popping releases it. This is
    /// the building block an optimizing variant of `compile_aarch64` would
    /// use to cache a value across adjacent pinned-function calls instead of
    /// re-deriving it from the buffer every time -- "regalloc over indices,
    /// not values": every `Target` here identifies a slot index, never the
    /// Boolean value itself.
    #[test]
    fn regalloc_over_indices() {
        use core::ops::{Index, IndexMut};
        use portal_solutions_asm_regalloc::{Length, RegAlloc, RegAllocFrame};

        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        struct Gpr;
        impl TryFrom<usize> for Gpr {
            type Error = core::convert::Infallible;
            fn try_from(_: usize) -> Result<Self, Self::Error> {
                Ok(Gpr)
            }
        }

        // `RegAlloc` is generic over how a caller stores its per-kind frame
        // arrays (to support multiple register classes); this backend only
        // ever needs one (general-purpose, integer) kind, so index by it
        // trivially.
        struct SingleKind<const N: usize>([RegAllocFrame<Gpr>; N]);
        impl<const N: usize> Index<Gpr> for SingleKind<N> {
            type Output = [RegAllocFrame<Gpr>; N];
            fn index(&self, _: Gpr) -> &Self::Output {
                &self.0
            }
        }
        impl<const N: usize> IndexMut<Gpr> for SingleKind<N> {
            fn index_mut(&mut self, _: Gpr) -> &mut Self::Output {
                &mut self.0
            }
        }
        impl<const N: usize> Length for SingleKind<N> {
            fn len(&self) -> usize {
                1
            }
        }

        const N: usize = 4;
        let mut alloc: RegAlloc<Gpr, N, SingleKind<N>> = RegAlloc {
            frames: SingleKind(core::array::from_fn(|_| RegAllocFrame::Empty)),
            tos: None,
        };

        // Push slot #0 (e.g. the result of `Op::BitAnd(a, b)` at index 0):
        // the allocator hands back a physical register and no spill code,
        // since every frame starts empty.
        let (physical_reg, commands) = alloc.push(Gpr).unwrap();
        assert_eq!(
            commands.count(),
            0,
            "an empty bank never needs to spill first"
        );
        assert!((physical_reg as usize) < N);

        // Popping it back (e.g. right before the call site that consumes
        // slot #0 as an operand) releases the register with no spill either,
        // since nothing else claimed it in between.
        let (target, commands) = alloc.pop(Gpr);
        assert_eq!(target.reg, physical_reg);
        assert_eq!(commands.count(), 0);
    }
}
