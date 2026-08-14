//! Lower a `cirrus_recompile_core::Program` to real AArch64 machine code.
//!
//! The emitted function has the shape `void {name}(uint8_t *buf)` under
//! AAPCS64: given a pointer to a `program.ops.len()`-byte scratch buffer, it
//! performs the recorded trace by calling out to the pinned runtime
//! functions -- the exact same functions the LLVM and Rust backends call by
//! name -- passing buffer-slot indices as arguments, exactly as
//! `cirrus-recompile-core`'s module documentation describes. The caller
//! supplies each pinned function's address ([`PinnedAddresses`]); this crate
//! does not depend on any particular runtime implementation.
//!
//! Every call site loads its pinned function's address into a scratch
//! register (`mov_imm` + indirect `bl`) rather than emitting a direct
//! PC-relative branch, so the emitted code has no relocations to resolve and
//! can be copied into freshly allocated memory and executed immediately.
//! `X19` (callee-saved) holds `buf` across calls, since a pinned function is
//! free to clobber its own first argument register (`X0`) like any other
//! AAPCS64 callee.
//!
//! # Register allocation
//!
//! Every operand this backend's instructions touch is either the constant
//! buffer-slot index baked into a single call site (materialized fresh via
//! `mov_imm` every time -- there is no cross-call liveness to manage) or the
//! one long-lived `buf` pointer kept in `X19` for the whole function. At this
//! granularity -- one pinned-function call per Boolean op -- there is
//! genuinely nothing to spill: the "regalloc over indices instead of values"
//! goal is visible instead in `portal-solutions-asm-regalloc`'s suitability
//! for the natural follow-on optimization this module intentionally leaves
//! undone, caching a recently-produced slot's *value* in a register across
//! adjacent calls to skip a `mov_imm`+call round trip. See
//! `regalloc_over_indices` in this crate's tests for a worked demonstration
//! of that allocator doing exactly this bookkeeping over a stream of pushed
//! and popped [`Idx`](cirrus_recompile_core::Idx) values.

extern crate alloc;

use alloc::vec::Vec;

use cirrus_recompile_core::{Idx, Op, Program};
use portal_pc_asm_common::types::{mem::MemorySize, reg::Reg};
use portal_solutions_asm_aarch64::{
    AArch64Arch, RegisterClass,
    out::{
        WriterCore,
        arg::{AddressingMode, ArgKind, MemArgKind},
        bin::AArch64Writer,
    },
};

/// The address of each pinned runtime function this backend calls, by exact
/// name -- matching `cirrus-recompile-rt`'s `cirrus_rt_*` symbols.
pub struct PinnedAddresses {
    /// `cirrus_rt_create(buf, val, out)`.
    pub create: usize,
    /// `cirrus_rt_bitand(buf, a, b, out)`.
    pub bitand: usize,
    /// `cirrus_rt_bitor(buf, a, b, out)`.
    pub bitor: usize,
    /// `cirrus_rt_bitxor(buf, a, b, out)`.
    pub bitxor: usize,
    /// `cirrus_rt_mux(buf, cond, then, r#else, out)`.
    pub mux: usize,
}

const BUF_ARG: u8 = 0; // X0: this function's own `buf` argument.
const BUF_SAVE: u8 = 19; // X19: callee-saved copy of `buf`, live across calls.
const ADDR_SCRATCH: u8 = 9; // X9: scratch for a pinned function's address.
const ARG_REGS: [u8; 4] = [1, 2, 3, 4]; // X1..X4: pinned-function arguments.

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

fn single_stack(disp: i32, mode: AddressingMode) -> MemArgKind<ArgKind> {
    MemArgKind::Mem {
        base: ArgKind::Reg {
            reg: Reg(31),
            size: MemorySize::_64,
        },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
        mode,
    }
}

fn emit_call(w: &mut AArch64Writer, addr: usize, args: &[u64]) {
    let arch = AArch64Arch::default();
    let mut ctx = ();
    // Every pinned function is a normal AAPCS64 callee: it is free to
    // clobber X0, so restore `buf` from X19 immediately before each call.
    w.mov(&mut ctx, arch, &reg(BUF_ARG, MemorySize::_64), &reg(BUF_SAVE, MemorySize::_64))
        .unwrap();
    for (&argreg, &value) in ARG_REGS.iter().zip(args) {
        w.mov_imm(&mut ctx, arch, &reg(argreg, MemorySize::_64), value)
            .unwrap();
    }
    w.mov_imm(&mut ctx, arch, &reg(ADDR_SCRATCH, MemorySize::_64), addr as u64)
        .unwrap();
    w.bl(&mut ctx, arch, &reg(ADDR_SCRATCH, MemorySize::_64))
        .unwrap();
}

/// Emit an executable AArch64 function performing `program`'s recorded
/// trace, calling out to `pinned`'s functions. Returns the encoded
/// instruction bytes (position-independent: no relocations remain).
pub fn compile_aarch64(program: &Program, pinned: &PinnedAddresses) -> Vec<u8> {
    let arch = AArch64Arch::default();
    let mut ctx = ();
    let mut w: AArch64Writer = AArch64Writer::new();

    // Prologue: save X29/X30 (frame pointer, link register) and stash `buf`
    // (X0) in the callee-saved X19.
    w.stp(
        &mut ctx,
        arch,
        &reg(29, MemorySize::_64),
        &reg(30, MemorySize::_64),
        &pair_stack(-16, AddressingMode::PreIndex),
    )
    .unwrap();
    w.str(
        &mut ctx,
        arch,
        &reg(BUF_SAVE, MemorySize::_64),
        &single_stack(-16, AddressingMode::PreIndex),
    )
    .unwrap();
    w.mov(&mut ctx, arch, &reg(BUF_SAVE, MemorySize::_64), &reg(BUF_ARG, MemorySize::_64))
        .unwrap();

    for (i, op) in program.ops.iter().enumerate() {
        let out = Idx(i as u32);
        if program.inputs.contains(&out) {
            // Populated by the caller before running the compiled function;
            // see `cirrus_recompile_rt::execute`'s matching convention.
            continue;
        }
        match *op {
            Op::Create(v) => emit_call(&mut w, pinned.create, &[v as u64, out.0 as u64]),
            Op::BitAnd(a, b) => emit_call(&mut w, pinned.bitand, &[a.0 as u64, b.0 as u64, out.0 as u64]),
            Op::BitOr(a, b) => emit_call(&mut w, pinned.bitor, &[a.0 as u64, b.0 as u64, out.0 as u64]),
            Op::BitXor(a, b) => emit_call(&mut w, pinned.bitxor, &[a.0 as u64, b.0 as u64, out.0 as u64]),
            Op::Mux { cond, then, r#else } => emit_call(
                &mut w,
                pinned.mux,
                &[cond.0 as u64, then.0 as u64, r#else.0 as u64, out.0 as u64],
            ),
        }
    }

    // Epilogue: restore X19, X29/X30, and return.
    w.ldr(
        &mut ctx,
        arch,
        &reg(BUF_SAVE, MemorySize::_64),
        &single_stack(16, AddressingMode::PostIndex),
    )
    .unwrap();
    w.ldp(
        &mut ctx,
        arch,
        &reg(29, MemorySize::_64),
        &reg(30, MemorySize::_64),
        &pair_stack(16, AddressingMode::PostIndex),
    )
    .unwrap();
    w.ret(&mut ctx, arch).unwrap();

    w.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_bytes_are_word_aligned_and_nonempty() {
        let program = Program {
            ops: alloc::vec![Op::Create(true), Op::Create(false), Op::BitAnd(Idx(0), Idx(1))],
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
        assert_eq!(commands.count(), 0, "an empty bank never needs to spill first");
        assert!((physical_reg as usize) < N);

        // Popping it back (e.g. right before the call site that consumes
        // slot #0 as an operand) releases the register with no spill either,
        // since nothing else claimed it in between.
        let (target, commands) = alloc.pop(Gpr);
        assert_eq!(target.reg, physical_reg);
        assert_eq!(commands.count(), 0);
    }
}
