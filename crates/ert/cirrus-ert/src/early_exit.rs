//! Static recognizer for the opt-in "deoptimize secret-dependent early-exit
//! loops" idiom (see [`cirrus_ert_core::EarlyExitLoopOptions`]).
//!
//! RV32 here is fixed-width (no compressed instructions), so a bounded
//! forward disassembly from a candidate branch is a safe, cheap way to
//! classify the surrounding loop shape — there is no need for a general
//! CFG/dominance analysis. Two forward walks do all the work:
//!
//! - [`resolve_continue`] proves that one successor of the candidate branch
//!   leads, through only unconditional jumps and ordinary (non-branching)
//!   instructions, back to a loop's own backward branch that encloses the
//!   candidate. Those skipped-over instructions are never elided by the
//!   deoptimization — they still execute for real, exactly as before — so
//!   nothing about them needs restricting beyond "control flow resolves
//!   statically".
//! - [`resolve_landing`] proves that the *other* successor (the early exit)
//!   reaches, through only unconditional jumps and whitelisted
//!   constant-register writes, the same landing address that the loop's own
//!   natural (concrete) exit reaches. Anything else — a call, an indirect
//!   jump, another branch — stops the walk there instead of being chased
//!   through, because unlike the continue side, the early-exit side's
//!   instructions are the ones the deoptimization would otherwise skip by
//!   always forcing the branch to continue the loop. Only unconditional
//!   jumps and constant writes are safe to elide that way; everything past
//!   the landing point is shared code that still executes for real, once,
//!   when the loop's own concrete exit is genuinely reached.
//!
//! A recognized site is only ever a relaxation of today's hard error: any
//! shape that doesn't fit cleanly falls straight through to it, unchanged.

use rv_asm::{Imm, Inst, Reg, Xlen};

use cirrus_ert_core::ComparePredicate;

use crate::RawMemory;

/// Bound on how many `(register, constant)` writes the early-exit path may
/// perform before reaching the shared landing point.
pub(crate) const MAX_EXIT_WRITES: usize = 4;

/// A branch recognized as a secret-dependent early exit from a
/// concrete-bounded loop, along with everything needed to replay it.
#[derive(Clone, Copy)]
pub(crate) struct RecognizedSite {
    pub(crate) branch_pc: u64,
    pub(crate) predicate: ComparePredicate,
    /// `true` when the branch condition being *true* means "take the early
    /// exit"; `false` when it means "keep looping" (the exit is the
    /// not-taken/fallthrough successor instead).
    pub(crate) exit_when_taken: bool,
    /// Where control always goes instead of exiting.
    pub(crate) continue_target: u64,
    pub(crate) exit_writes: [Option<(Reg, u64)>; MAX_EXIT_WRITES],
    pub(crate) exit_write_count: usize,
}

fn decode_at(mem: &RawMemory<'_>, address: u64, xlen: Xlen) -> Option<(Inst, u64)> {
    let half = u16::from_le_bytes(mem.read64::<2>(address)?);
    if Inst::first_byte_is_compressed(half as u8) {
        return Inst::decode_compressed(half, xlen)
            .ok()
            .map(|instruction| (instruction, 2));
    }
    let bytes = mem.read64::<4>(address)?;
    Inst::decode(u32::from_le_bytes(bytes), xlen)
        .ok()
        .map(|(instruction, _)| (instruction, 4))
}

/// Walk forward from `start`, proving it reaches a backward branch (the
/// enclosing loop's latch) that encloses `branch_pc`, without passing
/// through any other conditional branch or indirect jump. Returns that
/// latch's not-taken (natural exit) successor address.
fn resolve_continue(
    mem: &RawMemory<'_>,
    start: u64,
    branch_pc: u64,
    budget: &mut u16,
    xlen: Xlen,
) -> Option<u64> {
    let mut pc = start;
    loop {
        *budget = budget.checked_sub(1)?;
        match decode_at(mem, pc, xlen)?.0 {
            Inst::Jal { offset, dest } if dest == Reg::ZERO => {
                pc = pc.wrapping_add_signed(i64::from(offset.as_i32()));
            }
            Inst::Jalr { .. } => return None,
            Inst::Beq { offset, .. }
            | Inst::Bne { offset, .. }
            | Inst::Bge { offset, .. }
            | Inst::Blt { offset, .. }
            | Inst::Bgeu { offset, .. }
            | Inst::Bltu { offset, .. } => {
                let target = pc.wrapping_add_signed(i64::from(offset.as_i32()));
                if target >= pc {
                    // Not a backward edge — a second real branch on this
                    // path, which the idiom's "exactly one candidate" rule
                    // does not allow.
                    return None;
                }
                if target > branch_pc {
                    // A backward branch was found, but it doesn't enclose
                    // the candidate — this isn't the loop we're looking for.
                    return None;
                }
                return Some(pc.wrapping_add(decode_at(mem, pc, xlen)?.1));
            }
            _ => pc = pc.wrapping_add(decode_at(mem, pc, xlen)?.1),
        }
    }
}

/// Walk forward from `natural_exit` through only unconditional jumps,
/// stopping at the first instruction that isn't one. This is the loop's
/// natural-exit merge point `M` — computed *without* chasing constant
/// writes, since the natural-exit path never needs eliding (it always
/// executes for real) and must not be confused with common/shared epilogue
/// code that happens to also start with an `addi rd, x0, imm`.
fn resolve_natural_landing(
    mem: &RawMemory<'_>,
    natural_exit: u64,
    budget: &mut u16,
    xlen: Xlen,
) -> Option<u64> {
    let mut pc = natural_exit;
    loop {
        *budget = budget.checked_sub(1)?;
        match decode_at(mem, pc, xlen)?.0 {
            Inst::Jal { offset, dest } if dest == Reg::ZERO => {
                pc = pc.wrapping_add_signed(i64::from(offset.as_i32()));
            }
            _ => return Some(pc),
        }
    }
}

/// The result of walking the early-exit side forward to the merge point.
struct ExitLanding {
    writes: [Option<(Reg, u64)>; MAX_EXIT_WRITES],
    write_count: usize,
}

/// Walk forward from `start` through only unconditional jumps and
/// `addi rd, x0, imm` constant writes, requiring that walk to land *exactly*
/// on `merge` (the natural-exit path's own merge point) before consuming any
/// instruction at or past it. Reaching anything else — a non-whitelisted
/// instruction before `merge`, or overshooting past it — is a rejection:
/// this early-exit path doesn't provably reconverge with the natural exit.
fn resolve_exit_landing(
    mem: &RawMemory<'_>,
    start: u64,
    merge: u64,
    budget: &mut u16,
    xlen: Xlen,
) -> Option<ExitLanding> {
    let mut pc = start;
    let mut writes = [None; MAX_EXIT_WRITES];
    let mut write_count = 0usize;
    loop {
        if pc == merge {
            return Some(ExitLanding {
                writes,
                write_count,
            });
        }
        *budget = budget.checked_sub(1)?;
        match decode_at(mem, pc, xlen)?.0 {
            Inst::Jal { offset, dest } if dest == Reg::ZERO => {
                pc = pc.wrapping_add_signed(i64::from(offset.as_i32()));
            }
            Inst::Addi { imm, dest, src1 } if src1 == Reg::ZERO => {
                if write_count >= MAX_EXIT_WRITES {
                    return None;
                }
                writes[write_count] = Some((dest, imm.as_i32() as i64 as u64));
                write_count += 1;
                pc = pc.wrapping_add(decode_at(mem, pc, xlen)?.1);
            }
            _ => return None,
        }
    }
}

/// Attempt to recognize `branch_pc` (a conditional branch RV32 `execute()`
/// could not resolve concretely) as a secret-dependent early exit from a
/// concrete-bounded loop.
pub(crate) fn recognize(
    mem: &RawMemory<'_>,
    branch_pc: u64,
    offset: Imm,
    predicate: ComparePredicate,
    max_lookahead: u16,
    xlen: Xlen,
) -> Option<RecognizedSite> {
    let taken_target = branch_pc.wrapping_add_signed(i64::from(offset.as_i32()));
    let not_taken_target = branch_pc.wrapping_add(decode_at(mem, branch_pc, xlen)?.1);

    for exit_when_taken in [false, true] {
        let (continue_start, exit_start) = if exit_when_taken {
            (not_taken_target, taken_target)
        } else {
            (taken_target, not_taken_target)
        };

        let mut continue_budget = max_lookahead;
        let Some(natural_exit) =
            resolve_continue(mem, continue_start, branch_pc, &mut continue_budget, xlen)
        else {
            continue;
        };

        let mut natural_budget = max_lookahead;
        let Some(merge) = resolve_natural_landing(mem, natural_exit, &mut natural_budget, xlen) else {
            continue;
        };

        let mut exit_budget = max_lookahead;
        let Some(exit_landing) = resolve_exit_landing(mem, exit_start, merge, &mut exit_budget, xlen)
        else {
            continue;
        };

        return Some(RecognizedSite {
            branch_pc,
            predicate,
            exit_when_taken,
            continue_target: continue_start,
            exit_writes: exit_landing.writes,
            exit_write_count: exit_landing.write_count,
        });
    }

    None
}
