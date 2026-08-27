//! Static recognizer for the opt-in "deoptimize secret-dependent early-exit
//! loops" idiom (see [`cirrus_ert_core::EarlyExitLoopOptions`]), ported from
//! `cirrus-ert`'s RV32 recognizer to Thumb-2's variable-width encoding.
//!
//! v1 scope is deliberately narrower than RV32's: only `CBZ`/`CBNZ`
//! (`Op::CompareBranch`) branches are recognized, both as the candidate
//! early exit and as the loop's own natural (concrete) guard. General
//! `Bcc` branches fed by a prior `CMP`/`SUBS` are not supported, since
//! validating which two operands fed the flags they read would need real
//! flag-source tracking this facade doesn't otherwise keep. `CBZ`/`CBNZ`
//! is still a common codegen shape for both "is this loaded value zero"
//! bulk-reduction idioms and, via `SUBS`-then-`CBNZ`, equality compares.
//! An `IT` instruction anywhere in a scanned region is an unconditional
//! rejection: this recognizer does not model Thumb-2 predication.
//!
//! See `cirrus-ert`'s `early_exit` module for the shared walk-forward
//! design rationale — this is the same two-walk approach (`resolve_continue`
//! / `resolve_natural_landing` + `resolve_exit_landing`), adapted to
//! variable instruction length and absolute (pre-resolved) branch targets.

use crate::{Op, Operand, RawMemory, decode16, decode32};

/// Bound on how many `(register, constant)` writes the early-exit path may
/// perform before reaching the shared landing point.
pub(crate) const MAX_EXIT_WRITES: usize = 4;

/// A `CBZ`/`CBNZ` recognized as a secret-dependent early exit from a
/// concrete-bounded loop, along with everything needed to replay it.
#[derive(Clone, Copy)]
pub(crate) struct RecognizedSite {
    pub(crate) branch_pc: u32,
    pub(crate) register: u8,
    /// The recognized branch's own `nonzero` flag (`true` for `CBNZ`).
    pub(crate) nonzero: bool,
    /// `true` when the register being nonzero/zero (per `nonzero`) means
    /// "take the early exit"; `false` when it means "keep looping".
    pub(crate) exit_when_taken: bool,
    /// Where control always goes instead of exiting.
    pub(crate) continue_target: u32,
    pub(crate) exit_writes: [Option<(u8, u32)>; MAX_EXIT_WRITES],
    pub(crate) exit_write_count: usize,
}

fn decode_at(mem: &RawMemory<'_>, pc: u32) -> Option<(Op, u32)> {
    let first = u16::from_le_bytes(mem.read::<2>(pc)?);
    let wide = first & 0xe000 == 0xe000 && first & 0x1800 != 0;
    let decoded = if wide {
        let second = u16::from_le_bytes(mem.read::<2>(pc.wrapping_add(2))?);
        decode32(pc, first, second).ok()?
    } else {
        decode16(pc, first).ok()?
    };
    Some((decoded.operation, decoded.len))
}

/// Walk forward from `start`, proving it reaches a backward branch (the
/// enclosing loop's latch) that encloses `branch_pc`, without passing
/// through an `IT` block, another candidate branch, or any indirect control
/// transfer. Returns that latch's not-taken (natural exit) successor
/// address.
///
/// The latch may be a `CBZ`/`CBNZ` *or* a `Bcc` — real Thumb-2 `CBZ`/`CBNZ`
/// are forward-only on actual hardware, so a real toolchain's own
/// concrete-bounded loop back edge is normally a `Bcc` fed by a prior
/// `CMP`/`SUBS`, not a `CompareBranch`. That `Bcc` is not itself something
/// this recognizer resolves -- it is walked over, unmodified, exactly like
/// any other loop-body instruction, and is expected to already resolve
/// through the interpreter's existing concrete-flags branch handling at
/// runtime (the loop's own trip count is concrete by the idiom's contract).
fn resolve_continue(
    mem: &RawMemory<'_>,
    start: u32,
    branch_pc: u32,
    budget: &mut u16,
) -> Option<u32> {
    let mut pc = start;
    loop {
        *budget = budget.checked_sub(1)?;
        let (op, len) = decode_at(mem, pc)?;
        let target = match op {
            Op::It { .. } => return None,
            Op::Branch {
                target,
                condition: None,
            } => {
                pc = target;
                continue;
            }
            Op::Branch {
                target,
                condition: Some(_),
            } => target,
            Op::CompareBranch { target, .. } => target,
            Op::BranchRegister { .. }
            | Op::BranchExchangeNonSecure { .. }
            | Op::CallRegister { .. }
            | Op::SecureGateway => return None,
            _ => {
                pc = pc.wrapping_add(len);
                continue;
            }
        };
        if target >= pc {
            // Not a backward edge -- a second real branch on this path,
            // which the idiom's "exactly one candidate" rule does not
            // allow.
            return None;
        }
        if target > branch_pc {
            return None;
        }
        return Some(pc.wrapping_add(len));
    }
}

/// Walk forward from `natural_exit` through only unconditional `B` jumps,
/// stopping at the first instruction that isn't one. This is the loop's
/// natural-exit merge point, computed without chasing constant writes for
/// the same reason `cirrus-ert`'s equivalent does not: the natural-exit
/// path always executes for real and must not be confused with shared
/// epilogue code that happens to also start with a `MOV rd, #imm`.
fn resolve_natural_landing(
    mem: &RawMemory<'_>,
    natural_exit: u32,
    budget: &mut u16,
) -> Option<u32> {
    let mut pc = natural_exit;
    loop {
        *budget = budget.checked_sub(1)?;
        let (op, _len) = decode_at(mem, pc)?;
        match op {
            Op::Branch {
                target,
                condition: None,
            } => pc = target,
            _ => return Some(pc),
        }
    }
}

struct ExitLanding {
    writes: [Option<(u8, u32)>; MAX_EXIT_WRITES],
    write_count: usize,
}

/// Walk forward from `start` through only unconditional `B` jumps and
/// non-flag-setting `MOV rd, #imm` constant writes, requiring that walk to
/// land *exactly* on `merge` before consuming any instruction at or past
/// it. A flag-setting `MOVS` is deliberately excluded from the whitelist:
/// its NZ flags are an observable side effect this recognizer does not
/// replay, so eliding it would be unsound if shared code past `merge`
/// reads those flags.
fn resolve_exit_landing(
    mem: &RawMemory<'_>,
    start: u32,
    merge: u32,
    budget: &mut u16,
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
        let (op, len) = decode_at(mem, pc)?;
        match op {
            Op::Branch {
                target,
                condition: None,
            } => pc = target,
            Op::Move {
                dest,
                source: Operand::Immediate(value),
                set_flags: false,
            } => {
                if write_count >= MAX_EXIT_WRITES {
                    return None;
                }
                writes[write_count] = Some((dest, value));
                write_count += 1;
                pc = pc.wrapping_add(len);
            }
            _ => return None,
        }
    }
}

/// Attempt to recognize `branch_pc` (a `CBZ`/`CBNZ` `execute()` could not
/// resolve concretely) as a secret-dependent early exit from a
/// concrete-bounded loop.
pub(crate) fn recognize(
    mem: &RawMemory<'_>,
    branch_pc: u32,
    register: u8,
    nonzero: bool,
    len: u32,
    taken_target: u32,
    max_lookahead: u16,
) -> Option<RecognizedSite> {
    let not_taken_target = branch_pc.wrapping_add(len);

    for exit_when_taken in [false, true] {
        let (continue_start, exit_start) = if exit_when_taken {
            (not_taken_target, taken_target)
        } else {
            (taken_target, not_taken_target)
        };

        let mut continue_budget = max_lookahead;
        let Some(natural_exit) =
            resolve_continue(mem, continue_start, branch_pc, &mut continue_budget)
        else {
            continue;
        };

        let mut natural_budget = max_lookahead;
        let Some(merge) = resolve_natural_landing(mem, natural_exit, &mut natural_budget) else {
            continue;
        };

        let mut exit_budget = max_lookahead;
        let Some(exit_landing) = resolve_exit_landing(mem, exit_start, merge, &mut exit_budget)
        else {
            continue;
        };

        return Some(RecognizedSite {
            branch_pc,
            register,
            nonzero,
            exit_when_taken,
            continue_target: continue_start,
            exit_writes: exit_landing.writes,
            exit_write_count: exit_landing.write_count,
        });
    }

    None
}
