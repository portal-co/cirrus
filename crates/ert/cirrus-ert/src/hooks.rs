//! An opt-in, allocation-using call-interception registry.
//!
//! This is the `call-hooks` feature's home: bare-metal ERT users never link
//! it, and the discipline mirrors the `prepared-recording` feature —
//! allocator-requiring machinery stays out of the default build.
//!
//! Recorder discipline: replacements here only write registers through the
//! caller's symbolic constants, so a recording context observes no hidden
//! state. A replacement that computes from symbolic inputs should emit gates
//! through the handler's own context instead of using this registry.

use alloc::collections::BTreeMap;

use rv_asm::Reg;

use crate::{CallAction, CallEvent};

/// A canned call replacement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallReplacement {
    /// Replace the call with no effect: registers keep their values and the
    /// caller continues at the instruction following the call.
    NoOp,
    /// Write `a0` and `a1` with the constants and return immediately.
    ReturnConstants {
        /// The `a0` result value.
        a0: u64,
        /// The `a1` result value.
        a1: u64,
    },
}

/// A map from call-target addresses to canned replacements.
///
/// Typical use is inside a handler's [`RvHandler::call_hook`](crate::RvHandler::call_hook):
/// consult [`CallRegistry::lookup`] and apply hits with
/// [`apply_replacement`].
#[derive(Clone, Debug, Default)]
pub struct CallRegistry {
    replacements: BTreeMap<u64, CallReplacement>,
}

impl CallRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a replacement for calls to `target`, returning any previous
    /// entry.
    pub fn insert(&mut self, target: u64, replacement: CallReplacement) -> Option<CallReplacement> {
        self.replacements.insert(target, replacement)
    }

    /// The replacement registered for `event`'s call target, if any. Returns
    /// are never intercepted: they name no call target.
    pub fn lookup(&self, event: &CallEvent) -> Option<CallReplacement> {
        let target = match event {
            CallEvent::Jal { target, .. } | CallEvent::Jalr { target, .. } => *target,
            CallEvent::Return { .. } | CallEvent::UnresolvedJalr { .. } => return None,
        };
        self.replacements.get(&target).copied()
    }
}

/// Apply a registry replacement to the register file, producing the
/// corresponding action. Register writes use the caller's symbolic
/// constants, so a recording context sees no hidden state.
pub fn apply_replacement<W: Clone, const BITS: usize>(
    replacement: CallReplacement,
    regs: &mut [[W; BITS]],
    reg_consts: &mut [Option<u64>],
    offsets: &mut [Option<i64>],
    zero: &W,
    one: &W,
) -> CallAction {
    match replacement {
        CallReplacement::NoOp => CallAction::ReturnNow,
        CallReplacement::ReturnConstants { a0, a1 } => {
            let mask = if BITS == 64 {
                u64::MAX
            } else {
                (1u64 << BITS) - 1
            };
            for (register, value) in [(Reg::A0, a0 & mask), (Reg::A1, a1 & mask)] {
                let index = register.0 as usize;
                reg_consts[index] = Some(value & mask);
                offsets[index] = None;
                for bit in 0..BITS {
                    regs[index][bit] = if (value >> bit) & 1 == 0 {
                        zero.clone()
                    } else {
                        one.clone()
                    };
                }
            }
            CallAction::ReturnNow
        }
    }
}
