//! Allocation-using canned replacements for Arm call hooks.
//!
//! This module is behind `call-hooks`, so the synchronous [`ArmHandler`]
//! extension remains available to allocator-free bare-metal users. Replacements
//! intentionally affect registers only: a storage-mutating replacement needs a
//! separately designed, policy-safe storage capability.

use alloc::collections::BTreeMap;

use crate::{ArmCallAction, ArmCallEvent};

/// A canned replacement for a direct Arm call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArmCallReplacement {
    /// Continue after the call without changing registers.
    NoOp,
    /// Write `r0` through `r3` then continue after the call.
    ReturnConstants {
        /// The `r0` result.
        r0: u32,
        /// The `r1` result.
        r1: u32,
        /// The `r2` result.
        r2: u32,
        /// The `r3` result.
        r3: u32,
    },
}

/// A map from normalized direct-call target PCs to canned replacements.
#[derive(Clone, Debug, Default)]
pub struct ArmCallRegistry {
    replacements: BTreeMap<u32, ArmCallReplacement>,
}

impl ArmCallRegistry {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `replacement` at the normalized direct `BL` target, returning
    /// the old entry when present.
    pub fn insert(
        &mut self,
        target: u32,
        replacement: ArmCallReplacement,
    ) -> Option<ArmCallReplacement> {
        self.replacements.insert(target, replacement)
    }

    /// Look up a replacement for a direct `BL`. Register calls and returns
    /// deliberately have no registry target and therefore never match.
    pub fn lookup(&self, event: &ArmCallEvent) -> Option<ArmCallReplacement> {
        let ArmCallEvent::DirectCall { target, .. } = event else {
            return None;
        };
        self.replacements.get(target).copied()
    }
}

/// Apply a replacement to an Arm register view and return its hook action.
///
/// Values are written as caller-owned symbolic constants, hence a recording
/// context observes no hidden work. Callers needing outputs derived from
/// symbolic inputs must create them through their context before calling this.
pub fn apply_replacement<W: Clone>(
    replacement: ArmCallReplacement,
    regs: &mut [[W; 32]],
    constants: &mut [Option<u32>],
    offsets: &mut [Option<i32>],
    zero: &W,
    one: &W,
) -> ArmCallAction {
    let ArmCallReplacement::ReturnConstants { r0, r1, r2, r3 } = replacement else {
        return ArmCallAction::ReturnNow;
    };
    for (register, value) in [r0, r1, r2, r3].into_iter().enumerate() {
        constants[register] = Some(value);
        offsets[register] = None;
        for bit in 0..32 {
            regs[register][bit] = if value & (1 << bit) == 0 {
                zero.clone()
            } else {
                one.clone()
            };
        }
    }
    ArmCallAction::ReturnNow
}
