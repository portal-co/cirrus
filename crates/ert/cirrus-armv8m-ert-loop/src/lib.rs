#![no_std]
#![warn(missing_docs)]

//! Thumb-2 loop-adapter state for the shared ERT scheduler.
//!
//! This crate deliberately begins with only the fail-closed concrete-state
//! merge contract. The Arm interpreter owns instruction semantics; the shared
//! loop core owns candidate scheduling. [`ThumbAgreement`] is the bridge: at a
//! multi-candidate boundary, every host-only Arm state component must agree.
//! In particular that includes SP, private return-stack state, ITSTATE, and
//! virtual TrustZone state. A later wire-backed register/NZCV fold can only
//! run after this agreement check has succeeded.

use cirrus_armv8m_ert::SecurityState;

/// Concrete Arm state which may not be selected by a symbolic candidate
/// predicate in the first Thumb loop-adapter cut.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThumbAgreement<const FRAMES: usize> {
    /// Concrete architectural stack pointer.
    pub sp: u32,
    /// Depth of the interpreter's private return stack.
    pub rstack_depth: usize,
    /// In-flight private return addresses above the boundary's base depth.
    pub rstack: [u32; FRAMES],
    /// Thumb ITSTATE after the body.
    pub itstate: u8,
    /// Virtual TrustZone-M state after the body.
    pub security_state: SecurityState,
}

/// A candidate state did not satisfy the safe first-cut merge contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThumbMergeError {
    /// Candidate states disagree on concrete-only architectural state.
    ConcreteStateDivergence,
}

/// Merge `candidate` into the prior agreement.
///
/// The first candidate establishes the agreement. Subsequent candidates must
/// compare equal, otherwise the adapter fails closed rather than treating a
/// host-only state component as a symbolic value.
pub fn merge_agreement<const FRAMES: usize>(
    agreement: &mut Option<ThumbAgreement<FRAMES>>,
    candidate: ThumbAgreement<FRAMES>,
) -> Result<(), ThumbMergeError> {
    match agreement {
        Some(previous) if *previous != candidate => Err(ThumbMergeError::ConcreteStateDivergence),
        Some(_) => Ok(()),
        slot @ None => {
            *slot = Some(candidate);
            Ok(())
        }
    }
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::{ThumbAgreement, ThumbMergeError, merge_agreement};
    use cirrus_armv8m_ert::SecurityState;

    fn state() -> ThumbAgreement<2> {
        ThumbAgreement {
            sp: 64,
            rstack_depth: 1,
            rstack: [4, 0],
            itstate: 0,
            security_state: SecurityState::Secure,
        }
    }

    #[test]
    fn equal_concrete_state_merges() {
        let mut agreement = None;
        assert_eq!(merge_agreement(&mut agreement, state()), Ok(()));
        assert_eq!(merge_agreement(&mut agreement, state()), Ok(()));
    }

    #[test]
    fn trustzone_or_itstate_divergence_fails_closed() {
        let mut agreement = Some(state());
        let mut non_secure = state();
        non_secure.security_state = SecurityState::NonSecure;
        assert_eq!(
            merge_agreement(&mut agreement, non_secure),
            Err(ThumbMergeError::ConcreteStateDivergence)
        );
        let mut it = state();
        it.itstate = 0x18;
        assert_eq!(
            merge_agreement(&mut agreement, it),
            Err(ThumbMergeError::ConcreteStateDivergence)
        );
    }
}
