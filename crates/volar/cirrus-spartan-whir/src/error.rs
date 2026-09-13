//! Exhaustive core error types for the Spartan reduction layer.

use core::fmt;

/// Why a sumcheck round, Spartan evaluation reduction, or DirectSparse
/// verification step failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpartanError {
    /// A proof carries the wrong number of sumcheck rounds.
    InvalidRoundCount {
        /// Expected round count.
        expected: usize,
        /// Round count found in the proof.
        found: usize,
    },
    /// A round polynomial or sumcheck table has an invalid shape.
    InvalidRoundPolynomial,
    /// The running sumcheck claim does not match the final evaluation.
    SumcheckFailed,
    /// The public input vector has the wrong length.
    InvalidPublicInputLength {
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        found: usize,
    },
    /// A witness or assignment vector has the wrong length.
    InvalidWitnessLength {
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        found: usize,
    },
    /// A required field inversion hit zero.
    NonInvertibleElement,
    /// The R1CS shape is invalid for this protocol layer.
    InvalidShape,
    /// A challenge-slot schedule does not fit the public input vector.
    InvalidChallengeSlots {
        /// Number of trailing challenge slots requested.
        slots: usize,
        /// Number of public inputs available.
        public_inputs: usize,
    },
}

impl fmt::Display for SpartanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRoundCount { expected, found } => {
                write!(
                    f,
                    "invalid sumcheck round count {found}, expected {expected}"
                )
            }
            Self::InvalidRoundPolynomial => f.write_str("invalid sumcheck round polynomial"),
            Self::SumcheckFailed => f.write_str("sumcheck verification failed"),
            Self::InvalidPublicInputLength { expected, found } => {
                write!(
                    f,
                    "invalid public input length {found}, expected {expected}"
                )
            }
            Self::InvalidWitnessLength { expected, found } => {
                write!(f, "invalid witness length {found}, expected {expected}")
            }
            Self::NonInvertibleElement => f.write_str("field element is not invertible"),
            Self::InvalidShape => f.write_str("invalid R1CS shape for Spartan reduction"),
            Self::InvalidChallengeSlots {
                slots,
                public_inputs,
            } => write!(
                f,
                "challenge schedule requests {slots} slots but only {public_inputs} public \
                 inputs exist"
            ),
        }
    }
}

impl core::error::Error for SpartanError {}
