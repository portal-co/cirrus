//! Small field trait shared by the polynomial and R1CS substrates.
//!
//! This is intentionally much narrower than Plonky3's `Field` hierarchy. It
//! exposes only the operations used by the current `no_std` implementation.

use core::ops::{Add, Mul, Neg, Sub};

use crate::{KoalaBear, QuinticExtension};

/// Minimal field operations required by polynomial and R1CS code.
pub trait FieldElement:
    Copy + Eq + Add<Output = Self> + Sub<Output = Self> + Neg<Output = Self> + Mul<Output = Self>
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;

    /// Reduce an unsigned small integer into the field.
    fn from_u32(value: u32) -> Self;

    /// Multiplicative inverse, or `None` for zero.
    fn inverse(self) -> Option<Self>;
}

impl FieldElement for KoalaBear {
    const ZERO: Self = KoalaBear::ZERO;
    const ONE: Self = KoalaBear::ONE;

    fn from_u32(value: u32) -> Self {
        Self::from_u64(u64::from(value))
    }

    fn inverse(self) -> Option<Self> {
        Self::inverse(self).ok()
    }
}

impl FieldElement for QuinticExtension {
    const ZERO: Self = QuinticExtension::ZERO;
    const ONE: Self = QuinticExtension::ONE;

    fn from_u32(value: u32) -> Self {
        Self::new([
            KoalaBear::from_u64(u64::from(value)),
            KoalaBear::ZERO,
            KoalaBear::ZERO,
            KoalaBear::ZERO,
            KoalaBear::ZERO,
        ])
    }

    fn inverse(self) -> Option<Self> {
        Self::inverse(self).ok()
    }
}
