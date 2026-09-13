//! Dense multilinear polynomial helpers used by Spartan and WHIR.

use alloc::vec;
use alloc::vec::Vec;

use crate::FieldElement;

/// A point in a multilinear evaluation domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultilinearPoint<F>(pub Vec<F>);

/// Dense evaluations of the multilinear equality polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EqPolynomial<F> {
    /// Point at which the equality polynomial is evaluated.
    pub point: MultilinearPoint<F>,
}

impl<F> EqPolynomial<F> {
    /// Construct the equality polynomial for `point`.
    pub fn new(point: MultilinearPoint<F>) -> Self {
        Self { point }
    }
}

impl<F: FieldElement> EqPolynomial<F> {
    /// Return `eq(x, point)` for every Boolean `x`, in the same lexicographic
    /// order as the pinned upstream implementation.
    pub fn evals_from_point(point: &[F]) -> Vec<F> {
        let mut evals = vec![F::ONE];
        for &r_i in point.iter().rev() {
            let half = evals.len();
            let mut next = Vec::with_capacity(half * 2);
            for &v in &evals {
                next.push(v * (F::ONE - r_i));
            }
            for &v in &evals {
                next.push(v * r_i);
            }
            evals = next;
        }
        evals
    }
}

/// Evaluate a dense power-of-two MLE table at `point`.
pub fn evaluate_mle_table<F: FieldElement>(table: &[F], point: &[F]) -> Result<F, PolyError> {
    if table.is_empty() || !table.len().is_power_of_two() {
        return Err(PolyError::InvalidTable);
    }
    if table.len() != (1usize << point.len()) {
        return Err(PolyError::InvalidPointLength);
    }

    let mut layer = table.to_vec();
    let mut active = layer.len();
    for &r_i in point {
        if active < 2 {
            return Err(PolyError::InvalidTable);
        }
        let half = active / 2;
        for i in 0..half {
            let lo = layer[i];
            let hi = layer[i + half];
            layer[i] = lo + r_i * (hi - lo);
        }
        active = half;
    }
    Ok(layer[0])
}

/// Polynomial substrate errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolyError {
    /// The dense table is empty or not a power of two.
    InvalidTable,
    /// The point length does not match the table dimension.
    InvalidPointLength,
}

impl core::fmt::Display for PolyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidTable => f.write_str("invalid multilinear evaluation table"),
            Self::InvalidPointLength => {
                f.write_str("multilinear point length does not match the table")
            }
        }
    }
}

impl core::error::Error for PolyError {}
