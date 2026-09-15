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

/// A cubic sumcheck round polynomial in upstream compact encoding
/// `[h(0), h(2), h(3)]`; `h(1)` is recovered from the running claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CubicRoundPoly<F>(pub [F; 3]);

impl<F> AsRef<[F]> for CubicRoundPoly<F> {
    fn as_ref(&self) -> &[F] {
        &self.0
    }
}

impl<F: FieldElement> CubicRoundPoly<F> {
    /// `h(0)`.
    pub fn eval_at_zero(&self) -> F {
        self.0[0]
    }

    /// `h(2)`.
    pub fn eval_at_two(&self) -> F {
        self.0[1]
    }

    /// `h(3)`.
    pub fn eval_at_three(&self) -> F {
        self.0[2]
    }

    /// `h(1)`, recovered as `claim - h(0)`.
    pub fn eval_at_one_from_claim(&self, claim: F) -> F {
        claim - self.eval_at_zero()
    }

    /// Lagrange-interpolate `h` over nodes `{0, 1, 2, 3}` and evaluate at
    /// `r`, recovering `h(1)` from `claim`.
    pub fn evaluate_at(&self, r: F, claim: F) -> F {
        let h0 = self.eval_at_zero();
        let h1 = self.eval_at_one_from_claim(claim);
        let h2 = self.eval_at_two();
        let h3 = self.eval_at_three();

        let two = F::from_u32(2);
        let three = F::from_u32(3);
        let inv_two = two.inverse().expect("two is nonzero");
        let inv_six = F::from_u32(6).inverse().expect("six is nonzero");

        let l0 = -(r - F::ONE) * (r - two) * (r - three) * inv_six;
        let l1 = r * (r - two) * (r - three) * inv_two;
        let l2 = -r * (r - F::ONE) * (r - three) * inv_two;
        let l3 = r * (r - F::ONE) * (r - two) * inv_six;

        h0 * l0 + h1 * l1 + h2 * l2 + h3 * l3
    }
}

/// A quadratic sumcheck round polynomial in upstream compact encoding
/// `[h(0), h(2)]`; `h(1)` is recovered from the running claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuadraticRoundPoly<F>(pub [F; 2]);

impl<F> AsRef<[F]> for QuadraticRoundPoly<F> {
    fn as_ref(&self) -> &[F] {
        &self.0
    }
}

impl<F: FieldElement> QuadraticRoundPoly<F> {
    /// `h(0)`.
    pub fn eval_at_zero(&self) -> F {
        self.0[0]
    }

    /// `h(2)`.
    pub fn eval_at_two(&self) -> F {
        self.0[1]
    }

    /// `h(1)`, recovered as `claim - h(0)`.
    pub fn eval_at_one_from_claim(&self, claim: F) -> F {
        claim - self.eval_at_zero()
    }

    /// Lagrange-interpolate `h` over nodes `{0, 1, 2}` and evaluate at `r`,
    /// recovering `h(1)` from `claim`.
    pub fn evaluate_at(&self, r: F, claim: F) -> F {
        let h0 = self.eval_at_zero();
        let h1 = self.eval_at_one_from_claim(claim);
        let h2 = self.eval_at_two();

        let two = F::from_u32(2);
        let inv_two = two.inverse().expect("two is nonzero");
        let l0 = (r - F::ONE) * (r - two) * inv_two;
        let l1 = -r * (r - two);
        let l2 = r * (r - F::ONE) * inv_two;

        h0 * l0 + h1 * l1 + h2 * l2
    }
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
