//! WHIR-internal quadratic sumcheck.
//!
//! Mirrors the pinned upstream `p3-sumcheck` pieces used by `p3-whir`:
//! compact `[h(0), h(inf)]` round data with `h(1) = claim - h(0)`, per-round
//! optional proof-of-work, equality/selection statement batching under one
//! challenge, and a scalar prefix-binding product polynomial. Only the
//! prefix variable order is implemented; that is the only order the
//! Spartan-WHIR adapter uses.

use alloc::vec::Vec;
use core::fmt;

use crate::poly::evaluate_mle_table;
use crate::{KoalaBear, PoseidonTranscript, QuinticExtension};

/// Lagrange extrapolation through `(0, e0)`, `(1, e1)`, `(inf, e_inf)`:
/// `e0 * (1 - r) + e1 * r + e_inf * r * (r - 1)`.
pub fn extrapolate_01inf(
    e0: QuinticExtension,
    e1: QuinticExtension,
    e_inf: QuinticExtension,
    r: QuinticExtension,
) -> QuinticExtension {
    e0 * (QuinticExtension::ONE - r) + e1 * r + e_inf * (r * (r - QuinticExtension::ONE))
}

/// Sumcheck failure modes, mirroring upstream `SumcheckError`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WhirSumcheckError {
    /// The proof carries the wrong number of rounds.
    RoundCountMismatch {
        /// Protocol-fixed round count.
        expected: usize,
        /// Round count carried by the proof.
        actual: usize,
    },
    /// The PoW witness count does not match the round count.
    PowWitnessCountMismatch {
        /// Expected witness count.
        expected: usize,
        /// Supplied witness count.
        actual: usize,
    },
    /// A proof-of-work witness failed the grinding check.
    InvalidPowWitness,
    /// The final sumcheck data is missing although rounds were expected.
    MissingSumcheckData {
        /// Expected round count.
        expected_rounds: usize,
    },
}

impl fmt::Display for WhirSumcheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RoundCountMismatch { expected, actual } => write!(
                f,
                "sumcheck round count mismatch: expected {expected}, got {actual}"
            ),
            Self::PowWitnessCountMismatch { expected, actual } => write!(
                f,
                "sumcheck PoW witness count mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidPowWitness => f.write_str("invalid sumcheck proof-of-work witness"),
            Self::MissingSumcheckData { expected_rounds } => write!(
                f,
                "missing sumcheck data for {expected_rounds} expected rounds"
            ),
        }
    }
}

impl core::error::Error for WhirSumcheckError {}

/// Sumcheck polynomial data: one `[h(0), h(inf)]` pair per round plus one
/// PoW witness per ground round.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SumcheckData {
    /// `[h(0), h(inf)]` per round; `h(1)` is derived as `claimed_sum - h(0)`.
    pub polynomial_evaluations: Vec<[QuinticExtension; 2]>,
    /// PoW witnesses for each sumcheck round.
    pub pow_witnesses: Vec<KoalaBear>,
}

impl SumcheckData {
    /// Commit round coefficients to the proof and transcript, optionally
    /// grind, and sample the next challenge. Mirrors upstream
    /// `SumcheckData::observe_and_sample`.
    pub fn observe_and_sample(
        &mut self,
        transcript: &mut PoseidonTranscript,
        c0: QuinticExtension,
        c_inf: QuinticExtension,
        pow_bits: usize,
    ) -> QuinticExtension {
        self.polynomial_evaluations.push([c0, c_inf]);
        transcript.observe_quintic_slice(&[c0, c_inf]);
        if pow_bits > 0 {
            let witness = transcript
                .grind(pow_bits)
                .expect("validated PoW bits grind successfully");
            self.pow_witnesses.push(witness);
        }
        transcript.sample_quintic()
    }

    /// Verify standard sumcheck rounds and return the folding randomness.
    /// Mirrors upstream `SumcheckData::verify_rounds`.
    pub fn verify_rounds(
        &self,
        transcript: &mut PoseidonTranscript,
        claimed_sum: &mut QuinticExtension,
        expected_rounds: usize,
        pow_bits: usize,
    ) -> Result<Vec<QuinticExtension>, WhirSumcheckError> {
        if self.polynomial_evaluations.len() != expected_rounds {
            return Err(WhirSumcheckError::RoundCountMismatch {
                expected: expected_rounds,
                actual: self.polynomial_evaluations.len(),
            });
        }
        if pow_bits > 0 && self.pow_witnesses.len() != self.polynomial_evaluations.len() {
            return Err(WhirSumcheckError::PowWitnessCountMismatch {
                expected: self.polynomial_evaluations.len(),
                actual: self.pow_witnesses.len(),
            });
        }
        let mut randomness = Vec::with_capacity(self.polynomial_evaluations.len());
        for (i, &[c0, c_inf]) in self.polynomial_evaluations.iter().enumerate() {
            transcript.observe_quintic_slice(&[c0, c_inf]);
            if pow_bits > 0 {
                let ok = transcript
                    .check_witness(pow_bits, self.pow_witnesses[i])
                    .map_err(|_| WhirSumcheckError::InvalidPowWitness)?;
                if !ok {
                    return Err(WhirSumcheckError::InvalidPowWitness);
                }
            }
            let r = transcript.sample_quintic();
            *claimed_sum = extrapolate_01inf(c0, *claimed_sum - c0, c_inf, r);
            randomness.push(r);
        }
        Ok(randomness)
    }
}

/// Verify the final sumcheck rounds, which may be absent entirely.
/// Mirrors upstream `verify_final_sumcheck_rounds`.
pub fn verify_final_sumcheck_rounds(
    final_sumcheck: Option<&SumcheckData>,
    transcript: &mut PoseidonTranscript,
    claimed_sum: &mut QuinticExtension,
    rounds: usize,
    pow_bits: usize,
) -> Result<Vec<QuinticExtension>, WhirSumcheckError> {
    if rounds == 0 {
        return Ok(Vec::new());
    }
    let sumcheck = final_sumcheck.ok_or(WhirSumcheckError::MissingSumcheckData {
        expected_rounds: rounds,
    })?;
    sumcheck.verify_rounds(transcript, claimed_sum, rounds, pow_bits)
}

/// Equality constraints `p(z_i) = s_i` over extension points.
#[derive(Clone, Debug, Default)]
pub struct EqStatement {
    num_variables: usize,
    points: Vec<Vec<QuinticExtension>>,
    evaluations: Vec<QuinticExtension>,
}

impl EqStatement {
    /// An empty statement over `num_variables` variables.
    pub fn initialize(num_variables: usize) -> Self {
        Self {
            num_variables,
            points: Vec::new(),
            evaluations: Vec::new(),
        }
    }

    /// Number of variables in the polynomial space.
    pub fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Number of constraints in the statement.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the statement holds no constraints.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Iterate over `(point, evaluation)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&[QuinticExtension], &QuinticExtension)> {
        self.points
            .iter()
            .map(Vec::as_slice)
            .zip(self.evaluations.iter())
    }

    /// Add one evaluation constraint `p(point) = eval`.
    pub fn add_evaluated_constraint(
        &mut self,
        point: Vec<QuinticExtension>,
        eval: QuinticExtension,
    ) {
        assert_eq!(point.len(), self.num_variables);
        self.points.push(point);
        self.evaluations.push(eval);
    }

    /// One weight per constraint: `eq(z_i, row)`.
    pub fn weights_at(&self, row: &[QuinticExtension]) -> Vec<QuinticExtension> {
        self.points
            .iter()
            .map(|point| crate::spartan::eq_point_eval(point, row))
            .collect()
    }
}

/// Selection constraints: `p(z_i) = s_i` where `z_i` is a univariate point
/// expanded through the power map.
#[derive(Clone, Debug, Default)]
pub struct SelectStatement {
    num_variables: usize,
    vars: Vec<KoalaBear>,
    evaluations: Vec<QuinticExtension>,
}

impl SelectStatement {
    /// An empty statement over `num_variables` variables.
    pub fn initialize(num_variables: usize) -> Self {
        Self {
            num_variables,
            vars: Vec::new(),
            evaluations: Vec::new(),
        }
    }

    /// Number of variables in the polynomial space.
    pub fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Number of constraints in the statement.
    pub fn len(&self) -> usize {
        self.vars.len()
    }

    /// Whether the statement holds no constraints.
    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }

    /// Add one constraint `p(var) = eval`.
    pub fn add_constraint(&mut self, var: KoalaBear, eval: QuinticExtension) {
        self.vars.push(var);
        self.evaluations.push(eval);
    }

    /// Build from explicit parts (used by the verifier).
    pub fn new(
        num_variables: usize,
        vars: Vec<KoalaBear>,
        evaluations: Vec<QuinticExtension>,
    ) -> Self {
        assert_eq!(vars.len(), evaluations.len());
        Self {
            num_variables,
            vars,
            evaluations,
        }
    }

    /// One weight per constraint: `select(var_i, row)`.
    pub fn weights_at(&self, row: &[QuinticExtension]) -> Vec<QuinticExtension> {
        self.vars.iter().map(|&var| eval_select(var, row)).collect()
    }

    /// Verify the statement against a polynomial's evaluation table via
    /// Horner's method: `p(z) = c_0 + z (c_1 + z (c_2 + ...))`.
    pub fn verify(&self, poly: &[QuinticExtension]) -> bool {
        self.vars
            .iter()
            .zip(&self.evaluations)
            .all(|(&var, &expected)| {
                let mut acc = QuinticExtension::ZERO;
                for &coefficient in poly.iter().rev() {
                    acc = acc.mul_base(var) + coefficient;
                }
                acc == expected
            })
    }
}

/// `select(var, point)`: product over coordinates of
/// `r * (var^{2^j} - 1) + 1`, iterating coordinates from the last down with
/// `var` squared per step, mirroring upstream `Point::eval_select`.
pub fn eval_select(var: KoalaBear, point: &[QuinticExtension]) -> QuinticExtension {
    let mut var = var;
    point
        .iter()
        .rev()
        .map(|&r| {
            // r * (var - 1) + 1 in mixed base/extension arithmetic.
            let term = r.mul_base(var) - r + QuinticExtension::ONE;
            var = var.square();
            term
        })
        .fold(QuinticExtension::ONE, |acc, term| acc * term)
}

/// One explicitly ordered group of evaluation constraints.
#[derive(Clone, Debug)]
pub enum Statements {
    /// Plain multilinear evaluations at concrete points.
    Eq(EqStatement),
    /// Selection-based evaluations through the power-map expansion.
    Select(SelectStatement),
}

impl Statements {
    /// Number of variables shared by the group.
    pub fn num_variables(&self) -> usize {
        match self {
            Self::Eq(statement) => statement.num_variables(),
            Self::Select(statement) => statement.num_variables(),
        }
    }

    /// Number of constraints in the group.
    pub fn len(&self) -> usize {
        match self {
            Self::Eq(statement) => statement.len(),
            Self::Select(statement) => statement.len(),
        }
    }

    /// Whether the group holds no constraints.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Eq(statement) => statement.is_empty(),
            Self::Select(statement) => statement.is_empty(),
        }
    }

    fn weights_at(&self, row: &[QuinticExtension]) -> Vec<QuinticExtension> {
        match self {
            Self::Eq(statement) => statement.weights_at(row),
            Self::Select(statement) => statement.weights_at(row),
        }
    }
}

/// An ordered batch of evaluation constraint groups folded under one
/// challenge, mirroring upstream `Constraint`.
#[derive(Clone, Debug)]
pub struct Constraint {
    num_variables: usize,
    statements: Vec<Statements>,
    challenge: QuinticExtension,
}

impl Constraint {
    /// Build a batch from explicitly ordered statement groups.
    pub fn new(
        challenge: QuinticExtension,
        num_variables: usize,
        statements: Vec<Statements>,
    ) -> Self {
        assert!(
            statements
                .iter()
                .all(|statement| statement.num_variables() == num_variables)
        );
        Self {
            num_variables,
            statements,
            challenge,
        }
    }

    /// The shared variable count.
    pub fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// The statement groups in batching order.
    pub fn statements(&self) -> &[Statements] {
        &self.statements
    }

    /// The batching challenge.
    pub fn challenge(&self) -> QuinticExtension {
        self.challenge
    }

    /// `challenge^exponent` via binary exponentiation.
    fn challenge_pow(&self, exponent: usize) -> QuinticExtension {
        let mut result = QuinticExtension::ONE;
        let mut base = self.challenge;
        let mut e = exponent;
        while e != 0 {
            if e & 1 != 0 {
                result = result * base;
            }
            base = base * base;
            e >>= 1;
        }
        result
    }

    /// Accumulate the challenge-weighted expected value across all groups.
    /// Mirrors upstream `Constraint::combine_evals`.
    pub fn combine_evals(&self, eval: &mut QuinticExtension) {
        let mut shift = 0;
        for statement in &self.statements {
            match statement {
                Statements::Eq(eq) => {
                    for (i, (_, &value)) in eq.iter().enumerate() {
                        *eval += self.challenge_pow(shift + i) * value;
                    }
                }
                Statements::Select(select) => {
                    for (i, &value) in select.evaluations.iter().enumerate() {
                        *eval += self.challenge_pow(shift + i) * value;
                    }
                }
            }
            shift += statement.len();
        }
    }

    /// Add the batched weight polynomial and expected value onto existing
    /// accumulators. Mirrors upstream `Constraint::combine` (scalar path).
    pub fn combine(&self, combined: &mut [QuinticExtension], eval: &mut QuinticExtension) {
        debug_assert_eq!(combined.len(), 1 << self.num_variables);
        let mut shift = 0;
        for statement in &self.statements {
            if statement.is_empty() {
                continue;
            }
            match statement {
                Statements::Eq(eq) => {
                    for (i, (point, &value)) in eq.iter().enumerate() {
                        let weight = self.challenge_pow(shift + i);
                        *eval += weight * value;
                        let table = crate::poly::EqPolynomial::evals_from_point(point);
                        for (out, entry) in combined.iter_mut().zip(table.iter()) {
                            *out += weight * *entry;
                        }
                    }
                }
                Statements::Select(select) => {
                    for (i, (&var, &value)) in
                        select.vars.iter().zip(&select.evaluations).enumerate()
                    {
                        let weight = self.challenge_pow(shift + i);
                        *eval += weight * value;
                        // select(var, x) = var^x over the hypercube index.
                        let mut power = QuinticExtension::ONE;
                        let base = QuinticExtension::from(var);
                        for out in combined.iter_mut() {
                            *out += weight * power;
                            power = power * base;
                        }
                    }
                }
            }
            shift += statement.len();
        }
    }

    /// Build the batched weight polynomial and expected value in fresh
    /// accumulators, mirroring upstream `Constraint::combine_new`.
    pub fn combine_new(&self) -> (Vec<QuinticExtension>, QuinticExtension) {
        let mut combined = alloc::vec![QuinticExtension::ZERO; 1 << self.num_variables];
        let mut eval = QuinticExtension::ZERO;
        self.combine(&mut combined, &mut eval);
        (combined, eval)
    }
}

/// Evaluate the batched weight polynomials of all constraints at the full
/// folding point, prefix order. Mirrors upstream
/// `VariableOrder::Prefix::eval_constraints_poly`.
pub fn eval_constraints_poly(
    constraints: &[Constraint],
    challenge: &[QuinticExtension],
) -> QuinticExtension {
    let reversed: Vec<QuinticExtension> = challenge.iter().rev().copied().collect();
    let mut total = QuinticExtension::ZERO;
    for constraint in constraints {
        let mut local: Vec<QuinticExtension> = reversed[..constraint.num_variables()].to_vec();
        local.reverse();
        let mut shift = 0;
        let mut acc = QuinticExtension::ZERO;
        for statement in constraint.statements() {
            let weights = statement.weights_at(&local);
            for (i, weight) in weights.into_iter().enumerate() {
                acc += constraint.challenge_pow(shift + i) * weight;
            }
            shift += statement.len();
        }
        total += acc;
    }
    total
}

/// Bind the prefix variable of an evaluation table to `r` in place:
/// `lo[i] += (hi[i] - lo[i]) * r`, then truncate to half.
pub fn fix_prefix_var(table: &mut Vec<QuinticExtension>, r: QuinticExtension) {
    let mid = table.len() / 2;
    debug_assert!(mid > 0);
    let (lo, hi) = table.split_at_mut(mid);
    for (lo, &hi) in lo.iter_mut().zip(hi.iter()) {
        *lo += (hi - *lo) * r;
    }
    table.truncate(mid);
}

/// Scalar prefix-binding product-polynomial prover, mirroring upstream
/// `SumcheckProver` over `ProductPolynomial` in the unpacked representation.
#[derive(Clone, Debug)]
pub struct WhirSumcheckProver {
    evals: Vec<QuinticExtension>,
    weights: Vec<QuinticExtension>,
    sum: QuinticExtension,
}

impl WhirSumcheckProver {
    /// A prover from evaluation and weight tables and their dot-product sum.
    pub fn new(
        evals: Vec<QuinticExtension>,
        weights: Vec<QuinticExtension>,
        sum: QuinticExtension,
    ) -> Self {
        debug_assert_eq!(evals.len(), weights.len());
        debug_assert!(evals.len().is_power_of_two());
        Self {
            evals,
            weights,
            sum,
        }
    }

    /// Remaining unbound variables.
    pub fn num_variables(&self) -> usize {
        self.evals.len().ilog2() as usize
    }

    /// The current claimed sum.
    pub fn claimed_sum(&self) -> QuinticExtension {
        self.sum
    }

    /// The current evaluation table.
    pub fn evals(&self) -> &[QuinticExtension] {
        &self.evals
    }

    /// Evaluate the current evaluation polynomial at a multilinear point.
    pub fn eval_at(&self, point: &[QuinticExtension]) -> QuinticExtension {
        evaluate_mle_table(&self.evals, point).expect("point length matches the table")
    }

    /// `(h(0), h(inf))` for the current round, prefix binding:
    /// `h(0) = sum lo products`, `h(inf) = sum of difference products`.
    fn round_coefficients(&self) -> (QuinticExtension, QuinticExtension) {
        let half = self.evals.len() / 2;
        let mut c0 = QuinticExtension::ZERO;
        let mut c_inf = QuinticExtension::ZERO;
        for i in 0..half {
            let (e_lo, e_hi) = (self.evals[i], self.evals[half + i]);
            let (w_lo, w_hi) = (self.weights[i], self.weights[half + i]);
            c0 += w_lo * e_lo;
            c_inf += (w_hi - w_lo) * (e_hi - e_lo);
        }
        (c0, c_inf)
    }

    /// Fold both tables by one challenge and update the claimed sum.
    fn fold_round(&mut self, c0: QuinticExtension, c_inf: QuinticExtension, r: QuinticExtension) {
        self.sum = extrapolate_01inf(c0, self.sum - c0, c_inf, r);
        fix_prefix_var(&mut self.evals, r);
        fix_prefix_var(&mut self.weights, r);
    }

    /// Run `folding_factor` rounds, optionally folding a new constraint into
    /// the weights first. Mirrors upstream
    /// `SumcheckProver::compute_sumcheck_polynomials`.
    pub fn compute_sumcheck_polynomials(
        &mut self,
        data: &mut SumcheckData,
        transcript: &mut PoseidonTranscript,
        folding_factor: usize,
        pow_bits: usize,
        constraint: Option<&Constraint>,
    ) -> Vec<QuinticExtension> {
        if let Some(constraint) = constraint {
            assert_eq!(constraint.num_variables(), self.num_variables());
            constraint.combine(&mut self.weights, &mut self.sum);
        }
        let mut randomness = Vec::with_capacity(folding_factor);
        for _ in 0..folding_factor {
            let (c0, c_inf) = self.round_coefficients();
            let r = data.observe_and_sample(transcript, c0, c_inf, pow_bits);
            self.fold_round(c0, c_inf, r);
            randomness.push(r);
        }
        randomness
    }
}
