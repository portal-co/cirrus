//! Spartan sumcheck rounds over the KoalaBear quintic extension.
//!
//! This mirrors the pinned upstream implementation's no-ZK DirectSparse
//! sumcheck schedule exactly:
//!
//! - the outer sumcheck runs cubic rounds over
//!   `eq(tau, x) * (az(x) * bz(x) - cz(x))` with an initial zero claim, and
//!   the honest round-0 constant term is the exact zero (every satisfied
//!   R1CS row vanishes on the Boolean hypercube);
//! - the inner sumcheck runs quadratic rounds over `z(y) * abc(y)`;
//! - round polynomials use the compact `[h(0), h(2), ...]` encoding with
//!   `h(1)` recovered from the running claim;
//! - the verifier replays rounds with generic Lagrange interpolation over
//!   the nodes `{0, 1, ..., degree}`.
//!
//! The prover keeps full dense tables rather than upstream's split-eq memory
//! optimization; both compute identical round polynomials, so transcripts
//! are byte-identical. The split-eq layout is deferred performance work.

use alloc::vec::Vec;

use crate::error::SpartanError;
use crate::poly::{CubicRoundPoly, EqPolynomial, QuadraticRoundPoly};
use crate::{FieldElement, KoalaBear, MultilinearPoint, PoseidonTranscript, QuinticExtension};

/// Outer sumcheck proof: one cubic round polynomial per constraint variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OuterSumcheckProof {
    /// Compact cubic rounds `[h(0), h(2), h(3)]`.
    pub rounds: Vec<CubicRoundPoly<QuinticExtension>>,
}

/// Inner sumcheck proof: one quadratic round polynomial per assignment
/// variable (including the witness/public selector coordinate).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InnerSumcheckProof {
    /// Compact quadratic rounds `[h(0), h(2)]`.
    pub rounds: Vec<QuadraticRoundPoly<QuinticExtension>>,
}

/// Prove the outer Spartan sumcheck.
///
/// `az`, `bz`, and `cz` are the base-field matrix-vector products of length
/// `num_cons`, and `tau` has one extension challenge per constraint
/// variable. Returns the proof, the challenge point `r_x`, and the final
/// claims `(az(r_x), bz(r_x), cz(r_x))`.
pub fn prove_outer(
    az: &[KoalaBear],
    bz: &[KoalaBear],
    cz: &[KoalaBear],
    tau: &MultilinearPoint<QuinticExtension>,
    transcript: &mut PoseidonTranscript,
) -> Result<
    (
        OuterSumcheckProof,
        MultilinearPoint<QuinticExtension>,
        (QuinticExtension, QuinticExtension, QuinticExtension),
    ),
    SpartanError,
> {
    let n = az.len();
    if !n.is_power_of_two() || bz.len() != n || cz.len() != n {
        return Err(SpartanError::InvalidRoundPolynomial);
    }
    if tau.0.len() != n.ilog2() as usize {
        return Err(SpartanError::InvalidRoundCount {
            expected: n.ilog2() as usize,
            found: tau.0.len(),
        });
    }

    let mut claim = QuinticExtension::ZERO;
    // Upstream observes the initial (zero) claim before any round.
    transcript.observe_quintic(claim);

    if tau.0.is_empty() {
        return Ok((
            OuterSumcheckProof { rounds: Vec::new() },
            MultilinearPoint(Vec::new()),
            (az[0].into(), bz[0].into(), cz[0].into()),
        ));
    }

    let mut eq = EqPolynomial::evals_from_point(&tau.0);
    let mut rounds = Vec::with_capacity(tau.0.len());
    let mut r_x = Vec::with_capacity(tau.0.len());

    // Round 0 runs over the base-field tables; the honest constant term is
    // the exact zero and the quadratic/cubic evaluations are computed in the
    // base field before extension weighting.
    let half = eq.len() / 2;
    let mut h2 = QuinticExtension::ZERO;
    let mut h3 = QuinticExtension::ZERO;
    for i in 0..half {
        let (_eq0, eq2, eq3) = extrapolated_eq(&eq, i, half);
        let (q2, q3) = outer_unweighted_base(az, bz, cz, i, half);
        h2 = h2 + eq2.mul_base(q2);
        h3 = h3 + eq3.mul_base(q3);
    }
    let round_poly = CubicRoundPoly([QuinticExtension::ZERO, h2, h3]);
    transcript.observe_quintic_slice(&round_poly.0);
    let r_0 = transcript.sample_quintic();
    claim = round_poly.evaluate_at(r_0, claim);
    rounds.push(round_poly);
    r_x.push(r_0);

    let mut az_tab = bind_base_to_ext(az, r_0)?;
    let mut bz_tab = bind_base_to_ext(bz, r_0)?;
    let mut cz_tab = bind_base_to_ext(cz, r_0)?;
    bind_ext_half(&mut eq, r_0)?;

    for _ in 1..tau.0.len() {
        let half = eq.len() / 2;
        let mut h0 = QuinticExtension::ZERO;
        let mut h2 = QuinticExtension::ZERO;
        let mut h3 = QuinticExtension::ZERO;
        for i in 0..half {
            let (eq0, eq2, eq3) = extrapolated_eq(&eq, i, half);
            let (q0, q2, q3) = outer_unweighted_ext(&az_tab, &bz_tab, &cz_tab, i, half);
            h0 = h0 + eq0 * q0;
            h2 = h2 + eq2 * q2;
            h3 = h3 + eq3 * q3;
        }

        let round_poly = CubicRoundPoly([h0, h2, h3]);
        transcript.observe_quintic_slice(&round_poly.0);
        let r_i = transcript.sample_quintic();

        claim = round_poly.evaluate_at(r_i, claim);
        rounds.push(round_poly);
        r_x.push(r_i);

        bind_ext_half(&mut az_tab, r_i)?;
        bind_ext_half(&mut bz_tab, r_i)?;
        bind_ext_half(&mut cz_tab, r_i)?;
        bind_ext_half(&mut eq, r_i)?;
    }

    Ok((
        OuterSumcheckProof { rounds },
        MultilinearPoint(r_x),
        (az_tab[0], bz_tab[0], cz_tab[0]),
    ))
}

/// Prove the inner Spartan sumcheck over `z(y) * abc(y)`.
///
/// `poly_abc` has length `2 * num_vars` and `z` is the full assignment
/// table of the same length. Returns the proof, the challenge point `r_y`,
/// and the final `z(r_y)` evaluation.
pub fn prove_inner(
    initial_claim: QuinticExtension,
    poly_abc: &[QuinticExtension],
    z: &[KoalaBear],
    transcript: &mut PoseidonTranscript,
) -> Result<
    (
        InnerSumcheckProof,
        MultilinearPoint<QuinticExtension>,
        QuinticExtension,
    ),
    SpartanError,
> {
    if poly_abc.is_empty() || !poly_abc.len().is_power_of_two() || z.len() != poly_abc.len() {
        return Err(SpartanError::InvalidRoundPolynomial);
    }
    let num_rounds = poly_abc.len().ilog2() as usize;

    let mut claim = initial_claim;
    transcript.observe_quintic(claim);

    let mut rounds = Vec::with_capacity(num_rounds);
    let mut r_y = Vec::with_capacity(num_rounds);
    let mut abc_tab = poly_abc.to_vec();

    if num_rounds == 0 {
        return Ok((
            InnerSumcheckProof { rounds },
            MultilinearPoint(r_y),
            z[0].into(),
        ));
    }

    // Round 0 runs with the base-field z table.
    let (h0, h2) = inner_coefficients_base(z, &abc_tab, claim);
    let round_poly = QuadraticRoundPoly([h0, h2]);
    transcript.observe_quintic_slice(&round_poly.0);
    let r_i = transcript.sample_quintic();
    claim = round_poly.evaluate_at(r_i, claim);
    rounds.push(round_poly);
    r_y.push(r_i);
    let mut z_tab = bind_base_to_ext(z, r_i)?;
    bind_ext_half(&mut abc_tab, r_i)?;

    for _ in 1..num_rounds {
        let (h0, h2) = inner_coefficients_ext(&z_tab, &abc_tab, claim);
        let round_poly = QuadraticRoundPoly([h0, h2]);
        transcript.observe_quintic_slice(&round_poly.0);
        let r_i = transcript.sample_quintic();
        claim = round_poly.evaluate_at(r_i, claim);
        rounds.push(round_poly);
        r_y.push(r_i);
        bind_ext_half(&mut z_tab, r_i)?;
        bind_ext_half(&mut abc_tab, r_i)?;
    }

    if z_tab.len() != 1 {
        return Err(SpartanError::SumcheckFailed);
    }

    Ok((
        InnerSumcheckProof { rounds },
        MultilinearPoint(r_y),
        z_tab[0],
    ))
}

/// Verify the outer sumcheck, replaying compact cubic rounds from the
/// initial claim. Returns the challenge point and the final claim, which
/// the caller must still relate to the R1CS evaluation claims.
pub fn verify_outer(
    proof: &OuterSumcheckProof,
    initial_claim: QuinticExtension,
    expected_rounds: usize,
    transcript: &mut PoseidonTranscript,
) -> Result<(MultilinearPoint<QuinticExtension>, QuinticExtension), SpartanError> {
    replay_compact_rounds(&proof.rounds, initial_claim, expected_rounds, 3, transcript)
}

/// Verify the inner sumcheck, replaying compact quadratic rounds from the
/// joint claim. Returns the challenge point and the final claim.
pub fn verify_inner(
    proof: &InnerSumcheckProof,
    initial_claim: QuinticExtension,
    expected_rounds: usize,
    transcript: &mut PoseidonTranscript,
) -> Result<(MultilinearPoint<QuinticExtension>, QuinticExtension), SpartanError> {
    replay_compact_rounds(&proof.rounds, initial_claim, expected_rounds, 2, transcript)
}

/// Upstream-compatible compact-round replay: observe the claim, then for
/// each round observe the compact evaluations, sample the challenge, and
/// advance the claim by Lagrange interpolation over `{0, 1, ..., degree}`.
fn replay_compact_rounds<R: AsRef<[QuinticExtension]>>(
    rounds: &[R],
    initial_claim: QuinticExtension,
    expected_rounds: usize,
    degree: usize,
    transcript: &mut PoseidonTranscript,
) -> Result<(MultilinearPoint<QuinticExtension>, QuinticExtension), SpartanError> {
    if rounds.len() != expected_rounds {
        return Err(SpartanError::InvalidRoundCount {
            expected: expected_rounds,
            found: rounds.len(),
        });
    }
    if degree == 0 {
        return Err(SpartanError::InvalidRoundPolynomial);
    }

    transcript.observe_quintic(initial_claim);

    let denominator_inverses = compact_round_denominator_inverses(degree);
    let mut claim = initial_claim;
    let mut point = Vec::with_capacity(expected_rounds);
    for round in rounds {
        let evals = round.as_ref();
        if evals.len() != degree {
            return Err(SpartanError::InvalidRoundPolynomial);
        }
        transcript.observe_quintic_slice(evals);
        let challenge = transcript.sample_quintic();
        claim = evaluate_compact_round(evals, claim, challenge, &denominator_inverses);
        point.push(challenge);
    }

    Ok((MultilinearPoint(point), claim))
}

/// Inverses of the Lagrange denominators for the nodes `{0, 1, ..., degree}`.
fn compact_round_denominator_inverses(degree: usize) -> Vec<QuinticExtension> {
    let point_count = degree + 1;
    let mut out = Vec::with_capacity(point_count);
    for i in 0..point_count {
        let x_i = QuinticExtension::from_u32(i as u32);
        let mut denominator = QuinticExtension::ONE;
        for (j, _) in (0..point_count).enumerate().filter(|(j, _)| *j != i) {
            let x_j = QuinticExtension::from_u32(j as u32);
            denominator = denominator * (x_i - x_j);
        }
        out.push(denominator.inverse().expect("distinct small nodes"));
    }
    out
}

/// Evaluate one compact round polynomial at `challenge`, recovering `h(1)`
/// as `claim - h(0)`.
fn evaluate_compact_round(
    evals: &[QuinticExtension],
    claim: QuinticExtension,
    challenge: QuinticExtension,
    denominator_inverses: &[QuinticExtension],
) -> QuinticExtension {
    let point_count = evals.len() + 1;
    debug_assert_eq!(denominator_inverses.len(), point_count);
    let mut out = QuinticExtension::ZERO;

    for i in 0..point_count {
        let y_i = match i {
            0 => evals[0],
            1 => claim - evals[0],
            _ => evals[i - 1],
        };
        let mut numerator = QuinticExtension::ONE;
        for (j, _) in (0..point_count).enumerate().filter(|(j, _)| *j != i) {
            let x_j = QuinticExtension::from_u32(j as u32);
            numerator = numerator * (challenge - x_j);
        }
        out = out + y_i * numerator * denominator_inverses[i];
    }

    out
}

/// Equality-table evaluations at nodes 0, 2, and 3 for pair `i`.
fn extrapolated_eq(
    eq: &[QuinticExtension],
    i: usize,
    half: usize,
) -> (QuinticExtension, QuinticExtension, QuinticExtension) {
    let eq0 = eq[i];
    let eq1 = eq[i + half];
    let delta = eq1 - eq0;
    (eq0, eq1 + delta, eq1 + delta + delta)
}

/// Quadratic/cubic univariate evaluations of `az * bz - cz` over the
/// base-field tables at pair `i`.
fn outer_unweighted_base(
    az: &[KoalaBear],
    bz: &[KoalaBear],
    cz: &[KoalaBear],
    i: usize,
    half: usize,
) -> (KoalaBear, KoalaBear) {
    let a0 = az[i];
    let a1 = az[i + half];
    let b0 = bz[i];
    let b1 = bz[i + half];
    let c0 = cz[i];
    let c1 = cz[i + half];

    let a_delta = a1 - a0;
    let b_delta = b1 - b0;
    let c_delta = c1 - c0;
    let a2 = a1 + a_delta;
    let b2 = b1 + b_delta;
    let c2 = c1 + c_delta;
    let a3 = a2 + a_delta;
    let b3 = b2 + b_delta;
    let c3 = c2 + c_delta;

    (a2 * b2 - c2, a3 * b3 - c3)
}

/// Univariate evaluations of `az * bz - cz` at nodes 0, 2, and 3 over the
/// extension-field tables at pair `i`.
fn outer_unweighted_ext(
    az: &[QuinticExtension],
    bz: &[QuinticExtension],
    cz: &[QuinticExtension],
    i: usize,
    half: usize,
) -> (QuinticExtension, QuinticExtension, QuinticExtension) {
    let a0 = az[i];
    let a1 = az[i + half];
    let b0 = bz[i];
    let b1 = bz[i + half];
    let c0 = cz[i];
    let c1 = cz[i + half];

    let a_delta = a1 - a0;
    let b_delta = b1 - b0;
    let c_delta = c1 - c0;
    let a2 = a1 + a_delta;
    let b2 = b1 + b_delta;
    let c2 = c1 + c_delta;
    let a3 = a2 + a_delta;
    let b3 = b2 + b_delta;
    let c3 = c2 + c_delta;

    (a0 * b0 - c0, a2 * b2 - c2, a3 * b3 - c3)
}

/// `(h(0), h(2))` for one inner round with the base-field `z` table.
fn inner_coefficients_base(
    z: &[KoalaBear],
    abc: &[QuinticExtension],
    claim: QuinticExtension,
) -> (QuinticExtension, QuinticExtension) {
    let half = z.len() / 2;
    let mut h0 = QuinticExtension::ZERO;
    let mut h_inf = QuinticExtension::ZERO;
    for i in 0..half {
        let z0 = z[i];
        let z_delta = z[i + half] - z0;
        let a0 = abc[i];
        let a_delta = abc[i + half] - a0;
        h0 = h0 + a0.mul_base(z0);
        h_inf = h_inf + a_delta.mul_base(z_delta);
    }
    quadratic_h2_from_hinf(h0, h_inf, claim)
}

/// `(h(0), h(2))` for one inner round with the extension-field `z` table.
fn inner_coefficients_ext(
    z: &[QuinticExtension],
    abc: &[QuinticExtension],
    claim: QuinticExtension,
) -> (QuinticExtension, QuinticExtension) {
    let half = z.len() / 2;
    let mut h0 = QuinticExtension::ZERO;
    let mut h_inf = QuinticExtension::ZERO;
    for i in 0..half {
        let z0 = z[i];
        let z_delta = z[i + half] - z0;
        let a0 = abc[i];
        let a_delta = abc[i + half] - a0;
        h0 = h0 + a0 * z0;
        h_inf = h_inf + a_delta * z_delta;
    }
    quadratic_h2_from_hinf(h0, h_inf, claim)
}

/// Recover `h(2)` from `h(0)`, the leading coefficient `h(inf)`, and the
/// running claim: `h(1) = claim - h(0)` and `h(2) = 2*h(1) - h(0) + 2*h(inf)`.
fn quadratic_h2_from_hinf(
    h0: QuinticExtension,
    h_inf: QuinticExtension,
    claim: QuinticExtension,
) -> (QuinticExtension, QuinticExtension) {
    let h1 = claim - h0;
    (h0, h1 + h1 - h0 + h_inf + h_inf)
}

/// Bind a base-field table's first variable to `r`, producing an
/// extension-field table of half the length.
fn bind_base_to_ext(
    table: &[KoalaBear],
    r: QuinticExtension,
) -> Result<Vec<QuinticExtension>, SpartanError> {
    if table.len() < 2 || table.len() % 2 != 0 {
        return Err(SpartanError::InvalidRoundPolynomial);
    }
    let half = table.len() / 2;
    let mut out = Vec::with_capacity(half);
    for i in 0..half {
        let lo = table[i];
        let delta = table[i + half] - lo;
        out.push(QuinticExtension::from(lo) + r.mul_base(delta));
    }
    Ok(out)
}

/// Bind an extension-field table's first variable to `r`, truncating to
/// half the length in place.
fn bind_ext_half(
    table: &mut Vec<QuinticExtension>,
    r: QuinticExtension,
) -> Result<(), SpartanError> {
    if table.len() < 2 || table.len() % 2 != 0 {
        return Err(SpartanError::InvalidRoundPolynomial);
    }
    let half = table.len() / 2;
    for i in 0..half {
        let lo = table[i];
        table[i] = lo + r * (table[i + half] - lo);
    }
    table.truncate(half);
    Ok(())
}
