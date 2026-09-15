//! Spartan's R1CS evaluation reduction in DirectSparse mode.
//!
//! This module mirrors the pinned upstream no-ZK DirectSparse schedule:
//!
//! 1. observe the opaque context bytes (upstream: domain-separator bytes via
//!    `F::from_u8`) and the public inputs;
//! 2. commit the zero-padded witness through the caller-supplied PCS;
//! 3. sample `tau`, run the outer sumcheck over
//!    `eq(tau, x) * (az(x) * bz(x) - cz(x))`, and check
//!    `eq(tau, r_x) * (a * b - c)`;
//! 4. batch the three claims with a joint challenge `r` and run the inner
//!    sumcheck over `z(y) * abc(y)`;
//! 5. verify `(a(r_x, r_y) + r * b(r_x, r_y) + r^2 * c(r_x, r_y)) * z(r_y)`
//!    from direct sparse-matrix evaluations, the claimed witness evaluation,
//!    and the public-input half;
//! 6. open the witness commitment at `r_y[1..]` through the PCS.
//!
//! The PCS boundary is a trait so the WHIR polynomial commitment (a later
//! slice) plugs in without changing this reduction or its transcript
//! schedule. No PCS implementation lives in this module; tests use an
//! explicit insecure stub.

use alloc::vec::Vec;
use core::fmt;

use crate::error::SpartanError;
use crate::poly::EqPolynomial;
use crate::sumcheck::{
    InnerSumcheckProof, OuterSumcheckProof, prove_inner, prove_outer, verify_inner, verify_outer,
};
use crate::{
    KoalaBear, MultilinearPoint, PoseidonTranscript, QuinticExtension, R1csError, R1csShape,
    R1csWitness, SparseMatrix, evaluate_mle_table,
};

/// The polynomial-commitment boundary used by the DirectSparse reduction.
///
/// The four hooks sit at fixed transcript positions: `commit` between the
/// context observation and the `tau` sample, and `open` after the inner
/// sumcheck; on the verifier side `verify_commitment` precedes the `tau`
/// sample and `verify_opening` runs after the inner evaluation check.
pub trait DirectSparsePcs {
    /// Public commitment object carried in the R1CS instance.
    type Commitment;
    /// Opening proof carried in the Spartan proof.
    type Proof;
    /// Verifier-side parsed commitment returned by the parse hook and
    /// consumed by the finalize hook, mirroring upstream's
    /// `verify_parse_commitment`/`verify_finalize` split.
    type ParsedCommitment;
    /// PCS-specific failure.
    type Error;

    /// Commit the zero-padded witness table of length `num_vars`.
    fn commit(
        &mut self,
        witness: &[KoalaBear],
        transcript: &mut PoseidonTranscript,
    ) -> Result<Self::Commitment, Self::Error>;

    /// Open the committed witness at `point` with claimed `value`.
    fn open(
        &mut self,
        point: &[QuinticExtension],
        value: QuinticExtension,
        transcript: &mut PoseidonTranscript,
    ) -> Result<Self::Proof, Self::Error>;

    /// Absorb and structurally validate a commitment on the verifier side,
    /// replaying every transcript step upstream performs before the
    /// Spartan challenges are sampled.
    fn verify_commitment(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        transcript: &mut PoseidonTranscript,
    ) -> Result<Self::ParsedCommitment, Self::Error>;

    /// Verify an opening of the parsed commitment at `point` with claimed
    /// `value`.
    fn verify_opening(
        &self,
        parsed: &Self::ParsedCommitment,
        point: &[QuinticExtension],
        value: QuinticExtension,
        proof: &Self::Proof,
        transcript: &mut PoseidonTranscript,
    ) -> Result<(), Self::Error>;
}

/// A DirectSparse R1CS instance: public inputs plus the witness commitment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csInstance<C> {
    /// Public input values in shape order.
    pub public_inputs: Vec<KoalaBear>,
    /// Commitment to the zero-padded witness.
    pub witness_commitment: C,
}

/// A DirectSparse Spartan proof up to the PCS opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectSparseProof<P> {
    /// Outer sumcheck rounds.
    pub outer_sumcheck: OuterSumcheckProof,
    /// Final outer claims `(az(r_x), bz(r_x), cz(r_x))`.
    pub outer_claims: (QuinticExtension, QuinticExtension, QuinticExtension),
    /// Inner sumcheck rounds.
    pub inner_sumcheck: InnerSumcheckProof,
    /// Claimed witness-table evaluation at `r_y[1..]`.
    pub witness_eval: QuinticExtension,
    /// PCS opening proof for `witness_eval`.
    pub pcs_proof: P,
}

/// Why DirectSparse proving or verification failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectSparseError<PcsError> {
    /// The Spartan reduction layer failed.
    Spartan(SpartanError),
    /// The polynomial-commitment layer failed.
    Pcs(PcsError),
    /// The supplied public challenge-slot values differ from the values
    /// derived from the post-commitment transcript state.
    ChallengeMismatch,
}

impl<PcsError: fmt::Display> fmt::Display for DirectSparseError<PcsError> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spartan(error) => write!(f, "spartan reduction error: {error}"),
            Self::Pcs(error) => write!(f, "pcs error: {error}"),
            Self::ChallengeMismatch => {
                f.write_str("public challenge slots do not match the transcript-derived values")
            }
        }
    }
}

/// A post-commitment challenge-slot schedule.
///
/// Profiles with transcript-derived public challenge slots (the Mode-B RAM
/// profile's `gamma`/`eta` slots) observe the challenge-independent public
/// input prefix before the witness commitment, derive the trailing
/// `slots` public values from the post-commitment transcript state, require
/// the supplied values to match the derived ones, and only then observe the
/// challenge tail. With zero slots the transcript schedule is exactly the
/// upstream one (the whole public vector is observed before the commitment).
pub struct ChallengeSlotSchedule<'a> {
    /// Number of trailing public-input slots derived after the commitment.
    pub slots: usize,
    /// Derives the slot values from the post-commitment transcript state.
    pub derive: &'a mut dyn FnMut(&mut PoseidonTranscript) -> Vec<KoalaBear>,
}

impl<PcsError: fmt::Debug + fmt::Display> core::error::Error for DirectSparseError<PcsError> {}

impl<PcsError> From<SpartanError> for DirectSparseError<PcsError> {
    fn from(error: SpartanError) -> Self {
        Self::Spartan(error)
    }
}

/// Prove the DirectSparse Spartan reduction for a canonical shape.
///
/// The shape must already be padded to power-of-two `num_cons` and
/// `num_vars` (see [`R1csShape::pad_regular`]); the witness may be shorter
/// than `num_vars` and is zero-padded. `context` is observed byte-by-byte
/// before the public inputs, matching upstream's Poseidon context binding;
/// it must uniquely identify the shape and security profile (the domain
/// separator in upstream terms).
pub fn prove_direct_sparse<P: DirectSparsePcs>(
    shape: &R1csShape<KoalaBear>,
    context: &[u8],
    public_inputs: &[KoalaBear],
    witness: &R1csWitness<KoalaBear>,
    pcs: &mut P,
    transcript: &mut PoseidonTranscript,
) -> Result<(R1csInstance<P::Commitment>, DirectSparseProof<P::Proof>), DirectSparseError<P::Error>>
{
    prove_direct_sparse_with_schedule(
        shape,
        context,
        public_inputs,
        witness,
        pcs,
        transcript,
        None,
    )
}

/// [`prove_direct_sparse`] with an optional post-commitment challenge-slot
/// schedule; see [`ChallengeSlotSchedule`].
#[allow(clippy::too_many_arguments)]
pub fn prove_direct_sparse_with_schedule<P: DirectSparsePcs>(
    shape: &R1csShape<KoalaBear>,
    context: &[u8],
    public_inputs: &[KoalaBear],
    witness: &R1csWitness<KoalaBear>,
    pcs: &mut P,
    transcript: &mut PoseidonTranscript,
    schedule: Option<ChallengeSlotSchedule<'_>>,
) -> Result<(R1csInstance<P::Commitment>, DirectSparseProof<P::Proof>), DirectSparseError<P::Error>>
{
    validate_canonical_shape(shape)?;
    if public_inputs.len() != shape.num_io {
        return Err(SpartanError::InvalidPublicInputLength {
            expected: shape.num_io,
            found: public_inputs.len(),
        }
        .into());
    }
    if witness.w.len() > shape.num_vars {
        return Err(SpartanError::InvalidWitnessLength {
            expected: shape.num_vars,
            found: witness.w.len(),
        }
        .into());
    }
    let slots = schedule.as_ref().map_or(0, |schedule| schedule.slots);
    if slots > public_inputs.len() {
        return Err(SpartanError::InvalidChallengeSlots {
            slots,
            public_inputs: public_inputs.len(),
        }
        .into());
    }
    let prefix_len = public_inputs.len() - slots;

    observe_context(transcript, context, &public_inputs[..prefix_len]);

    let witness_padded = shape.witness_to_mle(&witness.w).map_err(shape_error)?;
    let witness_commitment = pcs
        .commit(&witness_padded, transcript)
        .map_err(DirectSparseError::Pcs)?;

    if let Some(schedule) = schedule {
        if slots > 0 {
            let derived = (schedule.derive)(transcript);
            if derived != public_inputs[prefix_len..] {
                return Err(DirectSparseError::ChallengeMismatch);
            }
            transcript.observe_slice(&public_inputs[prefix_len..]);
        }
    }

    let z_full = build_z_full(witness_padded, shape.num_vars, public_inputs);
    let z_short = matrix_z_slice(&z_full, shape.num_vars, public_inputs.len())?;
    let (az, bz, cz) = shape.multiply_vec_unchecked(z_short).map_err(shape_error)?;

    let num_rounds_x = shape.num_cons.ilog2() as usize;
    let tau = MultilinearPoint(sample_quintic_vec(transcript, num_rounds_x));

    let (outer_sumcheck, r_x, outer_claims) = prove_outer(&az, &bz, &cz, &tau, transcript)?;

    transcript.observe_quintic_slice(&[outer_claims.0, outer_claims.1, outer_claims.2]);
    let r = transcript.sample_quintic();
    let claim_inner_joint = outer_claims.0 + r * outer_claims.1 + r * r * outer_claims.2;

    let t_x = EqPolynomial::evals_from_point(&r_x.0);
    let poly_abc = bind_row_vars_joint(shape, &t_x, r)?;
    let (inner_sumcheck, r_y, eval_z) =
        prove_inner(claim_inner_joint, &poly_abc, &z_full, transcript)?;

    let eval_x = evaluate_public_half(shape.num_vars, public_inputs, &r_y.0[1..])?;
    let witness_eval = recover_witness_eval(r_y.0[0], eval_z, eval_x)?;

    let pcs_proof = pcs
        .open(&r_y.0[1..], witness_eval, transcript)
        .map_err(DirectSparseError::Pcs)?;

    Ok((
        R1csInstance {
            public_inputs: public_inputs.to_vec(),
            witness_commitment,
        },
        DirectSparseProof {
            outer_sumcheck,
            outer_claims,
            inner_sumcheck,
            witness_eval,
            pcs_proof,
        },
    ))
}

/// Verify the DirectSparse Spartan reduction for a canonical shape.
pub fn verify_direct_sparse<P: DirectSparsePcs>(
    shape: &R1csShape<KoalaBear>,
    context: &[u8],
    instance: &R1csInstance<P::Commitment>,
    proof: &DirectSparseProof<P::Proof>,
    pcs: &P,
    transcript: &mut PoseidonTranscript,
) -> Result<(), DirectSparseError<P::Error>> {
    verify_direct_sparse_with_schedule(shape, context, instance, proof, pcs, transcript, None)
}

/// [`verify_direct_sparse`] with an optional post-commitment challenge-slot
/// schedule; see [`ChallengeSlotSchedule`]. The verifier recomputes the
/// challenge slots independently and rejects the proof when the supplied
/// public vector differs from the derived vector.
pub fn verify_direct_sparse_with_schedule<P: DirectSparsePcs>(
    shape: &R1csShape<KoalaBear>,
    context: &[u8],
    instance: &R1csInstance<P::Commitment>,
    proof: &DirectSparseProof<P::Proof>,
    pcs: &P,
    transcript: &mut PoseidonTranscript,
    schedule: Option<ChallengeSlotSchedule<'_>>,
) -> Result<(), DirectSparseError<P::Error>> {
    validate_canonical_shape(shape)?;
    if instance.public_inputs.len() != shape.num_io {
        return Err(SpartanError::InvalidPublicInputLength {
            expected: shape.num_io,
            found: instance.public_inputs.len(),
        }
        .into());
    }
    let slots = schedule.as_ref().map_or(0, |schedule| schedule.slots);
    if slots > instance.public_inputs.len() {
        return Err(SpartanError::InvalidChallengeSlots {
            slots,
            public_inputs: instance.public_inputs.len(),
        }
        .into());
    }
    let prefix_len = instance.public_inputs.len() - slots;

    observe_context(transcript, context, &instance.public_inputs[..prefix_len]);

    let parsed_commitment = pcs
        .verify_commitment(&instance.witness_commitment, &proof.pcs_proof, transcript)
        .map_err(DirectSparseError::Pcs)?;

    if let Some(schedule) = schedule {
        if slots > 0 {
            let derived = (schedule.derive)(transcript);
            if derived != instance.public_inputs[prefix_len..] {
                return Err(DirectSparseError::ChallengeMismatch);
            }
            transcript.observe_slice(&instance.public_inputs[prefix_len..]);
        }
    }

    let num_rounds_x = shape.num_cons.ilog2() as usize;
    let tau = MultilinearPoint(sample_quintic_vec(transcript, num_rounds_x));

    let (r_x, final_outer_claim) = verify_outer(
        &proof.outer_sumcheck,
        QuinticExtension::ZERO,
        num_rounds_x,
        transcript,
    )?;

    let expected_outer = eq_point_eval(&tau.0, &r_x.0)
        * (proof.outer_claims.0 * proof.outer_claims.1 - proof.outer_claims.2);
    if final_outer_claim != expected_outer {
        return Err(SpartanError::SumcheckFailed.into());
    }

    transcript.observe_quintic_slice(&[
        proof.outer_claims.0,
        proof.outer_claims.1,
        proof.outer_claims.2,
    ]);
    let r = transcript.sample_quintic();
    let claim_inner_joint =
        proof.outer_claims.0 + r * proof.outer_claims.1 + r * r * proof.outer_claims.2;

    let num_rounds_y = shape.num_vars.ilog2() as usize + 1;
    let (r_y, inner_final_claim) = verify_inner(
        &proof.inner_sumcheck,
        claim_inner_joint,
        num_rounds_y,
        transcript,
    )?;

    let t_x = EqPolynomial::evals_from_point(&r_x.0);
    let t_y = EqPolynomial::evals_from_point(&r_y.0);
    let (eval_a, eval_b, eval_c) = evaluate_with_tables(shape, &t_x, &t_y)?;

    let eval_x = evaluate_public_half(shape.num_vars, &instance.public_inputs, &r_y.0[1..])?;
    let eval_z = (QuinticExtension::ONE - r_y.0[0]) * proof.witness_eval + r_y.0[0] * eval_x;
    let expected_inner = (eval_a + r * eval_b + r * r * eval_c) * eval_z;
    if inner_final_claim != expected_inner {
        return Err(SpartanError::SumcheckFailed.into());
    }

    pcs.verify_opening(
        &parsed_commitment,
        &r_y.0[1..],
        proof.witness_eval,
        &proof.pcs_proof,
        transcript,
    )
    .map_err(DirectSparseError::Pcs)?;

    Ok(())
}

/// Observe context bytes (`F::from_u8` per byte) followed by the public
/// inputs, matching upstream's Poseidon `observe_spartan_context`.
pub fn observe_context(
    transcript: &mut PoseidonTranscript,
    context: &[u8],
    public_inputs: &[KoalaBear],
) {
    for &byte in context {
        transcript.observe(KoalaBear::from_u64(u64::from(byte)));
    }
    transcript.observe_slice(public_inputs);
}

/// Build `witness || [1] || public` zero-padded to `2 * num_vars`.
pub fn build_z_full(
    mut witness_half: Vec<KoalaBear>,
    num_vars: usize,
    public_inputs: &[KoalaBear],
) -> Vec<KoalaBear> {
    debug_assert_eq!(witness_half.len(), num_vars);
    let public_offset = witness_half.len();
    witness_half.resize(public_offset + num_vars, KoalaBear::ZERO);
    witness_half[public_offset] = KoalaBear::ONE;
    for (i, &value) in public_inputs.iter().enumerate() {
        witness_half[public_offset + i + 1] = value;
    }
    witness_half
}

/// The matrix multiplication slice of `z_full`: `num_vars + 1 + num_io`.
pub fn matrix_z_slice(
    z_full: &[KoalaBear],
    num_vars: usize,
    public_input_len: usize,
) -> Result<&[KoalaBear], SpartanError> {
    let len = num_vars
        .checked_add(public_input_len)
        .and_then(|n| n.checked_add(1))
        .ok_or(SpartanError::InvalidWitnessLength {
            expected: usize::MAX,
            found: z_full.len(),
        })?;
    z_full.get(..len).ok_or(SpartanError::InvalidWitnessLength {
        expected: len,
        found: z_full.len(),
    })
}

/// Compute `eq(r_x, row) * (A + r * B + r^2 * C)` bound into the column
/// variables: the inner sumcheck's `poly_abc` table of length
/// `2 * num_vars`.
pub fn bind_row_vars_joint(
    shape: &R1csShape<KoalaBear>,
    eq_rx: &[QuinticExtension],
    r: QuinticExtension,
) -> Result<Vec<QuinticExtension>, SpartanError> {
    if eq_rx.len() != shape.num_cons {
        return Err(SpartanError::InvalidWitnessLength {
            expected: shape.num_cons,
            found: eq_rx.len(),
        });
    }
    let width = shape
        .num_vars
        .checked_mul(2)
        .ok_or(SpartanError::InvalidShape)?;

    let mut out = alloc::vec![QuinticExtension::ZERO; width];
    accumulate_bound_rows(&shape.a, eq_rx, &mut out)?;
    let eq_b = scale_eq_table(eq_rx, r);
    accumulate_bound_rows(&shape.b, &eq_b, &mut out)?;
    let eq_c = scale_eq_table(eq_rx, r * r);
    accumulate_bound_rows(&shape.c, &eq_c, &mut out)?;
    Ok(out)
}

/// Evaluate the three sparse matrices at `(r_x, r_y)` from precomputed
/// equality tables, returning `(A(r_x, r_y), B(r_x, r_y), C(r_x, r_y))`.
pub fn evaluate_with_tables(
    shape: &R1csShape<KoalaBear>,
    t_x: &[QuinticExtension],
    t_y: &[QuinticExtension],
) -> Result<(QuinticExtension, QuinticExtension, QuinticExtension), SpartanError> {
    let assignment_cols = shape.assignment_len().map_err(shape_error)?;
    if t_x.len() != shape.num_cons || t_y.len() < assignment_cols {
        return Err(SpartanError::InvalidWitnessLength {
            expected: shape.num_cons,
            found: t_x.len(),
        });
    }
    Ok((
        evaluate_sparse_matrix_with_tables(&shape.a, t_x, t_y)?,
        evaluate_sparse_matrix_with_tables(&shape.b, t_x, t_y)?,
        evaluate_sparse_matrix_with_tables(&shape.c, t_x, t_y)?,
    ))
}

/// Evaluate the public half `[1 | public_inputs | 0..]` at a point of
/// length `log2(num_vars)`, exploiting the zero padding.
pub fn evaluate_public_half(
    num_vars: usize,
    public_inputs: &[KoalaBear],
    point: &[QuinticExtension],
) -> Result<QuinticExtension, SpartanError> {
    if num_vars == 0 || !num_vars.is_power_of_two() {
        return Err(SpartanError::InvalidRoundPolynomial);
    }
    if point.len() != num_vars.ilog2() as usize {
        return Err(SpartanError::InvalidRoundPolynomial);
    }
    if public_inputs.len() >= num_vars {
        return Err(SpartanError::InvalidPublicInputLength {
            expected: num_vars - 1,
            found: public_inputs.len(),
        });
    }

    let active_len = public_inputs
        .len()
        .checked_add(1)
        .and_then(usize::checked_next_power_of_two)
        .ok_or(SpartanError::InvalidPublicInputLength {
            expected: usize::MAX,
            found: public_inputs.len(),
        })?;
    let active_vars = active_len.ilog2() as usize;
    let fixed_zero_vars = point.len() - active_vars;
    let fixed_zero_weight = point[..fixed_zero_vars]
        .iter()
        .fold(QuinticExtension::ONE, |acc, &r| {
            acc * (QuinticExtension::ONE - r)
        });

    let mut table = alloc::vec![QuinticExtension::ZERO; active_len];
    table[0] = QuinticExtension::ONE;
    for (i, &value) in public_inputs.iter().enumerate() {
        table[i + 1] = QuinticExtension::from(value);
    }
    let active_eval = evaluate_mle_table(&table, &point[fixed_zero_vars..])
        .map_err(|_| SpartanError::InvalidRoundPolynomial)?;
    Ok(fixed_zero_weight * active_eval)
}

/// Recover the witness-half evaluation from `z(r_y)`, the selector
/// `r_y[0]`, and the public-half evaluation.
pub fn recover_witness_eval(
    r0: QuinticExtension,
    eval_z: QuinticExtension,
    eval_x: QuinticExtension,
) -> Result<QuinticExtension, SpartanError> {
    let denominator = QuinticExtension::ONE - r0;
    let denominator_inverse = denominator
        .inverse()
        .map_err(|_| SpartanError::NonInvertibleElement)?;
    Ok((eval_z - r0 * eval_x) * denominator_inverse)
}

/// Evaluate `eq(a, b)` for two equal-length points.
pub fn eq_point_eval(a: &[QuinticExtension], b: &[QuinticExtension]) -> QuinticExtension {
    a.iter()
        .zip(b.iter())
        .fold(QuinticExtension::ONE, |acc, (&x, &y)| {
            acc * ((QuinticExtension::ONE - x) * (QuinticExtension::ONE - y) + x * y)
        })
}

/// Sample `len` quintic challenges from the transcript.
fn sample_quintic_vec(transcript: &mut PoseidonTranscript, len: usize) -> Vec<QuinticExtension> {
    (0..len).map(|_| transcript.sample_quintic()).collect()
}

/// Require power-of-two canonical shape dimensions.
fn validate_canonical_shape(shape: &R1csShape<KoalaBear>) -> Result<(), SpartanError> {
    shape.validate().map_err(shape_error)?;
    if !shape.num_cons.is_power_of_two()
        || shape.num_vars == 0
        || !shape.num_vars.is_power_of_two()
        || shape.num_io >= shape.num_vars
    {
        return Err(SpartanError::InvalidShape);
    }
    Ok(())
}

fn scale_eq_table(eq_rx: &[QuinticExtension], scale: QuinticExtension) -> Vec<QuinticExtension> {
    eq_rx.iter().map(|&value| scale * value).collect()
}

fn accumulate_bound_rows(
    matrix: &SparseMatrix<KoalaBear>,
    eq_rx: &[QuinticExtension],
    out: &mut [QuinticExtension],
) -> Result<(), SpartanError> {
    if out.len() < matrix.num_cols {
        return Err(SpartanError::InvalidShape);
    }
    for entry in &matrix.entries {
        out[entry.col] = out[entry.col] + eq_rx[entry.row].mul_base(entry.val);
    }
    Ok(())
}

fn evaluate_sparse_matrix_with_tables(
    matrix: &SparseMatrix<KoalaBear>,
    t_x: &[QuinticExtension],
    t_y: &[QuinticExtension],
) -> Result<QuinticExtension, SpartanError> {
    if t_x.len() != matrix.num_rows || t_y.len() < matrix.num_cols {
        return Err(SpartanError::InvalidWitnessLength {
            expected: matrix.num_rows,
            found: t_x.len(),
        });
    }
    let mut acc = QuinticExtension::ZERO;
    for entry in &matrix.entries {
        acc = acc + (t_x[entry.row] * t_y[entry.col]).mul_base(entry.val);
    }
    Ok(acc)
}

fn shape_error(error: R1csError) -> SpartanError {
    match error {
        R1csError::InvalidWitnessLength { expected, found } => {
            SpartanError::InvalidWitnessLength { expected, found }
        }
        _ => SpartanError::InvalidShape,
    }
}
