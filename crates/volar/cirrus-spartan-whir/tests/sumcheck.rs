use cirrus_spartan_whir::{
    DirectSparsePcs, DirectSparseProof, KoalaBear, PoseidonDigest, PoseidonTranscript,
    QuinticExtension, R1csInstance, R1csShape, R1csWitness, SparseMatEntry, SparseMatrix,
    SpartanError, bind_row_vars_joint, build_z_full, eq_point_eval, evaluate_mle_table,
    evaluate_public_half, evaluate_with_tables, matrix_z_slice, poseidon_hash_fixed,
    prove_direct_sparse, recover_witness_eval, verify_direct_sparse,
};

fn kb(value: u32) -> KoalaBear {
    KoalaBear::from_u64(u64::from(value))
}

fn ext(seed: u32) -> QuinticExtension {
    QuinticExtension::from_canonical_coefficients(core::array::from_fn(|index| {
        seed.wrapping_mul((index as u32) + 3)
            .wrapping_add(17 + index as u32)
    }))
    .unwrap()
}

fn entry<F: cirrus_spartan_whir::FieldElement>(
    row: usize,
    col: usize,
    val: u32,
) -> SparseMatEntry<F> {
    SparseMatEntry {
        row,
        col,
        val: F::from_u32(val),
    }
}

/// A satisfied chain `w1 = w0^2`, `w2 = w1 * x0`, `w3 = w2 * x1`, plus one
/// constant-check row. Padded dimensions: 4 constraints, 4 witness columns.
fn chain_shape() -> R1csShape<KoalaBear> {
    R1csShape {
        num_cons: 4,
        num_vars: 4,
        num_io: 2,
        a: SparseMatrix {
            num_rows: 4,
            num_cols: 7,
            entries: vec![
                entry(0, 0, 1),
                entry(1, 1, 1),
                entry(2, 2, 1),
                entry(3, 3, 1),
            ],
        },
        b: SparseMatrix {
            num_rows: 4,
            num_cols: 7,
            entries: vec![
                entry(0, 0, 1),
                entry(1, 5, 1),
                entry(2, 6, 1),
                entry(3, 4, 1),
            ],
        },
        c: SparseMatrix {
            num_rows: 4,
            num_cols: 7,
            entries: vec![
                entry(0, 1, 1),
                entry(1, 2, 1),
                entry(2, 3, 1),
                entry(3, 4, 60),
            ],
        },
    }
}

fn chain_witness() -> R1csWitness<KoalaBear> {
    R1csWitness {
        w: vec![kb(2), kb(4), kb(12), kb(60)],
    }
}

fn chain_public() -> Vec<KoalaBear> {
    vec![kb(3), kb(5)]
}

/// A one-constraint edge fixture: zero outer rounds, one inner round.
fn edge_shape() -> R1csShape<KoalaBear> {
    R1csShape {
        num_cons: 1,
        num_vars: 1,
        num_io: 0,
        a: SparseMatrix {
            num_rows: 1,
            num_cols: 2,
            entries: vec![entry(0, 0, 1)],
        },
        b: SparseMatrix {
            num_rows: 1,
            num_cols: 2,
            entries: vec![entry(0, 0, 1)],
        },
        c: SparseMatrix {
            num_rows: 1,
            num_cols: 2,
            entries: vec![entry(0, 0, 1)],
        },
    }
}

const CONTEXT: &[u8] = b"cirrus-direct-sparse-test-context-v1";

/// Insecure PCS stub used only to pin the reduction's transcript schedule.
///
/// The commitment is a Poseidon hash of the whole witness; `open`
/// recomputes the claimed evaluation from the stored witness so honest runs
/// are internally consistent. This is NOT a commitment scheme: it provides
/// no succinctness, no position binding for the verifier, and no hiding.
/// The real WHIR PCS replaces it in a later slice.
#[derive(Default)]
struct DevNullPcs {
    witness: Vec<KoalaBear>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DevNullPcsError {
    OpeningMismatch,
}

impl core::fmt::Display for DevNullPcsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OpeningMismatch => f.write_str("dev-null pcs opening mismatch"),
        }
    }
}

impl DirectSparsePcs for DevNullPcs {
    type Commitment = PoseidonDigest;
    type Proof = ();
    type Error = DevNullPcsError;

    fn commit(
        &mut self,
        witness: &[KoalaBear],
        transcript: &mut PoseidonTranscript,
    ) -> Result<Self::Commitment, Self::Error> {
        let digest = poseidon_hash_fixed(witness);
        transcript.observe_slice(&digest);
        self.witness = witness.to_vec();
        Ok(digest)
    }

    fn open(
        &mut self,
        point: &[QuinticExtension],
        value: QuinticExtension,
        transcript: &mut PoseidonTranscript,
    ) -> Result<Self::Proof, Self::Error> {
        transcript.observe_quintic(value);
        let table: Vec<QuinticExtension> = self.witness.iter().map(|&v| v.into()).collect();
        let expected = evaluate_mle_table(&table, point).expect("dev-null point length");
        if expected != value {
            return Err(DevNullPcsError::OpeningMismatch);
        }
        Ok(())
    }

    fn verify_commitment(
        &self,
        commitment: &Self::Commitment,
        transcript: &mut PoseidonTranscript,
    ) -> Result<(), Self::Error> {
        transcript.observe_slice(commitment);
        Ok(())
    }

    fn verify_opening(
        &self,
        _commitment: &Self::Commitment,
        _point: &[QuinticExtension],
        value: QuinticExtension,
        _proof: &Self::Proof,
        transcript: &mut PoseidonTranscript,
    ) -> Result<(), Self::Error> {
        transcript.observe_quintic(value);
        Ok(())
    }
}

type DevProof = DirectSparseProof<()>;
type DevInstance = R1csInstance<PoseidonDigest>;

fn prove_chain() -> (DevInstance, DevProof) {
    let shape = chain_shape();
    let mut pcs = DevNullPcs::default();
    let mut transcript = PoseidonTranscript::new();
    prove_direct_sparse(
        &shape,
        CONTEXT,
        &chain_public(),
        &chain_witness(),
        &mut pcs,
        &mut transcript,
    )
    .unwrap()
}

fn verify_chain(
    shape: &R1csShape<KoalaBear>,
    instance: &DevInstance,
    proof: &DevProof,
) -> Result<(), String> {
    let pcs = DevNullPcs::default();
    let mut transcript = PoseidonTranscript::new();
    verify_direct_sparse(&shape, CONTEXT, instance, proof, &pcs, &mut transcript)
        .map_err(|error| format!("{error:?}"))
}

#[test]
fn direct_sparse_reduction_accepts_honest_proofs() {
    let (instance, proof) = prove_chain();
    verify_chain(&chain_shape(), &instance, &proof).unwrap();

    let edge = edge_shape();
    let witness = R1csWitness { w: vec![kb(1)] };
    let mut pcs = DevNullPcs::default();
    let mut transcript = PoseidonTranscript::new();
    let (instance, proof) =
        prove_direct_sparse(&edge, CONTEXT, &[], &witness, &mut pcs, &mut transcript).unwrap();
    assert!(proof.outer_sumcheck.rounds.is_empty());
    assert_eq!(proof.inner_sumcheck.rounds.len(), 1);
    let pcs = DevNullPcs::default();
    let mut transcript = PoseidonTranscript::new();
    verify_direct_sparse(&edge, CONTEXT, &instance, &proof, &pcs, &mut transcript).unwrap();
}

#[test]
fn reduction_helpers_match_reference_semantics() {
    let shape = chain_shape();
    let witness_padded = shape.witness_to_mle(&chain_witness().w).unwrap();
    let z_full = build_z_full(witness_padded, shape.num_vars, &chain_public());
    assert_eq!(z_full.len(), 8);
    assert_eq!(
        z_full,
        vec![kb(2), kb(4), kb(12), kb(60), kb(1), kb(3), kb(5), kb(0)]
    );
    let z_short = matrix_z_slice(&z_full, shape.num_vars, 2).unwrap();
    assert_eq!(z_short.len(), 7);
    assert!(matrix_z_slice(&z_full[..6], shape.num_vars, 2).is_err());

    // evaluate_public_half equals the dense MLE of [1 | public | 0..].
    let point = vec![ext(11), ext(23)];
    let dense = vec![
        QuinticExtension::ONE,
        kb(3).into(),
        kb(5).into(),
        QuinticExtension::ZERO,
    ];
    let expected = evaluate_mle_table(&dense, &point).unwrap();
    assert_eq!(
        evaluate_public_half(shape.num_vars, &chain_public(), &point).unwrap(),
        expected
    );
    assert!(evaluate_public_half(shape.num_vars, &chain_public(), &point[..1]).is_err());
    assert!(evaluate_public_half(3, &chain_public(), &point).is_err());

    // eq_point_eval matches the dense equality-table evaluation.
    let tau = vec![ext(31), ext(47)];
    let r = vec![ext(59), ext(67)];
    let eq_table = cirrus_spartan_whir::EqPolynomial::evals_from_point(&tau);
    assert_eq!(
        eq_point_eval(&tau, &r),
        evaluate_mle_table(&eq_table, &r).unwrap()
    );

    // recover_witness_eval inverts z = (1 - r0) * w + r0 * x.
    let r0 = ext(71);
    let witness_eval = ext(83);
    let eval_x = ext(97);
    let eval_z = (QuinticExtension::ONE - r0) * witness_eval + r0 * eval_x;
    assert_eq!(
        recover_witness_eval(r0, eval_z, eval_x).unwrap(),
        witness_eval
    );
    assert_eq!(
        recover_witness_eval(QuinticExtension::ONE, eval_z, eval_x),
        Err(SpartanError::NonInvertibleElement)
    );

    // bind_row_vars_joint equals the dense row-fold of each matrix.
    let t_x = cirrus_spartan_whir::EqPolynomial::evals_from_point(&point);
    let r = ext(101);
    let poly_abc = bind_row_vars_joint(&shape, &t_x, r).unwrap();
    assert_eq!(poly_abc.len(), 8);

    let t_y = cirrus_spartan_whir::EqPolynomial::evals_from_point(&vec![ext(3), ext(5), ext(7)]);
    let (eval_a, eval_b, eval_c) = evaluate_with_tables(&shape, &t_x, &t_y).unwrap();
    let r_y_full = vec![ext(103), ext(107), ext(109)];
    let t_y_check = cirrus_spartan_whir::EqPolynomial::evals_from_point(&r_y_full);
    let (check_a, check_b, check_c) = evaluate_with_tables(&shape, &t_x, &t_y_check).unwrap();
    assert_ne!((eval_a, eval_b, eval_c), (check_a, check_b, check_c));

    // poly_abc(r_y) equals A + r * B + r^2 * C at (r_x, r_y).
    let r_y = vec![ext(113), ext(127), ext(131)];
    let t_y = cirrus_spartan_whir::EqPolynomial::evals_from_point(&r_y);
    let (eval_a, eval_b, eval_c) = evaluate_with_tables(&shape, &t_x, &t_y).unwrap();
    let poly_eval = evaluate_mle_table(&poly_abc, &r_y).unwrap();
    assert_eq!(poly_eval, eval_a + r * eval_b + r * r * eval_c);
}

#[test]
fn direct_sparse_rejects_mutated_round_polynomials() {
    let shape = chain_shape();
    let (instance, proof) = prove_chain();

    for round in 0..proof.outer_sumcheck.rounds.len() {
        for slot in 0..3 {
            let mut mutated = proof.clone();
            mutated.outer_sumcheck.rounds[round].0[slot] =
                mutated.outer_sumcheck.rounds[round].0[slot] + QuinticExtension::ONE;
            assert!(
                verify_chain(&shape, &instance, &mutated).is_err(),
                "outer round {round} slot {slot} mutation accepted"
            );
        }
    }
    for round in 0..proof.inner_sumcheck.rounds.len() {
        for slot in 0..2 {
            let mut mutated = proof.clone();
            mutated.inner_sumcheck.rounds[round].0[slot] =
                mutated.inner_sumcheck.rounds[round].0[slot] + QuinticExtension::ONE;
            assert!(
                verify_chain(&shape, &instance, &mutated).is_err(),
                "inner round {round} slot {slot} mutation accepted"
            );
        }
    }
}

#[test]
fn direct_sparse_rejects_mutated_claims_evals_and_instances() {
    let shape = chain_shape();
    let (instance, proof) = prove_chain();

    for index in 0..3 {
        let mut mutated = proof.clone();
        let claims = &mut mutated.outer_claims;
        match index {
            0 => claims.0 = claims.0 + QuinticExtension::ONE,
            1 => claims.1 = claims.1 + QuinticExtension::ONE,
            _ => claims.2 = claims.2 + QuinticExtension::ONE,
        }
        assert!(verify_chain(&shape, &instance, &mutated).is_err());
    }

    let mut mutated = proof.clone();
    mutated.witness_eval = mutated.witness_eval + QuinticExtension::ONE;
    assert!(verify_chain(&shape, &instance, &mutated).is_err());

    let mut bad_instance = instance.clone();
    bad_instance.public_inputs[0] = bad_instance.public_inputs[0] + KoalaBear::ONE;
    assert!(verify_chain(&shape, &bad_instance, &proof).is_err());

    let mut bad_instance = instance.clone();
    bad_instance.witness_commitment[0] = bad_instance.witness_commitment[0] + KoalaBear::ONE;
    assert!(verify_chain(&shape, &bad_instance, &proof).is_err());

    let mut short_rounds = proof.clone();
    short_rounds.outer_sumcheck.rounds.pop();
    assert!(verify_chain(&shape, &instance, &short_rounds).is_err());
    let mut long_rounds = proof.clone();
    let extra = long_rounds.inner_sumcheck.rounds[0];
    long_rounds.inner_sumcheck.rounds.push(extra);
    assert!(verify_chain(&shape, &instance, &long_rounds).is_err());
}

#[test]
fn direct_sparse_rejects_unsatisfied_witness_and_matrix_mutations() {
    let shape = chain_shape();
    let mut bad_witness = chain_witness();
    bad_witness.w[0] = kb(3);
    let mut pcs = DevNullPcs::default();
    let mut transcript = PoseidonTranscript::new();
    let (bad_instance, bad_proof) = prove_direct_sparse(
        &shape,
        CONTEXT,
        &chain_public(),
        &bad_witness,
        &mut pcs,
        &mut transcript,
    )
    .unwrap();
    assert!(verify_chain(&shape, &bad_instance, &bad_proof).is_err());

    let mut bad_shape = chain_shape();
    bad_shape.c.entries[0].val = kb(5);
    let mut pcs = DevNullPcs::default();
    let mut transcript = PoseidonTranscript::new();
    let (bad_instance, bad_proof) = prove_direct_sparse(
        &bad_shape,
        CONTEXT,
        &chain_public(),
        &chain_witness(),
        &mut pcs,
        &mut transcript,
    )
    .unwrap();
    assert!(verify_chain(&bad_shape, &bad_instance, &bad_proof).is_err());

    let mut long_public = chain_public();
    long_public.push(kb(7));
    let mut pcs = DevNullPcs::default();
    let mut transcript = PoseidonTranscript::new();
    assert!(
        prove_direct_sparse(
            &shape,
            CONTEXT,
            &long_public,
            &chain_witness(),
            &mut pcs,
            &mut transcript,
        )
        .is_err()
    );

    let mut long_witness = chain_witness();
    long_witness.w.push(kb(9));
    let mut pcs = DevNullPcs::default();
    let mut transcript = PoseidonTranscript::new();
    assert!(
        prove_direct_sparse(
            &shape,
            CONTEXT,
            &chain_public(),
            &long_witness,
            &mut pcs,
            &mut transcript,
        )
        .is_err()
    );
}

#[cfg(feature = "oracle-tests")]
mod oracle {
    use super::*;
    use cirrus_spartan_whir::{
        EqPolynomial, InnerSumcheckProof, MultilinearPoint, OuterSumcheckProof, observe_context,
        prove_inner, prove_outer, verify_inner, verify_outer,
    };
    use p3_challenger::{CanObserve, FieldChallenger};
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32};
    use spartan_whir::engine::{
        F as OracleF, PoseidonChallenger, QuinticExtension as OracleEF, poseidon_challenger,
    };

    fn oracle_kb(value: KoalaBear) -> OracleF {
        OracleF::from_u32(value.canonical())
    }

    fn oracle_ext(value: QuinticExtension) -> OracleEF {
        let coefficients = value.canonical_coefficients();
        OracleEF::from_basis_coefficients_fn(|index| OracleF::from_u32(coefficients[index]))
    }

    fn assert_ext_eq(ours: QuinticExtension, oracle: OracleEF) {
        let ours = ours.canonical_coefficients();
        let oracle = <OracleEF as BasedVectorSpace<OracleF>>::as_basis_coefficients_slice(&oracle);
        for index in 0..5 {
            assert_eq!(ours[index], oracle[index].as_canonical_u32());
        }
    }

    fn convert_matrix(matrix: &SparseMatrix<KoalaBear>) -> spartan_whir::SparseMatrix<OracleF> {
        spartan_whir::SparseMatrix {
            num_rows: matrix.num_rows,
            num_cols: matrix.num_cols,
            entries: matrix
                .entries
                .iter()
                .map(|entry| spartan_whir::SparseMatEntry {
                    row: entry.row,
                    col: entry.col,
                    val: oracle_kb(entry.val),
                })
                .collect(),
        }
    }

    fn oracle_shape(shape: &R1csShape<KoalaBear>) -> spartan_whir::R1csShape<OracleF> {
        spartan_whir::R1csShape {
            num_cons: shape.num_cons,
            num_vars: shape.num_vars,
            num_io: shape.num_io,
            a: convert_matrix(&shape.a),
            b: convert_matrix(&shape.b),
            c: convert_matrix(&shape.c),
        }
    }

    fn oracle_outer_proof(
        proof: &OuterSumcheckProof,
    ) -> spartan_whir::OuterSumcheckProof<OracleEF> {
        spartan_whir::OuterSumcheckProof {
            rounds: proof
                .rounds
                .iter()
                .map(|round| spartan_whir::CubicRoundPoly(round.0.map(oracle_ext)))
                .collect(),
        }
    }

    fn oracle_inner_proof(
        proof: &InnerSumcheckProof,
    ) -> spartan_whir::InnerSumcheckProof<OracleEF> {
        spartan_whir::InnerSumcheckProof {
            rounds: proof
                .rounds
                .iter()
                .map(|round| spartan_whir::QuadraticRoundPoly(round.0.map(oracle_ext)))
                .collect(),
        }
    }

    fn observe_context_oracle(
        challenger: &mut PoseidonChallenger,
        context: &[u8],
        public: &[KoalaBear],
    ) {
        for &byte in context {
            challenger.observe(OracleF::from_u8(byte));
        }
        for &value in public {
            challenger.observe(oracle_kb(value));
        }
    }

    struct OuterOracleRun {
        my_proof: OuterSumcheckProof,
        my_r_x: Vec<QuinticExtension>,
        my_claims: (QuinticExtension, QuinticExtension, QuinticExtension),
        oracle_proof: spartan_whir::OuterSumcheckProof<OracleEF>,
        oracle_r_x: Vec<OracleEF>,
        oracle_claims: (OracleEF, OracleEF, OracleEF),
        my_transcript: PoseidonTranscript,
        oracle_challenger: PoseidonChallenger,
    }

    fn run_outer_oracle() -> OuterOracleRun {
        let shape = chain_shape();
        let oshape = oracle_shape(&shape);
        let public = chain_public();
        let witness_padded = shape.witness_to_mle(&chain_witness().w).unwrap();
        let z_full = build_z_full(witness_padded, shape.num_vars, &public);
        let z_short = matrix_z_slice(&z_full, shape.num_vars, public.len()).unwrap();
        let (az, bz, cz) = shape.multiply_vec_unchecked(z_short).unwrap();
        let oracle_z: Vec<OracleF> = z_short.iter().map(|&v| oracle_kb(v)).collect();
        let (oaz, obz, ocz) = oshape.multiply_vec(&oracle_z).unwrap();

        let context = b"cirrus-sumcheck-oracle-context";
        let mut my_transcript = PoseidonTranscript::new();
        observe_context(&mut my_transcript, context, &public);
        let mut oracle_challenger = poseidon_challenger();
        observe_context_oracle(&mut oracle_challenger, context, &public);

        let num_rounds_x = shape.num_cons.ilog2() as usize;
        let my_tau = MultilinearPoint(
            (0..num_rounds_x)
                .map(|_| my_transcript.sample_quintic())
                .collect::<Vec<_>>(),
        );
        let oracle_tau = spartan_whir::MultilinearPoint(
            (0..num_rounds_x)
                .map(|_| oracle_challenger.sample_algebra_element::<OracleEF>())
                .collect::<Vec<_>>(),
        );
        for (ours, oracle) in my_tau.0.iter().zip(&oracle_tau.0) {
            assert_ext_eq(*ours, *oracle);
        }

        let (my_proof, my_r_x, my_claims) =
            prove_outer(&az, &bz, &cz, &my_tau, &mut my_transcript).unwrap();
        let (oracle_proof, oracle_r_x, oracle_claims) =
            spartan_whir::prove_outer_split_eq_base_first_owned(
                &oshape,
                oaz,
                obz,
                ocz,
                &oracle_tau,
                &mut oracle_challenger,
            )
            .unwrap();

        OuterOracleRun {
            my_proof,
            my_r_x: my_r_x.0,
            my_claims,
            oracle_proof,
            oracle_r_x: oracle_r_x.0,
            oracle_claims,
            my_transcript,
            oracle_challenger,
        }
    }

    #[test]
    fn outer_sumcheck_matches_upstream_oracle() {
        let run = run_outer_oracle();
        assert_eq!(run.my_proof.rounds.len(), run.oracle_proof.rounds.len());
        for (round, (ours, oracle)) in run
            .my_proof
            .rounds
            .iter()
            .zip(&run.oracle_proof.rounds)
            .enumerate()
        {
            for slot in 0..3 {
                assert_ext_eq(ours.0[slot], oracle.0[slot]);
            }
            let _ = round;
        }
        for (ours, oracle) in run.my_r_x.iter().zip(&run.oracle_r_x) {
            assert_ext_eq(*ours, *oracle);
        }
        assert_ext_eq(run.my_claims.0, run.oracle_claims.0);
        assert_ext_eq(run.my_claims.1, run.oracle_claims.1);
        assert_ext_eq(run.my_claims.2, run.oracle_claims.2);
    }

    #[test]
    fn outer_verifier_replay_matches_upstream_accept_and_reject() {
        let run = run_outer_oracle();

        // Cross-check: our proof through the upstream verifier, and the
        // upstream proof through our verifier, from identical schedules.
        let mut oracle_challenger = poseidon_challenger();
        observe_context_oracle(
            &mut oracle_challenger,
            b"cirrus-sumcheck-oracle-context",
            &chain_public(),
        );
        for _ in 0..2 {
            let _ = oracle_challenger.sample_algebra_element::<OracleEF>();
        }
        let (oracle_point, oracle_claim) = spartan_whir::verify_outer(
            &oracle_outer_proof(&run.my_proof),
            OracleEF::ZERO,
            2,
            &mut oracle_challenger,
        )
        .unwrap();

        let mut my_transcript = PoseidonTranscript::new();
        observe_context(
            &mut my_transcript,
            b"cirrus-sumcheck-oracle-context",
            &chain_public(),
        );
        for _ in 0..2 {
            let _ = my_transcript.sample_quintic();
        }
        let (my_point, my_claim) =
            verify_outer(&run.my_proof, QuinticExtension::ZERO, 2, &mut my_transcript).unwrap();

        for (ours, oracle) in my_point.0.iter().zip(&oracle_point.0) {
            assert_ext_eq(*ours, *oracle);
        }
        assert_ext_eq(my_claim, oracle_claim);

        // The upstream proof verifies under our verifier as well.
        let mut my_transcript = PoseidonTranscript::new();
        observe_context(
            &mut my_transcript,
            b"cirrus-sumcheck-oracle-context",
            &chain_public(),
        );
        for _ in 0..2 {
            let _ = my_transcript.sample_quintic();
        }
        let (point, claim) = verify_outer(
            &run.oracle_proof_to_mine(),
            QuinticExtension::ZERO,
            2,
            &mut my_transcript,
        )
        .unwrap();
        for (ours, oracle) in point.0.iter().zip(&run.oracle_r_x) {
            assert_ext_eq(*ours, *oracle);
        }
        assert_eq!(
            claim,
            eq_point_eval(&sample_oracle_tau_mine(), &run.oracle_r_x_mine())
                * (run.my_claims.0 * run.my_claims.1 - run.my_claims.2),
        );

        // Mutated round polys still replay to *some* claim on both
        // verifiers; both implementations must reach the identical replayed
        // point and claim (the final outer equation then rejects them).
        for round in 0..run.my_proof.rounds.len() {
            for slot in 0..3 {
                let mut mutated = run.my_proof.clone();
                mutated.rounds[round].0[slot] =
                    mutated.rounds[round].0[slot] + QuinticExtension::ONE;

                let mut my_transcript = PoseidonTranscript::new();
                observe_context(
                    &mut my_transcript,
                    b"cirrus-sumcheck-oracle-context",
                    &chain_public(),
                );
                for _ in 0..2 {
                    let _ = my_transcript.sample_quintic();
                }
                let (my_point, my_claim) =
                    verify_outer(&mutated, QuinticExtension::ZERO, 2, &mut my_transcript)
                        .expect("well-formed mutated rounds replay");

                let mut oracle_challenger = poseidon_challenger();
                observe_context_oracle(
                    &mut oracle_challenger,
                    b"cirrus-sumcheck-oracle-context",
                    &chain_public(),
                );
                for _ in 0..2 {
                    let _ = oracle_challenger.sample_algebra_element::<OracleEF>();
                }
                let (oracle_point, oracle_claim) = spartan_whir::verify_outer(
                    &oracle_outer_proof(&mutated),
                    OracleEF::ZERO,
                    2,
                    &mut oracle_challenger,
                )
                .expect("well-formed mutated rounds replay");

                for (ours, oracle) in my_point.0.iter().zip(&oracle_point.0) {
                    assert_ext_eq(*ours, *oracle);
                }
                assert_ext_eq(my_claim, oracle_claim);
                assert_ne!(
                    my_claim,
                    eq_point_eval(&sample_oracle_tau_mine(), &my_point.0)
                        * (run.my_claims.0 * run.my_claims.1 - run.my_claims.2),
                    "mutation did not change the replayed claim"
                );
            }
        }
    }

    impl OuterOracleRun {
        fn oracle_proof_to_mine(&self) -> OuterSumcheckProof {
            OuterSumcheckProof {
                rounds: self
                    .oracle_proof
                    .rounds
                    .iter()
                    .map(|round| {
                        cirrus_spartan_whir::CubicRoundPoly(round.0.map(|value| {
                            QuinticExtension::from_canonical_coefficients(
                                core::array::from_fn(|index| {
                                    <OracleEF as BasedVectorSpace<OracleF>>::as_basis_coefficients_slice(
                                        &value,
                                    )[index]
                                        .as_canonical_u32()
                                }),
                            )
                            .unwrap()
                        }))
                    })
                    .collect(),
            }
        }

        fn oracle_r_x_mine(&self) -> Vec<QuinticExtension> {
            self.oracle_r_x
                .iter()
                .map(|value| {
                    QuinticExtension::from_canonical_coefficients(core::array::from_fn(|index| {
                        <OracleEF as BasedVectorSpace<OracleF>>::as_basis_coefficients_slice(value)
                            [index]
                            .as_canonical_u32()
                    }))
                    .unwrap()
                })
                .collect()
        }
    }

    fn sample_oracle_tau_mine() -> Vec<QuinticExtension> {
        let mut transcript = PoseidonTranscript::new();
        observe_context(
            &mut transcript,
            b"cirrus-sumcheck-oracle-context",
            &chain_public(),
        );
        (0..2).map(|_| transcript.sample_quintic()).collect()
    }

    /// Prover-side schedule run through both sumchecks, retaining everything
    /// a verifier needs to replay the inner sumcheck.
    struct InnerOracleRun {
        my_outer: OuterSumcheckProof,
        my_inner: InnerSumcheckProof,
        my_claims: (QuinticExtension, QuinticExtension, QuinticExtension),
        my_r_x: Vec<QuinticExtension>,
        my_r_y: Vec<QuinticExtension>,
        my_eval_z: QuinticExtension,
        my_r: QuinticExtension,
        my_joint: QuinticExtension,
        my_poly_abc: Vec<QuinticExtension>,
        oracle_outer: spartan_whir::OuterSumcheckProof<OracleEF>,
        oracle_inner: spartan_whir::InnerSumcheckProof<OracleEF>,
        oracle_claims: (OracleEF, OracleEF, OracleEF),
        oracle_r_x: Vec<OracleEF>,
        oracle_r_y: Vec<OracleEF>,
        oracle_eval_z: OracleEF,
        oracle_r: OracleEF,
        oracle_joint: OracleEF,
        oracle_poly_abc: Vec<OracleEF>,
    }

    fn run_inner_oracle() -> InnerOracleRun {
        let shape = chain_shape();
        let oshape = oracle_shape(&shape);
        let mut run = run_outer_oracle();

        run.my_transcript.observe_quintic_slice(&[
            run.my_claims.0,
            run.my_claims.1,
            run.my_claims.2,
        ]);
        let my_r = run.my_transcript.sample_quintic();
        run.oracle_challenger.observe_algebra_slice(&[
            run.oracle_claims.0,
            run.oracle_claims.1,
            run.oracle_claims.2,
        ]);
        let oracle_r = run.oracle_challenger.sample_algebra_element::<OracleEF>();

        let my_joint = run.my_claims.0 + my_r * run.my_claims.1 + my_r * my_r * run.my_claims.2;
        let oracle_joint = run.oracle_claims.0
            + oracle_r * run.oracle_claims.1
            + oracle_r * oracle_r * run.oracle_claims.2;

        let my_t_x = EqPolynomial::evals_from_point(&run.my_r_x);
        let oracle_t_x = spartan_whir::EqPolynomial::evals_from_point(&run.oracle_r_x);
        let my_poly_abc = bind_row_vars_joint(&shape, &my_t_x, my_r).unwrap();
        let oracle_poly_abc = oshape
            .bind_row_vars_joint::<OracleEF>(&oracle_t_x, oracle_r)
            .unwrap();

        let witness_padded = shape.witness_to_mle(&chain_witness().w).unwrap();
        let z_full = build_z_full(witness_padded, shape.num_vars, &chain_public());
        let oracle_z_full: Vec<OracleF> = z_full.iter().map(|&v| oracle_kb(v)).collect();
        let (my_inner, my_r_y, my_eval_z) =
            prove_inner(my_joint, &my_poly_abc, &z_full, &mut run.my_transcript).unwrap();
        let (oracle_inner, oracle_r_y, oracle_eval_z) = spartan_whir::prove_inner_base_first(
            &oshape,
            oracle_joint,
            &oracle_poly_abc,
            &oracle_z_full,
            &mut run.oracle_challenger,
        )
        .unwrap();

        InnerOracleRun {
            my_outer: run.my_proof,
            my_inner,
            my_claims: run.my_claims,
            my_r_x: run.my_r_x,
            my_r_y: my_r_y.0,
            my_eval_z,
            my_r,
            my_joint,
            my_poly_abc,
            oracle_outer: run.oracle_proof,
            oracle_inner,
            oracle_claims: run.oracle_claims,
            oracle_r_x: run.oracle_r_x,
            oracle_r_y: oracle_r_y.0,
            oracle_eval_z,
            oracle_r,
            oracle_joint,
            oracle_poly_abc,
        }
    }

    /// Replay the verifier-side schedule up to the inner-sumcheck boundary
    /// under our implementation, returning the joint claim.
    fn verifier_joint_claim_mine(
        outer_proof: &OuterSumcheckProof,
        claims: &(QuinticExtension, QuinticExtension, QuinticExtension),
        transcript: &mut PoseidonTranscript,
    ) -> QuinticExtension {
        observe_context(
            transcript,
            b"cirrus-sumcheck-oracle-context",
            &chain_public(),
        );
        let tau: Vec<QuinticExtension> = (0..2).map(|_| transcript.sample_quintic()).collect();
        let (r_x, outer_claim) =
            verify_outer(outer_proof, QuinticExtension::ZERO, 2, transcript).unwrap();
        assert_eq!(
            outer_claim,
            eq_point_eval(&tau, &r_x.0) * (claims.0 * claims.1 - claims.2)
        );
        transcript.observe_quintic_slice(&[claims.0, claims.1, claims.2]);
        let r = transcript.sample_quintic();
        claims.0 + r * claims.1 + r * r * claims.2
    }

    /// The same schedule under the upstream implementation.
    fn verifier_joint_claim_oracle(
        outer_proof: &spartan_whir::OuterSumcheckProof<OracleEF>,
        claims: &(OracleEF, OracleEF, OracleEF),
        challenger: &mut PoseidonChallenger,
    ) -> OracleEF {
        observe_context_oracle(
            challenger,
            b"cirrus-sumcheck-oracle-context",
            &chain_public(),
        );
        for _ in 0..2 {
            let _ = challenger.sample_algebra_element::<OracleEF>();
        }
        let (_r_x, _outer_claim) =
            spartan_whir::verify_outer(outer_proof, OracleEF::ZERO, 2, challenger).unwrap();
        challenger.observe_algebra_slice(&[claims.0, claims.1, claims.2]);
        let r = challenger.sample_algebra_element::<OracleEF>();
        claims.0 + r * claims.1 + r * r * claims.2
    }

    #[test]
    fn inner_verifier_replay_matches_upstream_accept_and_reject() {
        let run = run_inner_oracle();

        // Accept path: our inner proof through the upstream verifier, and the
        // upstream inner proof through our verifier.
        let mut my_transcript = PoseidonTranscript::new();
        let my_joint = verifier_joint_claim_mine(&run.my_outer, &run.my_claims, &mut my_transcript);
        let (my_r_y, my_inner_claim) =
            verify_inner(&run.my_inner, my_joint, 3, &mut my_transcript).unwrap();
        for (ours, oracle) in my_r_y.0.iter().zip(&run.oracle_r_y) {
            assert_ext_eq(*ours, *oracle);
        }

        let mut oracle_challenger = poseidon_challenger();
        let oracle_joint = verifier_joint_claim_oracle(
            &run.oracle_outer,
            &run.oracle_claims,
            &mut oracle_challenger,
        );
        assert_ext_eq(my_joint, oracle_joint);
        let (upstream_point, upstream_claim) = spartan_whir::verify_inner(
            &oracle_inner_proof(&run.my_inner),
            oracle_joint,
            3,
            &mut oracle_challenger,
        )
        .unwrap();
        for (ours, oracle) in upstream_point.0.iter().zip(&my_r_y.0) {
            assert_ext_eq(*oracle, *ours);
        }
        assert_ext_eq(my_inner_claim, upstream_claim);

        // Mutated inner rounds replay identically on both verifiers.
        for round in 0..run.my_inner.rounds.len() {
            for slot in 0..2 {
                let mut mutated = run.my_inner.clone();
                mutated.rounds[round].0[slot] =
                    mutated.rounds[round].0[slot] + QuinticExtension::ONE;

                let mut my_transcript = PoseidonTranscript::new();
                let my_joint =
                    verifier_joint_claim_mine(&run.my_outer, &run.my_claims, &mut my_transcript);
                let (my_point, my_claim) = verify_inner(&mutated, my_joint, 3, &mut my_transcript)
                    .expect("well-formed mutated rounds replay");

                let mut oracle_challenger = poseidon_challenger();
                let oracle_joint = verifier_joint_claim_oracle(
                    &run.oracle_outer,
                    &run.oracle_claims,
                    &mut oracle_challenger,
                );
                let (oracle_point, oracle_claim) = spartan_whir::verify_inner(
                    &oracle_inner_proof(&mutated),
                    oracle_joint,
                    3,
                    &mut oracle_challenger,
                )
                .expect("well-formed mutated rounds replay");

                for (ours, oracle) in my_point.0.iter().zip(&oracle_point.0) {
                    assert_ext_eq(*ours, *oracle);
                }
                assert_ext_eq(my_claim, oracle_claim);
                assert_ne!(
                    my_claim, my_inner_claim,
                    "mutation did not change the replayed claim"
                );
            }
        }
    }

    #[test]
    fn inner_sumcheck_and_reduction_match_upstream_oracle() {
        let shape = chain_shape();
        let oshape = oracle_shape(&shape);
        let run = run_inner_oracle();

        // Joint challenge, joint claim, and poly_abc parity.
        assert_ext_eq(run.my_r, run.oracle_r);
        assert_ext_eq(run.my_joint, run.oracle_joint);
        assert_eq!(run.my_poly_abc.len(), run.oracle_poly_abc.len());
        for (ours, oracle) in run.my_poly_abc.iter().zip(&run.oracle_poly_abc) {
            assert_ext_eq(*ours, *oracle);
        }

        // Inner sumcheck parity.
        assert_eq!(run.my_inner.rounds.len(), run.oracle_inner.rounds.len());
        for (ours, oracle) in run.my_inner.rounds.iter().zip(&run.oracle_inner.rounds) {
            for slot in 0..2 {
                assert_ext_eq(ours.0[slot], oracle.0[slot]);
            }
        }
        for (ours, oracle) in run.my_r_y.iter().zip(&run.oracle_r_y) {
            assert_ext_eq(*ours, *oracle);
        }
        assert_ext_eq(run.my_eval_z, run.oracle_eval_z);

        // Matrix evaluations at (r_x, r_y).
        let my_t_x = EqPolynomial::evals_from_point(&run.my_r_x);
        let my_t_y = EqPolynomial::evals_from_point(&run.my_r_y);
        let oracle_t_x = spartan_whir::EqPolynomial::evals_from_point(&run.oracle_r_x);
        let oracle_t_y = spartan_whir::EqPolynomial::evals_from_point(&run.oracle_r_y);
        let (my_a, my_b, my_c) = evaluate_with_tables(&shape, &my_t_x, &my_t_y).unwrap();
        let (oracle_a, oracle_b, oracle_c) = oshape
            .evaluate_with_tables::<OracleEF>(&oracle_t_x, &oracle_t_y)
            .unwrap();
        assert_ext_eq(my_a, oracle_a);
        assert_ext_eq(my_b, oracle_b);
        assert_ext_eq(my_c, oracle_c);

        // Public half and witness evaluation recovery.
        let my_r_y_point = MultilinearPoint(run.my_r_y.clone());
        let my_r_y = my_r_y_point.0;
        let my_eval_z = run.my_eval_z;
        let my_eval_x =
            evaluate_public_half(shape.num_vars, &chain_public(), &my_r_y[1..]).unwrap();
        let dense = vec![
            QuinticExtension::ONE,
            kb(3).into(),
            kb(5).into(),
            QuinticExtension::ZERO,
        ];
        assert_eq!(my_eval_x, evaluate_mle_table(&dense, &my_r_y[1..]).unwrap());
        let my_witness_eval = recover_witness_eval(my_r_y[0], my_eval_z, my_eval_x).unwrap();
        // The recovered witness evaluation equals the dense witness MLE.
        let dense_witness: Vec<QuinticExtension> =
            chain_witness().w.iter().map(|&v| v.into()).collect();
        assert_eq!(
            my_witness_eval,
            evaluate_mle_table(&dense_witness, &my_r_y[1..]).unwrap()
        );

        // Verifier-side inner replay accepts the honest proof.
        let mut my_verifier = PoseidonTranscript::new();
        observe_context(
            &mut my_verifier,
            b"cirrus-sumcheck-oracle-context",
            &chain_public(),
        );
        for _ in 0..2 {
            let _ = my_verifier.sample_quintic();
        }
        let (point, outer_claim) =
            verify_outer(&run.my_outer, QuinticExtension::ZERO, 2, &mut my_verifier).unwrap();
        let tau = sample_oracle_tau_mine();
        assert_eq!(
            outer_claim,
            eq_point_eval(&tau, &point.0) * (run.my_claims.0 * run.my_claims.1 - run.my_claims.2)
        );
        my_verifier.observe_quintic_slice(&[run.my_claims.0, run.my_claims.1, run.my_claims.2]);
        let r = my_verifier.sample_quintic();
        let joint = run.my_claims.0 + r * run.my_claims.1 + r * r * run.my_claims.2;
        let (r_y, inner_claim) = verify_inner(&run.my_inner, joint, 3, &mut my_verifier).unwrap();
        let eval_z = (QuinticExtension::ONE - r_y.0[0]) * my_witness_eval + r_y.0[0] * my_eval_x;
        let t_x = EqPolynomial::evals_from_point(&point.0);
        let t_y = EqPolynomial::evals_from_point(&r_y.0);
        let (eval_a, eval_b, eval_c) = evaluate_with_tables(&shape, &t_x, &t_y).unwrap();
        assert_eq!(inner_claim, (eval_a + r * eval_b + r * r * eval_c) * eval_z);
    }
}
