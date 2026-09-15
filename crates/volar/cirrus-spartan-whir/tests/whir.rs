//! Deterministic tests for the no-ZK WHIR PCS and its DirectSparse adapter.

use cirrus_spartan_whir::whir::domain_separator::WhirDomainSeparator;
use cirrus_spartan_whir::whir::params::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig,
};
use cirrus_spartan_whir::whir::pcs::{
    QueryOpenings, SharedProofOpening, WhirError, WhirProof, get_challenge_stir_queries,
};
use cirrus_spartan_whir::whir::spartan::{SecurityProfile, WhirPcs};
use cirrus_spartan_whir::{
    DirectSparsePcs, KoalaBear, PoseidonDigest, PoseidonMerkleTree, PoseidonTranscript,
    PrunedMerklePaths, QuinticExtension, R1csShape, R1csWitness, SparseMatEntry, SparseMatrix,
    dft_base, dft_batch_base, dft_batch_ext, poseidon_hash_fixed, prove_direct_sparse,
    two_adic_generator, verify_direct_sparse,
};

fn kb(value: u32) -> KoalaBear {
    KoalaBear::from_u64(u64::from(value))
}

fn witness(num_variables: usize) -> Vec<KoalaBear> {
    (0..1usize << num_variables)
        .map(|i| {
            kb((i as u32)
                .wrapping_mul(40503)
                .wrapping_add(0x5bd1)
                .wrapping_add((i as u32) << 16))
        })
        .collect()
}

fn test_profile() -> SecurityProfile {
    SecurityProfile::capacity_bound_80_test()
}

fn pcs(num_variables: usize) -> WhirPcs {
    WhirPcs::new(num_variables, &test_profile()).expect("test profile derives a config")
}

#[test]
fn two_adic_generator_has_the_expected_orders_and_frozen_values() {
    // Frozen values from the pinned upstream `TWO_ADIC_GENERATORS` table.
    let frozen = [
        0x1u32, 0x7f000000, 0x7e010002, 0x6832fe4a, 0x8dbd69c, 0xa28f031, 0x5c4a5b99, 0x29b75a80,
        0x17668b8a, 0x27ad539b,
    ];
    for (bits, &expected) in frozen.iter().enumerate() {
        assert_eq!(
            two_adic_generator(bits).canonical(),
            expected,
            "two-adic generator mismatch at {bits} bits"
        );
    }
    for bits in 1..=24usize {
        let g = two_adic_generator(bits);
        assert_eq!(
            g.pow(1u64 << bits),
            KoalaBear::ONE,
            "order exceeds 2^{bits}"
        );
        assert_ne!(
            g.pow(1u64 << (bits - 1)),
            KoalaBear::ONE,
            "order below 2^{bits}"
        );
    }
}

#[test]
fn dft_matches_naive_subgroup_evaluation() {
    let log_n = 4usize;
    let n = 1usize << log_n;
    let g = two_adic_generator(log_n);
    let coeffs: Vec<KoalaBear> = (0..n).map(|i| kb(7 + 3 * i as u32)).collect();
    let evals = dft_base(&coeffs).unwrap();
    for (j, &got) in evals.iter().enumerate() {
        let mut expected = KoalaBear::ZERO;
        let mut power = KoalaBear::ONE;
        let point = g.pow(j as u64);
        for &c in &coeffs {
            expected = expected + c * power;
            power = power * point;
        }
        assert_eq!(got, expected, "DFT mismatch at index {j}");
    }
}

#[test]
fn batch_dft_matches_column_wise_single_dfts() {
    let (height, width) = (8usize, 4usize);
    let mut matrix: Vec<KoalaBear> = (0..height * width)
        .map(|i| kb((i as u32).wrapping_mul(97) + 13))
        .collect();
    dft_batch_base(&mut matrix, height, width).unwrap();
    for c in 0..width {
        let column: Vec<KoalaBear> = (0..height)
            .map(|r| kb((r * width + c) as u32 * 97 + 13))
            .collect();
        let expected = dft_base(&column).unwrap();
        for r in 0..height {
            assert_eq!(matrix[r * width + c], expected[r]);
        }
    }
}

#[test]
fn extension_batch_dft_matches_base_batch_dft() {
    let (height, width) = (8usize, 2usize);
    let mut ext: Vec<QuinticExtension> = (0..height * width)
        .map(|i| QuinticExtension::new(core::array::from_fn(|j| kb((i * 5 + j) as u32 * 31 + 7))))
        .collect();
    dft_batch_ext(&mut ext, height, width).unwrap();
    for coordinate in 0..5 {
        let mut base: Vec<KoalaBear> = (0..height * width)
            .map(|i| kb((i * 5 + coordinate) as u32 * 31 + 7))
            .collect();
        dft_batch_base(&mut base, height, width).unwrap();
        for (i, element) in ext.iter().enumerate() {
            assert_eq!(
                element.canonical_coefficients()[coordinate],
                base[i].canonical()
            );
        }
    }
}

#[test]
fn merkle_multi_opening_roundtrips_and_rejects_mutations() {
    let rows: Vec<Vec<KoalaBear>> = (0..16)
        .map(|r| (0..4).map(|c| kb(r * 11 + c + 1)).collect())
        .collect();
    let tree = PoseidonMerkleTree::commit_rows(&rows).unwrap();
    let root = tree.root();

    let indices = [1usize, 2, 5, 8, 13];
    let (opened, proof) = tree.open_multi(&indices, &rows).unwrap();
    assert_eq!(opened.len(), indices.len());
    PoseidonMerkleTree::verify_multi(&root, 16, 4, &indices, &opened, &proof).unwrap();

    // A digest fewer than the frontier requires is rejected.
    let mut short = proof.clone();
    short.sibling_hashes.pop();
    assert!(PoseidonMerkleTree::verify_multi(&root, 16, 4, &indices, &opened, &short).is_err());

    // A trailing extra digest is rejected.
    let mut long = proof.clone();
    long.sibling_hashes.push(PoseidonDigest::default());
    assert!(PoseidonMerkleTree::verify_multi(&root, 16, 4, &indices, &opened, &long).is_err());

    // A tampered row is rejected.
    let mut bad_rows = opened.clone();
    bad_rows[0][0] = bad_rows[0][0] + KoalaBear::ONE;
    assert!(PoseidonMerkleTree::verify_multi(&root, 16, 4, &indices, &bad_rows, &proof).is_err());

    // A tampered digest is rejected.
    let mut bad_proof = proof.clone();
    bad_proof.sibling_hashes[0][0] = bad_proof.sibling_hashes[0][0] + KoalaBear::ONE;
    assert!(PoseidonMerkleTree::verify_multi(&root, 16, 4, &indices, &opened, &bad_proof).is_err());

    // A wrong root is rejected.
    let other_root = poseidon_hash_fixed(&[kb(42)]);
    assert!(
        PoseidonMerkleTree::verify_multi(&other_root, 16, 4, &indices, &opened, &proof).is_err()
    );

    // An out-of-range index is rejected.
    assert!(tree.open_multi(&[16], &rows).is_err());
    assert!(
        PoseidonMerkleTree::verify_multi(&root, 16, 4, &[16], &[vec![kb(1); 4]], &proof).is_err()
    );

    // Duplicate queries with disagreeing rows are rejected.
    let dup_indices = [3usize, 3];
    let (dup_opened, dup_proof) = tree.open_multi(&dup_indices, &rows).unwrap();
    PoseidonMerkleTree::verify_multi(&root, 16, 4, &dup_indices, &dup_opened, &dup_proof).unwrap();
    let mut disagreeing = dup_opened.clone();
    disagreeing[1][0] = disagreeing[1][0] + KoalaBear::ONE;
    assert!(
        PoseidonMerkleTree::verify_multi(&root, 16, 4, &dup_indices, &disagreeing, &dup_proof)
            .is_err()
    );
}

#[test]
fn config_derivation_for_the_test_profile_is_stable() {
    let config = test_profile().derive_config(8).unwrap();
    assert_eq!(config.num_variables, 8);
    assert_eq!(config.params.starting_log_inv_rate, 6);
    assert_eq!(config.folding_schedule, vec![1, 1]);
    assert_eq!(config.n_rounds(), 1);
    assert_eq!(config.final_sumcheck_rounds, 6);
    assert_eq!(config.commitment_ood_samples, 1);
    // The 80-bit test profile derives no grinding on the 155-bit field.
    assert_eq!(config.starting_folding_pow_bits, 0);
    assert_eq!(config.final_pow_bits, 0);
    assert_eq!(config.final_folding_pow_bits, 0);
    for round in &config.round_parameters {
        assert_eq!(round.pow_bits, 0);
        assert_eq!(round.folding_pow_bits, 0);
        assert!(round.num_queries > 0);
        assert_eq!(round.ood_samples, 1);
    }
    assert!(config.final_queries > 0);

    // Domain geometry: 2^(8+6) = 16384 initial, fold 1 per round.
    assert_eq!(config.starting_domain_size(), 16384);
    assert_eq!(config.round_parameters[0].domain_size, 16384);
    assert_eq!(config.round_parameters[0].log_inv_rate, 6);
    assert_eq!(config.round_parameters[0].num_variables, 7);
    assert_eq!(config.final_round_config().domain_size, 8192);
    assert_eq!(config.final_round_config().num_variables, 6);
}

#[test]
fn config_derivation_rejects_invalid_profiles() {
    use cirrus_spartan_whir::whir::params::WhirConfigError;
    use cirrus_spartan_whir::whir::spartan::WhirConfigBuildError;

    // A non-redundant starting rate is rejected.
    let error = WhirConfig::new(
        4,
        ProtocolParameters {
            starting_log_inv_rate: 0,
            round_log_inv_rates: vec![],
            folding_factor: FoldingFactor::Constant(1),
            soundness_type: SecurityAssumption::CapacityBound,
            security_level: 80,
            pow_bits: 16,
        },
    )
    .unwrap_err();
    assert_eq!(
        error,
        WhirConfigError::NonRedundantStartingRate { log_inv_rate: 0 }
    );

    // Security below the 80-bit floor is rejected.
    let mut weak = test_profile();
    weak.security_level_bits = 79;
    assert_eq!(
        weak.derive_config(4).unwrap_err(),
        WhirConfigBuildError::SecurityBelowMinimum
    );

    // A zero initial reduction factor is rejected.
    let mut zero_reduction = test_profile();
    zero_reduction.whir.rs_domain_initial_reduction_factor = 0;
    assert_eq!(
        zero_reduction.derive_config(4).unwrap_err(),
        WhirConfigBuildError::ZeroRsDomainInitialReductionFactor
    );

    // A zero folding factor is rejected.
    let mut zero_folding = test_profile();
    zero_folding.whir.folding_factor = 0;
    assert_eq!(
        zero_folding.derive_config(4).unwrap_err(),
        WhirConfigBuildError::ZeroFoldingFactor
    );
}

#[test]
fn domain_separator_pattern_matches_hand_built_encoding() {
    let config = test_profile().derive_config(8).unwrap();
    let ds = WhirDomainSeparator::new(&config);

    let observe = |label: u32, count: u32| kb((1u32 << 24) | (label << 16) | count);
    let sample = |label: u32, count: u32| kb((label << 16) | count);
    let hint = |label: u32| kb((2u32 << 24) | (label << 16));
    let mut expected: Vec<KoalaBear> = Vec::new();
    let mut push_param = |value: u32| {
        expected.push(observe(5, 0)); // ProtocolParam marker
        expected.push(KoalaBear::from_u64(u64::from(value)));
    };
    // Configuration binding.
    push_param(8); // num_variables
    push_param(80); // security_level
    push_param(6); // starting_log_inv_rate
    push_param(0); // pow_bits
    push_param(1); // round count
    push_param(6); // round 0 log_inv_rate
    push_param(2); // CapacityBound discriminant
    push_param(0); // Constant folding encoding
    push_param(1); // factor
    // commit_statement tail.
    expected.push(observe(0, 8)); // MerkleDigest
    assert_eq!(config.commitment_ood_samples, 1);
    expected.push(sample(6, 1)); // OodQuery
    expected.push(observe(1, 1)); // OodAnswers
    // add_whir_proof: initial combination and first sumcheck (1 round).
    expected.push(sample(0, 1)); // InitialCombinationRandomness
    expected.push(observe(2, 2)); // SumcheckPoly
    expected.push(sample(1, 1)); // FoldingRandomness
    // One intermediate round: 16384 -> folded 8192, 2 domain bytes.
    expected.push(observe(0, 8)); // MerkleDigest
    assert_eq!(config.round_parameters[0].ood_samples, 1);
    expected.push(sample(6, 1)); // OodQuery
    expected.push(observe(1, 1)); // OodAnswers
    expected.push(sample(7, 1)); // TranscriptCheckpoint
    expected.push(sample(
        3,
        (config.round_parameters[0].num_queries * 2) as u32,
    )); // StirQueries
    expected.push(hint(0)); // StirQueries
    expected.push(hint(2)); // MerkleProof
    expected.push(sample(2, 1)); // CombinationRandomness
    expected.push(observe(2, 2)); // SumcheckPoly
    expected.push(sample(1, 1)); // FoldingRandomness
    // Final round: 64 final coefficients, final queries over 4096 positions
    // (2 domain bytes), then 6 final sumcheck rounds.
    expected.push(observe(3, 64)); // FinalCoeffs
    expected.push(sample(4, (config.final_queries * 2) as u32)); // FinalQueries
    expected.push(hint(1)); // StirAnswers
    expected.push(hint(2)); // MerkleProof
    for _ in 0..6 {
        expected.push(observe(2, 2)); // SumcheckPoly
        expected.push(sample(1, 1)); // FoldingRandomness
    }
    expected.push(hint(3)); // DeferredWeightEvaluations

    assert_eq!(
        ds.pattern()
            .iter()
            .map(|v| v.canonical())
            .collect::<Vec<_>>(),
        expected.iter().map(|v| v.canonical()).collect::<Vec<_>>()
    );
}

/// Prove and verify one opening with the test profile.
fn prove_opening(
    num_variables: usize,
) -> (
    PoseidonDigest,
    WhirProof,
    Vec<QuinticExtension>,
    QuinticExtension,
) {
    let poly = witness(num_variables);
    let mut prover = pcs(num_variables);
    let mut transcript = PoseidonTranscript::new();
    let commitment = prover.commit(&poly, &mut transcript).unwrap();

    // The opening point is a public input to open/verify; derive it from a
    // separate transcript so the prover/verifier schedules stay in lockstep.
    let mut point_transcript = PoseidonTranscript::new();
    point_transcript.observe(kb(num_variables as u32));
    let point: Vec<QuinticExtension> = (0..num_variables)
        .map(|_| point_transcript.sample_quintic())
        .collect();
    let value = cirrus_spartan_whir::whir::pcs::eval_mle_base_at_ext(&poly, &point);
    let proof = prover.open(&point, value, &mut transcript).unwrap();
    (commitment, proof, point, value)
}

#[test]
fn pcs_commit_open_verify_accepts_honest_proofs() {
    for num_variables in [1usize, 2, 4, 6] {
        let (commitment, proof, point, value) = prove_opening(num_variables);
        let verifier = pcs(num_variables);
        let mut transcript = PoseidonTranscript::new();
        let parsed = verifier
            .verify_commitment(&commitment, &proof, &mut transcript)
            .unwrap();
        verifier
            .verify_opening(&parsed, &point, value, &proof, &mut transcript)
            .unwrap();
    }
}

fn verify_opening(
    num_variables: usize,
    commitment: &PoseidonDigest,
    proof: &WhirProof,
    point: &[QuinticExtension],
    value: QuinticExtension,
) -> Result<(), WhirError> {
    let verifier = pcs(num_variables);
    let mut transcript = PoseidonTranscript::new();
    let parsed = verifier.verify_commitment(commitment, proof, &mut transcript)?;
    verifier.verify_opening(&parsed, point, value, proof, &mut transcript)
}

#[test]
fn pcs_verification_rejects_mutated_proofs_and_claims() {
    let num_variables = 8usize;
    let (commitment, proof, point, value) = prove_opening(num_variables);

    // A wrong claimed value is rejected.
    let wrong_value = value + QuinticExtension::ONE;
    assert!(
        verify_opening(num_variables, &commitment, &proof, &point, wrong_value).is_err(),
        "wrong claimed value accepted"
    );

    // A wrong opening point is rejected.
    let mut wrong_point = point.clone();
    wrong_point[0] = wrong_point[0] + QuinticExtension::ONE;
    assert!(
        verify_opening(num_variables, &commitment, &proof, &wrong_point, value).is_err(),
        "wrong opening point accepted"
    );

    // A tampered initial OOD answer is rejected.
    let mut mutated = proof.clone();
    if !mutated.initial_ood_answers.is_empty() {
        mutated.initial_ood_answers[0] = mutated.initial_ood_answers[0] + QuinticExtension::ONE;
        assert!(
            verify_opening(num_variables, &commitment, &mutated, &point, value).is_err(),
            "tampered initial OOD answer accepted"
        );
    }

    // A tampered round commitment is rejected.
    let mut mutated = proof.clone();
    mutated.rounds[0].commitment = Some(poseidon_hash_fixed(&[kb(7)]));
    assert!(
        verify_opening(num_variables, &commitment, &mutated, &point, value).is_err(),
        "tampered round commitment accepted"
    );

    // A tampered round OOD answer is rejected.
    let mut mutated = proof.clone();
    if !mutated.rounds[0].ood_answers.is_empty() {
        mutated.rounds[0].ood_answers[0] = mutated.rounds[0].ood_answers[0] + QuinticExtension::ONE;
        assert!(
            verify_opening(num_variables, &commitment, &mutated, &point, value).is_err(),
            "tampered round OOD answer accepted"
        );
    }

    // A tampered initial sumcheck round polynomial is rejected.
    let mut mutated = proof.clone();
    mutated.initial_sumcheck.polynomial_evaluations[0][0] =
        mutated.initial_sumcheck.polynomial_evaluations[0][0] + QuinticExtension::ONE;
    assert!(
        verify_opening(num_variables, &commitment, &mutated, &point, value).is_err(),
        "tampered initial sumcheck polynomial accepted"
    );

    // A tampered round sumcheck polynomial is rejected.
    let mut mutated = proof.clone();
    mutated.rounds[0].sumcheck.polynomial_evaluations[0][1] =
        mutated.rounds[0].sumcheck.polynomial_evaluations[0][1] + QuinticExtension::ONE;
    assert!(
        verify_opening(num_variables, &commitment, &mutated, &point, value).is_err(),
        "tampered round sumcheck polynomial accepted"
    );

    // A tampered final polynomial is rejected.
    let mut mutated = proof.clone();
    mutated.final_poly.as_mut().unwrap()[0] =
        mutated.final_poly.as_ref().unwrap()[0] + QuinticExtension::ONE;
    assert!(
        verify_opening(num_variables, &commitment, &mutated, &point, value).is_err(),
        "tampered final polynomial accepted"
    );

    // A missing final polynomial is rejected.
    let mut mutated = proof.clone();
    mutated.final_poly = None;
    assert_eq!(
        verify_opening(num_variables, &commitment, &mutated, &point, value),
        Err(WhirError::MissingFinalPoly)
    );

    // A missing round commitment is rejected.
    let mut mutated = proof.clone();
    mutated.rounds[0].commitment = None;
    assert_eq!(
        verify_opening(num_variables, &commitment, &mutated, &point, value),
        Err(WhirError::MissingRoundCommitment { round: 0 })
    );

    // A wrong proof round count is rejected.
    let mut mutated = proof.clone();
    mutated.rounds.pop();
    assert_eq!(
        verify_opening(num_variables, &commitment, &mutated, &point, value),
        Err(WhirError::RoundCountMismatch {
            expected: 1,
            actual: 0
        })
    );

    // A tampered STIR opening row is rejected.
    let mut mutated = proof.clone();
    match &mut mutated.rounds[0].openings {
        QueryOpenings::Base(opening) => {
            opening.rows[0][0] = opening.rows[0][0] + KoalaBear::ONE;
        }
        QueryOpenings::Extension(opening) => {
            opening.rows[0][0] = opening.rows[0][0] + QuinticExtension::ONE;
        }
    }
    assert!(
        verify_opening(num_variables, &commitment, &mutated, &point, value).is_err(),
        "tampered STIR row accepted"
    );

    // A tampered Merkle multiproof digest is rejected.
    let mut mutated = proof.clone();
    match &mut mutated.rounds[0].openings {
        QueryOpenings::Base(opening) => {
            opening.proof.sibling_hashes[0][0] =
                opening.proof.sibling_hashes[0][0] + KoalaBear::ONE;
        }
        QueryOpenings::Extension(opening) => {
            opening.proof.sibling_hashes[0][0] =
                opening.proof.sibling_hashes[0][0] + KoalaBear::ONE;
        }
    }
    assert!(
        verify_opening(num_variables, &commitment, &mutated, &point, value).is_err(),
        "tampered multiproof digest accepted"
    );

    // A wrong openings variant (Base in place of the final Extension
    // openings) is rejected.
    let mut mutated = proof.clone();
    let replacement = QueryOpenings::Base(SharedProofOpening {
        rows: vec![vec![kb(1); 2]],
        proof: PrunedMerklePaths::default(),
    });
    core::mem::swap(&mut mutated.final_openings, &mut { replacement });
    assert!(
        verify_opening(num_variables, &commitment, &mutated, &point, value).is_err(),
        "wrong openings variant accepted"
    );

    // A final-sumcheck mutation is rejected.
    let mut mutated = proof.clone();
    if let Some(final_sumcheck) = &mut mutated.final_sumcheck {
        final_sumcheck.polynomial_evaluations[0][0] =
            final_sumcheck.polynomial_evaluations[0][0] + QuinticExtension::ONE;
        assert!(
            verify_opening(num_variables, &commitment, &mutated, &point, value).is_err(),
            "tampered final sumcheck accepted"
        );
    }

    // A truncated final polynomial is rejected with the structural error.
    let mut mutated = proof.clone();
    mutated.final_poly.as_mut().unwrap().pop();
    assert_eq!(
        verify_opening(num_variables, &commitment, &mutated, &point, value),
        Err(WhirError::FinalPolyLengthMismatch {
            expected: 64,
            actual: 63
        })
    );

    // A dropped final sumcheck is rejected.
    let mut mutated = proof.clone();
    mutated.final_sumcheck = None;
    assert!(
        verify_opening(num_variables, &commitment, &mutated, &point, value).is_err(),
        "dropped final sumcheck accepted"
    );

    // The untampered proof still verifies after all mutations.
    verify_opening(num_variables, &commitment, &proof, &point, value).unwrap();
}

#[test]
fn grind_and_check_witness_roundtrip() {
    for bits in [1usize, 3, 6] {
        let mut transcript = PoseidonTranscript::new();
        transcript.observe(kb(99));
        let witness = transcript.grind(bits).unwrap();
        let mut verifier = PoseidonTranscript::new();
        verifier.observe(kb(99));
        assert!(verifier.check_witness(bits, witness).unwrap());
        // A different witness almost surely fails; at minimum, the honest
        // grind must not validate a mutated witness with zero probability.
        let mut wrong = PoseidonTranscript::new();
        wrong.observe(kb(99));
        let flipped = KoalaBear::from_u64(u64::from(witness.canonical()) + 1);
        let _ = wrong.check_witness(bits, flipped).unwrap();
    }
}

#[test]
fn stir_queries_are_sorted_distinct_in_range_and_saturating() {
    let mut transcript = PoseidonTranscript::new();
    transcript.observe(kb(1));
    let queries = get_challenge_stir_queries(1024, 1, 14, &mut transcript);
    assert_eq!(queries.len(), 14);
    assert!(queries.windows(2).all(|w| w[0] < w[1]));
    assert!(queries.iter().all(|&q| q < 512));

    // Saturation: more queries than positions opens the full domain.
    let mut transcript = PoseidonTranscript::new();
    let queries = get_challenge_stir_queries(16, 2, 75, &mut transcript);
    assert_eq!(queries, vec![0, 1, 2, 3]);
}

/// The Slice-4 chain fixture, reused for the DirectSparse end-to-end.
fn chain_shape() -> R1csShape<KoalaBear> {
    let entry = |row: usize, col: usize, value: u32| SparseMatEntry {
        row,
        col,
        val: kb(value),
    };
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

#[test]
fn direct_sparse_end_to_end_with_whir_pcs() {
    let shape = chain_shape();
    let public = vec![kb(3), kb(5)];
    let witness = R1csWitness {
        w: vec![kb(2), kb(4), kb(12), kb(60)],
    };

    let profile = test_profile();
    let num_variables = shape.num_vars.ilog2() as usize;
    let mut prover_pcs = WhirPcs::new(num_variables, &profile).unwrap();
    let mut transcript = PoseidonTranscript::new();
    let (instance, proof) = prove_direct_sparse(
        &shape,
        b"cirrus-whir-direct-sparse-e2e",
        &public,
        &witness,
        &mut prover_pcs,
        &mut transcript,
    )
    .unwrap();

    let verifier_pcs = WhirPcs::new(num_variables, &profile).unwrap();
    let mut transcript = PoseidonTranscript::new();
    verify_direct_sparse(
        &shape,
        b"cirrus-whir-direct-sparse-e2e",
        &instance,
        &proof,
        &verifier_pcs,
        &mut transcript,
    )
    .unwrap();

    // A wrong public value is rejected.
    let mut wrong_instance = instance.clone();
    wrong_instance.public_inputs[0] = wrong_instance.public_inputs[0] + KoalaBear::ONE;
    let verifier_pcs = WhirPcs::new(num_variables, &profile).unwrap();
    let mut transcript = PoseidonTranscript::new();
    assert!(
        verify_direct_sparse(
            &shape,
            b"cirrus-whir-direct-sparse-e2e",
            &wrong_instance,
            &proof,
            &verifier_pcs,
            &mut transcript,
        )
        .is_err()
    );

    // A tampered witness evaluation is rejected.
    let mut tampered = proof.clone();
    tampered.witness_eval = tampered.witness_eval + QuinticExtension::ONE;
    let verifier_pcs = WhirPcs::new(num_variables, &profile).unwrap();
    let mut transcript = PoseidonTranscript::new();
    assert!(
        verify_direct_sparse(
            &shape,
            b"cirrus-whir-direct-sparse-e2e",
            &instance,
            &tampered,
            &verifier_pcs,
            &mut transcript,
        )
        .is_err()
    );
}

#[cfg(feature = "oracle-tests")]
mod oracle {
    include!("whir/oracle.rs");
}
