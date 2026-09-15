// Differential oracle tests for the WHIR PCS against the pinned upstream
// `spartan-whir`/`p3-whir` stack. Included from `tests/whir.rs`.

use super::*;

use cirrus_spartan_whir::whir::domain_separator::WhirDomainSeparator;
use cirrus_spartan_whir::whir::pcs::WhirRoundProof;
use cirrus_spartan_whir::whir::sumcheck::SumcheckData;
use cirrus_spartan_whir::{dft_batch_base, dft_batch_ext};

use p3_challenger::{CanObserve, CanSample, CanSampleUniformBits, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{BasedVectorSpace, Field, PrimeField64, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

type OracleF = p3_koala_bear::KoalaBear;
type OracleEF = spartan_whir::QuinticExtension;
type OracleChallenger = spartan_whir::PoseidonChallenger;

fn oracle_kb(value: KoalaBear) -> OracleF {
    OracleF::new(value.canonical())
}

fn oracle_ext(value: QuinticExtension) -> OracleEF {
    OracleEF::from_basis_coefficients_slice(&value.canonical_coefficients().map(OracleF::new))
        .expect("quintic basis length")
}

fn assert_base_eq(ours: KoalaBear, oracle: OracleF) {
    assert_eq!(ours.canonical(), oracle.as_canonical_u64() as u32);
}

fn assert_ext_eq(ours: QuinticExtension, oracle: OracleEF) {
    let ours_coeffs = ours.canonical_coefficients();
    let oracle_coeffs = BasedVectorSpace::<OracleF>::as_basis_coefficients_slice(&oracle);
    for i in 0..5 {
        assert_eq!(
            ours_coeffs[i],
            oracle_coeffs[i].as_canonical_u64() as u32,
            "extension coefficient {i} mismatch"
        );
    }
}

fn oracle_challenger() -> OracleChallenger {
    spartan_whir::poseidon_challenger()
}

fn oracle_whir_config(num_variables: usize) -> spartan_whir::WhirPcsConfig {
    spartan_whir::WhirPcsConfig {
        num_variables,
        security: spartan_whir::SecurityConfig {
            security_level_bits: 80,
            merkle_security_bits: 80,
            soundness_assumption: spartan_whir::SoundnessAssumption::CapacityBound,
        },
        whir: spartan_whir::WhirParams {
            pow_bits: 0,
            folding_factor: 1,
            starting_log_inv_rate: 6,
            rs_domain_initial_reduction_factor: 1,
            folding_schedule: None,
            round_log_inv_rates: vec![],
        },
    }
}

#[test]
fn two_adic_generators_match_upstream() {
    for bits in 0..=24usize {
        assert_base_eq(two_adic_generator(bits), OracleF::two_adic_generator(bits));
    }
}

#[test]
fn base_dft_matches_upstream_small_batch_dft() {
    let dft = Radix2DFTSmallBatch::<OracleF>::default();
    for (log_height, width) in [(1usize, 3usize), (4, 2), (6, 1)] {
        let height = 1usize << log_height;
        let values: Vec<KoalaBear> = (0..height * width)
            .map(|i| kb((i as u32).wrapping_mul(40503).wrapping_add(7)))
            .collect();
        let oracle_values: Vec<OracleF> = values.iter().map(|&v| oracle_kb(v)).collect();
        let oracle_output = dft
            .dft_batch(RowMajorMatrix::new(oracle_values, width))
            .values;

        let mut ours = values.clone();
        dft_batch_base(&mut ours, height, width).unwrap();
        for (ours, oracle) in ours.iter().zip(&oracle_output) {
            assert_base_eq(*ours, *oracle);
        }
    }
}

#[test]
fn extension_dft_matches_upstream_algebra_batch() {
    let dft = Radix2DFTSmallBatch::<OracleF>::default();
    for (log_height, width) in [(2usize, 2usize), (5, 4)] {
        let height = 1usize << log_height;
        let values: Vec<QuinticExtension> = (0..height * width)
            .map(|i| {
                QuinticExtension::new(core::array::from_fn(
                    |j| kb((i * 5 + j) as u32 * 40503 + 11),
                ))
            })
            .collect();
        let oracle_values: Vec<OracleEF> = values.iter().map(|&v| oracle_ext(v)).collect();
        let oracle_output = dft
            .dft_algebra_batch(RowMajorMatrix::new(oracle_values, width))
            .values;

        let mut ours = values.clone();
        dft_batch_ext(&mut ours, height, width).unwrap();
        for (ours, oracle) in ours.iter().zip(&oracle_output) {
            assert_ext_eq(*ours, *oracle);
        }
    }
}

#[test]
fn whir_config_derivation_matches_upstream() {
    use p3_whir::parameters::{
        FoldingFactor as UpFolding, ProtocolParameters as UpParams,
        SecurityAssumption as UpSoundness, WhirConfig as UpConfig,
    };

    for num_variables in [2usize, 4, 8, 12] {
        let ours = test_profile().derive_config(num_variables).unwrap();
        let oracle = UpConfig::<OracleEF, OracleF, OracleChallenger>::new(
            num_variables,
            UpParams {
                starting_log_inv_rate: 6,
                round_log_inv_rates: ours.params.round_log_inv_rates.clone(),
                folding_factor: UpFolding::Constant(1),
                soundness_type: UpSoundness::CapacityBound,
                security_level: 80,
                pow_bits: 0,
            },
        )
        .unwrap();

        assert_eq!(ours.num_variables, oracle.num_variables);
        assert_eq!(ours.folding_schedule, oracle.folding_schedule);
        assert_eq!(ours.commitment_ood_samples, oracle.commitment_ood_samples);
        assert_eq!(
            ours.starting_folding_pow_bits,
            oracle.starting_folding_pow_bits
        );
        assert_eq!(ours.final_queries, oracle.final_queries);
        assert_eq!(ours.final_pow_bits, oracle.final_pow_bits);
        assert_eq!(ours.final_sumcheck_rounds, oracle.final_sumcheck_rounds);
        assert_eq!(ours.final_folding_pow_bits, oracle.final_folding_pow_bits);
        assert_eq!(ours.round_parameters.len(), oracle.round_parameters.len());
        for (ours, oracle) in ours.round_parameters.iter().zip(&oracle.round_parameters) {
            assert_eq!(ours.pow_bits, oracle.pow_bits, "round pow bits");
            assert_eq!(ours.folding_pow_bits, oracle.folding_pow_bits);
            assert_eq!(ours.num_queries, oracle.num_queries);
            assert_eq!(ours.ood_samples, oracle.ood_samples);
            assert_eq!(ours.num_variables, oracle.num_variables);
            assert_eq!(ours.folding_factor, oracle.folding_factor);
            assert_eq!(ours.log_inv_rate, oracle.log_inv_rate);
            assert_eq!(ours.domain_size, oracle.domain_size);
            assert_base_eq(ours.folded_domain_gen, oracle.folded_domain_gen);
        }
    }
}

#[test]
fn domain_separator_transcript_state_matches_upstream() {
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_whir::fiat_shamir::domain_separator::DomainSeparator as UpDomainSeparator;
    use p3_whir::parameters::{
        FoldingFactor as UpFolding, ProtocolParameters as UpParams,
        SecurityAssumption as UpSoundness, WhirConfig as UpConfig,
    };
    use p3_whir::pcs::prover::WhirProver;

    type Mmcs = MerkleTreeMmcs<
        <OracleF as Field>::Packing,
        <OracleF as Field>::Packing,
        spartan_whir::PoseidonFieldHash,
        spartan_whir::PoseidonNodeCompress,
        2,
        8,
    >;

    for num_variables in [4usize, 8] {
        let ours_config = test_profile().derive_config(num_variables).unwrap();
        let oracle_config = UpConfig::<OracleEF, OracleF, OracleChallenger>::new(
            num_variables,
            UpParams {
                starting_log_inv_rate: 6,
                round_log_inv_rates: ours_config.params.round_log_inv_rates.clone(),
                folding_factor: UpFolding::Constant(1),
                soundness_type: UpSoundness::CapacityBound,
                security_level: 80,
                pow_bits: 0,
            },
        )
        .unwrap();

        let mmcs = Mmcs::new(
            spartan_whir::poseidon_merkle_hash(),
            spartan_whir::poseidon_merkle_compress(),
            0,
        );
        let prover: WhirProver<
            OracleEF,
            OracleF,
            Radix2DFTSmallBatch<OracleF>,
            Mmcs,
            OracleChallenger,
            p3_sumcheck::layout::PrefixProver<OracleF, OracleEF>,
        > = WhirProver::new(oracle_config, Radix2DFTSmallBatch::default(), mmcs);
        let mut up_separator = UpDomainSeparator::new(Vec::new());
        prover.add_domain_separator::<8>(&mut up_separator);

        let mut up_challenger = oracle_challenger();
        up_separator.observe_domain_separator(&mut up_challenger);

        let ours_separator = WhirDomainSeparator::new(&ours_config);
        let mut our_transcript = PoseidonTranscript::new();
        ours_separator.observe_into(&mut our_transcript);

        for _ in 0..20 {
            assert_base_eq(our_transcript.sample_base(), up_challenger.sample());
        }
    }
}

#[test]
fn pruned_multi_openings_match_upstream_merkle_mmcs() {
    use p3_merkle_tree::MerkleTreeMmcs;

    type Mmcs = MerkleTreeMmcs<
        <OracleF as Field>::Packing,
        <OracleF as Field>::Packing,
        spartan_whir::PoseidonFieldHash,
        spartan_whir::PoseidonNodeCompress,
        2,
        8,
    >;

    let mmcs = Mmcs::new(
        spartan_whir::poseidon_merkle_hash(),
        spartan_whir::poseidon_merkle_compress(),
        0,
    );

    let (height, width) = (32usize, 4usize);
    let rows: Vec<Vec<KoalaBear>> = (0..height)
        .map(|r| {
            (0..width)
                .map(|c| kb((r * width + c) as u32 * 40503 + 3))
                .collect()
        })
        .collect();
    let flat: Vec<KoalaBear> = rows.iter().flatten().copied().collect();

    // Upstream commit + open.
    let oracle_flat: Vec<OracleF> = flat.iter().map(|&v| oracle_kb(v)).collect();
    let (cap, prover_data) = mmcs.commit(vec![RowMajorMatrix::new(oracle_flat, width)]);
    let indices = [1usize, 2, 5, 8, 13, 27];
    let (up_values, up_proof) = mmcs.open_multi_batch(&indices, &prover_data);

    // Our commit + open.
    let tree = PoseidonMerkleTree::commit_rows(&rows).unwrap();
    let (our_opened, our_proof) = tree.open_multi(&indices, &rows).unwrap();

    // Same root.
    assert_eq!(cap.num_roots(), 1);
    for i in 0..8 {
        assert_base_eq(tree.root()[i], cap[0][i]);
    }

    // Same opened rows and identical pruned digest wire format.
    for (ours, up) in our_opened.iter().zip(&up_values) {
        assert_eq!(up.len(), 1);
        for (ours, up) in ours.iter().zip(&up[0]) {
            assert_base_eq(*ours, *up);
        }
    }
    assert_eq!(
        our_proof.sibling_hashes.len(),
        up_proof.sibling_hashes.len()
    );
    for (ours, up) in our_proof
        .sibling_hashes
        .iter()
        .zip(&up_proof.sibling_hashes)
    {
        for i in 0..8 {
            assert_base_eq(ours[i], up[i]);
        }
    }

    // Cross-verify: our proof through the upstream verifier and vice versa.
    let dims = vec![p3_matrix::Dimensions { height, width }];
    let up_opened_refs: Vec<Vec<&[OracleF]>> = up_values
        .iter()
        .map(|per| per.iter().map(|row| row.as_slice()).collect())
        .collect();
    mmcs.verify_multi_batch(&cap, &dims, &indices, &up_opened_refs, &up_proof)
        .unwrap();

    let cross_proof = p3_merkle_tree::PrunedMerklePaths {
        sibling_hashes: our_proof
            .sibling_hashes
            .iter()
            .map(|d| core::array::from_fn(|i| oracle_kb(d[i])))
            .collect(),
    };
    mmcs.verify_multi_batch(&cap, &dims, &indices, &up_opened_refs, &cross_proof)
        .expect("our pruned proof must pass the upstream verifier");

    let upstream_proof_in_our_shape = PrunedMerklePaths {
        sibling_hashes: up_proof
            .sibling_hashes
            .iter()
            .map(|d| core::array::from_fn(|i| kb(d[i].as_canonical_u64() as u32)))
            .collect(),
    };
    PoseidonMerkleTree::verify_multi(
        &tree.root(),
        height,
        width,
        &indices,
        &our_opened,
        &upstream_proof_in_our_shape,
    )
    .expect("upstream pruned proof must pass our verifier");
}

#[test]
fn pow_grind_and_uniform_bits_match_upstream() {
    for bits in [1usize, 4, 7] {
        let mut ours = PoseidonTranscript::new();
        ours.observe(kb(0xc0ffee));
        let witness = ours.grind(bits).unwrap();

        let mut up = oracle_challenger();
        up.observe(oracle_kb(kb(0xc0ffee)));
        let up_witness = GrindingChallenger::grind(&mut up, bits);

        assert_base_eq(witness, up_witness);
        // Post-grind transcript states agree.
        assert_base_eq(ours.sample_base(), up.sample());

        // check_witness agrees on accept and reject.
        let mut ours = PoseidonTranscript::new();
        ours.observe(kb(0xc0ffee));
        assert!(ours.check_witness(bits, witness).unwrap());
        let mut up = oracle_challenger();
        up.observe(oracle_kb(kb(0xc0ffee)));
        assert!(GrindingChallenger::check_witness(&mut up, bits, up_witness));

        let wrong = KoalaBear::from_u64(u64::from(witness.canonical()) + 1);
        let mut ours = PoseidonTranscript::new();
        ours.observe(kb(0xc0ffee));
        let ours_wrong = ours.check_witness(bits, wrong).unwrap();
        let mut up = oracle_challenger();
        up.observe(oracle_kb(kb(0xc0ffee)));
        let up_wrong = GrindingChallenger::check_witness(&mut up, bits, oracle_kb(wrong));
        assert_eq!(ours_wrong, up_wrong, "reject decision mismatch");
    }

    // Uniform bit sampling, including the two-sample slow path.
    for k in [1usize, 9, 24, 30] {
        let mut ours = PoseidonTranscript::new();
        ours.observe(kb(5));
        let mut up = oracle_challenger();
        up.observe(oracle_kb(kb(5)));
        for _ in 0..4 {
            let ours = ours.sample_uniform_bits(k).unwrap();
            let up = up.sample_uniform_bits::<true>(k).unwrap();
            assert_eq!(ours, up, "uniform bit sample mismatch at {k} bits");
        }
    }
}

/// Convert our proof into the upstream proof shape (in place over a clone
/// of the upstream proof) for cross-verification.
fn overwrite_upstream_proof(
    target: &mut <spartan_whir::Plonky3WhirPcs as spartan_whir::MlePcs<
        spartan_whir::PoseidonQuinticEngine,
    >>::Proof,
    ours: &WhirProof,
) {
    use p3_merkle_tree::PrunedMerklePaths as UpPruned;
    use p3_multilinear_util::poly::Poly as UpPoly;
    use p3_whir::pcs::proof::{
        QueryOpenings as UpQueryOpenings, SharedProofOpening as UpSharedProofOpening,
        SumcheckData as UpSumcheckData,
    };

    let up_digest = |d: &PoseidonDigest| core::array::from_fn(|i| oracle_kb(d[i]));
    let up_cap = |d: &PoseidonDigest| p3_symmetric::MerkleCap::new(vec![up_digest(d)]);
    let up_sumcheck = |s: &SumcheckData| UpSumcheckData {
        polynomial_evaluations: s
            .polynomial_evaluations
            .iter()
            .map(|&[c0, c_inf]| [oracle_ext(c0), oracle_ext(c_inf)])
            .collect(),
        pow_witnesses: s.pow_witnesses.iter().map(|&w| oracle_kb(w)).collect(),
    };

    target.initial_ood_answers = ours
        .initial_ood_answers
        .iter()
        .map(|&v| oracle_ext(v))
        .collect();
    target.initial_sumcheck = up_sumcheck(&ours.initial_sumcheck);
    for (target_round, our_round) in target.rounds.iter_mut().zip(&ours.rounds) {
        target_round.commitment = our_round.commitment.as_ref().map(|d| up_cap(d));
        target_round.ood_answers = our_round
            .ood_answers
            .iter()
            .map(|&v| oracle_ext(v))
            .collect();
        target_round.pow_witness = oracle_kb(our_round.pow_witness);
        target_round.openings = match &our_round.openings {
            QueryOpenings::Base(opening) => UpQueryOpenings::Base(UpSharedProofOpening {
                rows: opening
                    .rows
                    .iter()
                    .map(|row| row.iter().map(|&v| oracle_kb(v)).collect())
                    .collect(),
                proof: UpPruned {
                    sibling_hashes: opening
                        .proof
                        .sibling_hashes
                        .iter()
                        .map(|d| up_digest(d))
                        .collect(),
                },
            }),
            QueryOpenings::Extension(opening) => UpQueryOpenings::Extension(UpSharedProofOpening {
                rows: opening
                    .rows
                    .iter()
                    .map(|row| row.iter().map(|&v| oracle_ext(v)).collect())
                    .collect(),
                proof: UpPruned {
                    sibling_hashes: opening
                        .proof
                        .sibling_hashes
                        .iter()
                        .map(|d| up_digest(d))
                        .collect(),
                },
            }),
        };
        target_round.sumcheck = up_sumcheck(&our_round.sumcheck);
    }
    target.final_poly = ours
        .final_poly
        .as_ref()
        .map(|poly| UpPoly::new(poly.iter().map(|&v| oracle_ext(v)).collect()));
    target.final_pow_witness = oracle_kb(ours.final_pow_witness);
    target.final_openings = match &ours.final_openings {
        QueryOpenings::Base(opening) => UpQueryOpenings::Base(UpSharedProofOpening {
            rows: opening
                .rows
                .iter()
                .map(|row| row.iter().map(|&v| oracle_kb(v)).collect())
                .collect(),
            proof: UpPruned {
                sibling_hashes: opening
                    .proof
                    .sibling_hashes
                    .iter()
                    .map(|d| up_digest(d))
                    .collect(),
            },
        }),
        QueryOpenings::Extension(opening) => UpQueryOpenings::Extension(UpSharedProofOpening {
            rows: opening
                .rows
                .iter()
                .map(|row| row.iter().map(|&v| oracle_ext(v)).collect())
                .collect(),
            proof: UpPruned {
                sibling_hashes: opening
                    .proof
                    .sibling_hashes
                    .iter()
                    .map(|d| up_digest(d))
                    .collect(),
            },
        }),
    };
    target.final_sumcheck = ours.final_sumcheck.as_ref().map(|s| up_sumcheck(s));
}

/// Convert the upstream proof into our proof shape for cross-verification.
fn convert_upstream_proof(
    upstream: &<spartan_whir::Plonky3WhirPcs as spartan_whir::MlePcs<
        spartan_whir::PoseidonQuinticEngine,
    >>::Proof,
) -> WhirProof {
    let our_kb = |v: &OracleF| kb(v.as_canonical_u64() as u32);
    let our_ext = |v: &OracleEF| {
        QuinticExtension::new(core::array::from_fn(|i| {
            kb(
                BasedVectorSpace::<OracleF>::as_basis_coefficients_slice(v)[i].as_canonical_u64()
                    as u32,
            )
        }))
    };
    let our_digest = |d: &[OracleF; 8]| core::array::from_fn(|i| our_kb(&d[i]));
    let our_sumcheck = |s: &p3_sumcheck::SumcheckData<OracleF, OracleEF>| SumcheckData {
        polynomial_evaluations: s
            .polynomial_evaluations
            .iter()
            .map(|&[c0, c_inf]| [our_ext(&c0), our_ext(&c_inf)])
            .collect(),
        pow_witnesses: s.pow_witnesses.iter().map(|w| our_kb(w)).collect(),
    };
    let our_openings = |openings: &p3_whir::pcs::proof::QueryOpenings<
        OracleF,
        OracleEF,
        p3_merkle_tree::PrunedMerklePaths<OracleF, 8>,
    >| match openings {
        p3_whir::pcs::proof::QueryOpenings::Base(opening) => {
            QueryOpenings::Base(SharedProofOpening {
                rows: opening
                    .rows
                    .iter()
                    .map(|row| row.iter().map(|v| our_kb(v)).collect())
                    .collect(),
                proof: PrunedMerklePaths {
                    sibling_hashes: opening
                        .proof
                        .sibling_hashes
                        .iter()
                        .map(|d| our_digest(d))
                        .collect(),
                },
            })
        }
        p3_whir::pcs::proof::QueryOpenings::Extension(opening) => {
            QueryOpenings::Extension(SharedProofOpening {
                rows: opening
                    .rows
                    .iter()
                    .map(|row| row.iter().map(|v| our_ext(v)).collect())
                    .collect(),
                proof: PrunedMerklePaths {
                    sibling_hashes: opening
                        .proof
                        .sibling_hashes
                        .iter()
                        .map(|d| our_digest(d))
                        .collect(),
                },
            })
        }
    };

    WhirProof {
        initial_ood_answers: upstream
            .initial_ood_answers
            .iter()
            .map(|v| our_ext(v))
            .collect(),
        initial_sumcheck: our_sumcheck(&upstream.initial_sumcheck),
        rounds: upstream
            .rounds
            .iter()
            .map(|round| WhirRoundProof {
                commitment: round.commitment.as_ref().map(|cap| our_digest(&cap[0])),
                ood_answers: round.ood_answers.iter().map(|v| our_ext(v)).collect(),
                pow_witness: our_kb(&round.pow_witness),
                openings: our_openings(&round.openings),
                sumcheck: our_sumcheck(&round.sumcheck),
            })
            .collect(),
        final_poly: upstream
            .final_poly
            .as_ref()
            .map(|poly| poly.as_slice().iter().map(|v| our_ext(v)).collect()),
        final_pow_witness: our_kb(&upstream.final_pow_witness),
        final_openings: our_openings(&upstream.final_openings),
        final_sumcheck: upstream.final_sumcheck.as_ref().map(|s| our_sumcheck(s)),
    }
}

#[test]
fn full_pcs_proof_matches_upstream_and_cross_verifies() {
    use spartan_whir::{MlePcs, Plonky3WhirPcs, PoseidonQuinticEngine};

    for num_variables in [4usize, 8] {
        let poly = witness(num_variables);
        let oracle_poly: Vec<OracleF> = poly.iter().map(|&v| oracle_kb(v)).collect();
        let config = oracle_whir_config(num_variables);

        // The opening point: public input derived off-schedule.
        let mut point_transcript = PoseidonTranscript::new();
        point_transcript.observe(kb(num_variables as u32));
        let point: Vec<QuinticExtension> = (0..num_variables)
            .map(|_| point_transcript.sample_quintic())
            .collect();
        let value = cirrus_spartan_whir::whir::pcs::eval_mle_base_at_ext(&poly, &point);
        let oracle_point: Vec<OracleEF> = point.iter().map(|&v| oracle_ext(v)).collect();
        let oracle_value = oracle_ext(value);

        // --- Upstream commit + open ---
        let mut up_challenger = oracle_challenger();
        let (up_commitment, up_prover_data) =
            <Plonky3WhirPcs as MlePcs<PoseidonQuinticEngine>>::commit(
                &config,
                &oracle_poly,
                &mut up_challenger,
            )
            .unwrap();
        let statement = spartan_whir::PcsStatementBuilder::<PoseidonQuinticEngine>::new()
            .add_point_eval(spartan_whir::PointEvalClaim {
                point: spartan_whir::MultilinearPoint(oracle_point.clone()),
                value: oracle_value,
            })
            .finalize()
            .unwrap();
        let up_proof = <Plonky3WhirPcs as MlePcs<PoseidonQuinticEngine>>::open(
            &config,
            up_prover_data,
            &statement,
            &mut up_challenger,
        )
        .unwrap();

        // --- Our commit + open ---
        let mut our_pcs = pcs(num_variables);
        let mut our_transcript = PoseidonTranscript::new();
        let our_commitment = our_pcs.commit(&poly, &mut our_transcript).unwrap();
        let our_proof = our_pcs.open(&point, value, &mut our_transcript).unwrap();

        // --- Byte-level commitments and proof parity ---
        assert_eq!(up_commitment.num_roots(), 1);
        for i in 0..8 {
            assert_base_eq(our_commitment[i], up_commitment[0][i]);
        }
        assert_eq!(
            our_proof.initial_ood_answers.len(),
            up_proof.initial_ood_answers.len()
        );
        for (ours, up) in our_proof
            .initial_ood_answers
            .iter()
            .zip(&up_proof.initial_ood_answers)
        {
            assert_ext_eq(*ours, *up);
        }
        assert_eq!(
            our_proof.initial_sumcheck.polynomial_evaluations.len(),
            up_proof.initial_sumcheck.polynomial_evaluations.len()
        );
        for (ours, up) in our_proof
            .initial_sumcheck
            .polynomial_evaluations
            .iter()
            .zip(&up_proof.initial_sumcheck.polynomial_evaluations)
        {
            assert_ext_eq(ours[0], up[0]);
            assert_ext_eq(ours[1], up[1]);
        }
        assert_eq!(our_proof.rounds.len(), up_proof.rounds.len());
        for (our_round, up_round) in our_proof.rounds.iter().zip(&up_proof.rounds) {
            for i in 0..8 {
                assert_base_eq(
                    our_round.commitment.as_ref().unwrap()[i],
                    up_round.commitment.as_ref().unwrap()[0][i],
                );
            }
            assert_eq!(our_round.ood_answers.len(), up_round.ood_answers.len());
            for (ours, up) in our_round.ood_answers.iter().zip(&up_round.ood_answers) {
                assert_ext_eq(*ours, *up);
            }
            assert_base_eq(our_round.pow_witness, up_round.pow_witness);
            assert_eq!(
                our_round.sumcheck.polynomial_evaluations.len(),
                up_round.sumcheck.polynomial_evaluations.len()
            );
            for (ours, up) in our_round
                .sumcheck
                .polynomial_evaluations
                .iter()
                .zip(&up_round.sumcheck.polynomial_evaluations)
            {
                assert_ext_eq(ours[0], up[0]);
                assert_ext_eq(ours[1], up[1]);
            }
            // Opening rows and pruned digests.
            let (our_rows, our_siblings) = match &our_round.openings {
                QueryOpenings::Base(opening) => (
                    opening
                        .rows
                        .iter()
                        .map(|row| row.iter().map(|&v| oracle_kb(v)).collect::<Vec<_>>())
                        .collect::<Vec<_>>(),
                    &opening.proof.sibling_hashes,
                ),
                QueryOpenings::Extension(opening) => (
                    opening
                        .rows
                        .iter()
                        .map(|row| {
                            row.iter()
                                .flat_map(|v| {
                                    Vec::from(v.canonical_coefficients().map(OracleF::new))
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>(),
                    &opening.proof.sibling_hashes,
                ),
            };
            let (up_rows, up_siblings) = match &up_round.openings {
                p3_whir::pcs::proof::QueryOpenings::Base(opening) => {
                    (opening.rows.clone(), &opening.proof.sibling_hashes)
                }
                p3_whir::pcs::proof::QueryOpenings::Extension(opening) => (
                    opening
                        .rows
                        .iter()
                        .map(|row| {
                            row.iter()
                                .flat_map(|v| {
                                    BasedVectorSpace::<OracleF>::as_basis_coefficients_slice(v)
                                        .to_vec()
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect(),
                    &opening.proof.sibling_hashes,
                ),
            };
            assert_eq!(our_rows, up_rows, "STIR opening rows mismatch");
            assert_eq!(our_siblings.len(), up_siblings.len());
            for (ours, up) in our_siblings.iter().zip(up_siblings) {
                for i in 0..8 {
                    assert_base_eq(ours[i], up[i]);
                }
            }
        }
        let up_final: Vec<OracleEF> = up_proof.final_poly.as_ref().unwrap().as_slice().to_vec();
        let our_final = our_proof.final_poly.as_ref().unwrap();
        assert_eq!(our_final.len(), up_final.len());
        for (ours, up) in our_final.iter().zip(&up_final) {
            assert_ext_eq(*ours, *up);
        }
        assert_base_eq(our_proof.final_pow_witness, up_proof.final_pow_witness);
        match (&our_proof.final_sumcheck, &up_proof.final_sumcheck) {
            (Some(ours), Some(up)) => {
                assert_eq!(
                    ours.polynomial_evaluations.len(),
                    up.polynomial_evaluations.len()
                );
                for (ours, up) in ours
                    .polynomial_evaluations
                    .iter()
                    .zip(&up.polynomial_evaluations)
                {
                    assert_ext_eq(ours[0], up[0]);
                    assert_ext_eq(ours[1], up[1]);
                }
            }
            (None, None) => {}
            _ => panic!("final sumcheck presence mismatch"),
        }

        // --- Cross-verification both ways ---
        // Upstream verifier on our proof (rewritten into upstream types).
        let mut cross_proof = up_proof.clone();
        overwrite_upstream_proof(&mut cross_proof, &our_proof);
        let mut up_verify_challenger = oracle_challenger();
        <Plonky3WhirPcs as MlePcs<PoseidonQuinticEngine>>::verify(
            &config,
            &up_commitment,
            &statement,
            &cross_proof,
            &mut up_verify_challenger,
        )
        .expect("upstream verifier must accept our byte-identical proof");

        // Our verifier on the upstream proof (converted into our types).
        let converted = convert_upstream_proof(&up_proof);
        let our_root: PoseidonDigest =
            core::array::from_fn(|i| kb(up_commitment[0][i].as_canonical_u64() as u32));
        verify_opening(num_variables, &our_root, &converted, &point, value)
            .expect("our verifier must accept the upstream proof");

        // And the honest paths agree as well.
        let mut up_verify_challenger = oracle_challenger();
        <Plonky3WhirPcs as MlePcs<PoseidonQuinticEngine>>::verify(
            &config,
            &up_commitment,
            &statement,
            &up_proof,
            &mut up_verify_challenger,
        )
        .unwrap();
        verify_opening(num_variables, &our_commitment, &our_proof, &point, value).unwrap();
    }
}
