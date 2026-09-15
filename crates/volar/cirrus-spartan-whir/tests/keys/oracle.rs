// Differential oracle tests for the key API against the pinned upstream
// `spartan-whir` stack. Included from `tests/keys.rs`.

use super::*;

use cirrus_spartan_whir::whir::pcs::{QueryOpenings, SharedProofOpening, WhirRoundProof};
use cirrus_spartan_whir::whir::sumcheck::SumcheckData;
use cirrus_spartan_whir::{PoseidonDigest, PrunedMerklePaths};

use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};

type OracleF = p3_koala_bear::KoalaBear;
type OracleEF = spartan_whir::QuinticExtension;
type OracleProtocol = spartan_whir::PoseidonSpartanProtocol<OracleEF>;

fn oracle_kb(value: KoalaBear) -> OracleF {
    OracleF::new(value.canonical())
}

fn oracle_ext(value: QuinticExtension) -> OracleEF {
    OracleEF::from_basis_coefficients_slice(&value.canonical_coefficients().map(OracleF::new))
        .expect("quintic basis length")
}

fn oracle_shape(shape: &R1csShape<KoalaBear>) -> spartan_whir::R1csShape<OracleF> {
    let matrix = |matrix: &SparseMatrix<KoalaBear>| spartan_whir::SparseMatrix {
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
    };
    spartan_whir::R1csShape {
        num_cons: shape.num_cons,
        num_vars: shape.num_vars,
        num_io: shape.num_io,
        a: matrix(&shape.a),
        b: matrix(&shape.b),
        c: matrix(&shape.c),
    }
}

fn oracle_security(profile: &SecurityProfile) -> spartan_whir::SecurityConfig {
    spartan_whir::SecurityConfig {
        security_level_bits: profile.security_level_bits,
        merkle_security_bits: profile.merkle_security_bits,
        soundness_assumption: match profile.soundness {
            SecurityAssumption::UniqueDecoding => spartan_whir::SoundnessAssumption::UniqueDecoding,
            SecurityAssumption::JohnsonBound => spartan_whir::SoundnessAssumption::JohnsonBound,
            SecurityAssumption::CapacityBound => spartan_whir::SoundnessAssumption::CapacityBound,
        },
    }
}

fn oracle_whir_params(params: &WhirParams) -> spartan_whir::WhirParams {
    spartan_whir::WhirParams {
        pow_bits: params.pow_bits,
        folding_factor: params.folding_factor,
        starting_log_inv_rate: params.starting_log_inv_rate,
        rs_domain_initial_reduction_factor: params.rs_domain_initial_reduction_factor,
        folding_schedule: params
            .folding_schedule
            .as_ref()
            .map(|schedule| {
                match schedule {
                cirrus_spartan_whir::whir::spartan::WhirFoldingSchedule::Constant(factor) => {
                    spartan_whir::WhirFoldingSchedule::Constant(*factor)
                }
                cirrus_spartan_whir::whir::spartan::WhirFoldingSchedule::ConstantFromSecondRound {
                    first,
                    rest,
                } => spartan_whir::WhirFoldingSchedule::ConstantFromSecondRound {
                    first: *first,
                    rest: *rest,
                },
                cirrus_spartan_whir::whir::spartan::WhirFoldingSchedule::PerRound(factors) => {
                    spartan_whir::WhirFoldingSchedule::PerRound(factors.clone())
                }
            }
            }),
        round_log_inv_rates: params.round_log_inv_rates.clone(),
    }
}

fn oracle_config(profile: &SecurityProfile) -> spartan_whir::SpartanSnarkConfig {
    spartan_whir::SpartanSnarkConfig {
        matrix_closing: spartan_whir::MatrixClosingMode::DirectSparse,
        security: oracle_security(profile),
        whir_params: oracle_whir_params(&profile.whir),
        spark_whir_params: None,
    }
}

#[test]
fn setup_matches_upstream_keys() {
    let profile = test_profile();
    let (pk, vk) = setup(&chain_shape(), &profile).unwrap();

    let upstream_shape = oracle_shape(&chain_shape());
    let (up_pk, up_vk) =
        OracleProtocol::setup_with_config(&upstream_shape, &oracle_config(&profile)).unwrap();

    // Domain separator bytes.
    assert_eq!(
        pk.domain_separator(),
        up_pk.domain_separator.to_bytes().as_slice()
    );
    assert_eq!(
        vk.domain_separator(),
        up_vk.domain_separator().to_bytes().as_slice()
    );

    // Composed component security levels.
    assert_eq!(
        pk.component_security().security_level_bits,
        up_vk.pcs_config().security.security_level_bits
    );
    assert_eq!(
        pk.component_security().merkle_security_bits,
        up_vk.pcs_config().security.merkle_security_bits
    );

    // Canonical padded shape and WHIR parameter binding.
    assert_eq!(
        vk.shape_canonical().num_cons,
        up_vk.shape_canonical().num_cons
    );
    assert_eq!(
        vk.shape_canonical().num_vars,
        up_vk.shape_canonical().num_vars
    );
    assert_eq!(vk.shape_canonical().num_io, up_vk.shape_canonical().num_io);
    assert_eq!(
        vk.shape_canonical().a.entries.len(),
        up_vk.shape_canonical().a.entries.len()
    );
    for (ours, up) in vk
        .shape_canonical()
        .a
        .entries
        .iter()
        .zip(&up_vk.shape_canonical().a.entries)
    {
        assert_eq!(ours.row, up.row);
        assert_eq!(ours.col, up.col);
        assert_eq!(ours.val.canonical(), up.val.as_canonical_u64() as u32);
    }
    assert_eq!(
        vk.pcs_config().num_variables,
        up_vk.pcs_config().num_variables
    );
    assert_eq!(
        oracle_whir_params(&vk.profile().whir),
        up_vk.pcs_config().whir
    );
}

#[test]
fn attainable_security_boundary_matches_upstream_setup() {
    let shape = chain_shape();
    let upstream_shape = oracle_shape(&shape);

    // Binary-search the largest requested level our setup accepts.
    let mut low = 80u32;
    let mut high = 124u32; // always rejected (above the digest maximum)
    while high - low > 1 {
        let mid = (low + high) / 2;
        let mut profile = test_profile();
        profile.security_level_bits = mid;
        profile.merkle_security_bits = mid;
        if setup(&shape, &profile).is_ok() {
            low = mid;
        } else {
            high = mid;
        }
    }

    // The upstream setup must accept exactly the same boundary.
    let mut profile = test_profile();
    profile.security_level_bits = low;
    profile.merkle_security_bits = low;
    OracleProtocol::setup_with_config(&upstream_shape, &oracle_config(&profile))
        .expect("upstream setup must accept our attainable level");

    let mut profile = test_profile();
    profile.security_level_bits = low + 1;
    profile.merkle_security_bits = low + 1;
    assert!(
        OracleProtocol::setup_with_config(&upstream_shape, &oracle_config(&profile)).is_err(),
        "upstream setup must reject one bit above our attainable level"
    );
}

fn oracle_sumcheck_data(data: &SumcheckData) -> p3_sumcheck::SumcheckData<OracleF, OracleEF> {
    p3_sumcheck::SumcheckData {
        polynomial_evaluations: data
            .polynomial_evaluations
            .iter()
            .map(|&[c0, c_inf]| [oracle_ext(c0), oracle_ext(c_inf)])
            .collect(),
        pow_witnesses: data.pow_witnesses.iter().map(|&w| oracle_kb(w)).collect(),
    }
}

fn oracle_digest(digest: &PoseidonDigest) -> [OracleF; 8] {
    core::array::from_fn(|i| oracle_kb(digest[i]))
}

/// Convert our WHIR proof into the upstream proof shape (constructing the
/// upstream cap/proof types through a cloned upstream proof skeleton).
fn overwrite_upstream_whir_proof(
    target: &mut <spartan_whir::Plonky3WhirPcs as spartan_whir::MlePcs<
        spartan_whir::PoseidonQuinticEngine,
    >>::Proof,
    ours: &cirrus_spartan_whir::whir::pcs::WhirProof,
) {
    use p3_merkle_tree::PrunedMerklePaths as UpPruned;
    use p3_multilinear_util::poly::Poly as UpPoly;
    use p3_whir::pcs::proof::{
        QueryOpenings as UpQueryOpenings, SharedProofOpening as UpSharedProofOpening,
    };

    let up_cap =
        |digest: &PoseidonDigest| p3_symmetric::MerkleCap::new(vec![oracle_digest(digest)]);
    let up_openings = |openings: &QueryOpenings| match openings {
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
                    .map(oracle_digest)
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
                    .map(oracle_digest)
                    .collect(),
            },
        }),
    };

    target.initial_ood_answers = ours
        .initial_ood_answers
        .iter()
        .map(|&v| oracle_ext(v))
        .collect();
    target.initial_sumcheck = oracle_sumcheck_data(&ours.initial_sumcheck);
    for (target_round, our_round) in target.rounds.iter_mut().zip(&ours.rounds) {
        target_round.commitment = our_round.commitment.as_ref().map(|d| up_cap(d));
        target_round.ood_answers = our_round
            .ood_answers
            .iter()
            .map(|&v| oracle_ext(v))
            .collect();
        target_round.pow_witness = oracle_kb(our_round.pow_witness);
        target_round.openings = up_openings(&our_round.openings);
        target_round.sumcheck = oracle_sumcheck_data(&our_round.sumcheck);
    }
    target.final_poly = ours
        .final_poly
        .as_ref()
        .map(|poly| UpPoly::new(poly.iter().map(|&v| oracle_ext(v)).collect()));
    target.final_pow_witness = oracle_kb(ours.final_pow_witness);
    target.final_openings = up_openings(&ours.final_openings);
    target.final_sumcheck = ours.final_sumcheck.as_ref().map(oracle_sumcheck_data);
}

/// Convert an upstream WHIR proof into our proof shape.
fn convert_upstream_whir_proof(
    upstream: &<spartan_whir::Plonky3WhirPcs as spartan_whir::MlePcs<
        spartan_whir::PoseidonQuinticEngine,
    >>::Proof,
) -> cirrus_spartan_whir::whir::pcs::WhirProof {
    let our_kb = |v: &OracleF| KoalaBear::from_u64(v.as_canonical_u64());
    let our_ext = |v: &OracleEF| {
        QuinticExtension::new(core::array::from_fn(|i| {
            KoalaBear::from_u64(
                BasedVectorSpace::<OracleF>::as_basis_coefficients_slice(v)[i].as_canonical_u64(),
            )
        }))
    };
    let our_digest = |digest: &[OracleF; 8]| core::array::from_fn(|i| our_kb(&digest[i]));
    let our_sumcheck = |data: &p3_sumcheck::SumcheckData<OracleF, OracleEF>| SumcheckData {
        polynomial_evaluations: data
            .polynomial_evaluations
            .iter()
            .map(|&[c0, c_inf]| [our_ext(&c0), our_ext(&c_inf)])
            .collect(),
        pow_witnesses: data.pow_witnesses.iter().map(|w| our_kb(w)).collect(),
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
                        .map(our_digest)
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
                        .map(our_digest)
                        .collect(),
                },
            })
        }
    };

    cirrus_spartan_whir::whir::pcs::WhirProof {
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
fn full_proof_matches_upstream_and_cross_verifies_both_ways() {
    let profile = test_profile();
    let (pk, vk) = setup(&chain_shape(), &profile).unwrap();
    let our_proof = pk.prove(&chain_witness(), &chain_public()).unwrap();

    let upstream_shape = oracle_shape(&chain_shape());
    let (up_pk, up_vk) =
        OracleProtocol::setup_with_config(&upstream_shape, &oracle_config(&profile)).unwrap();
    let upstream_witness = spartan_whir::R1csWitness {
        w: chain_witness().into_iter().map(oracle_kb).collect(),
    };
    let upstream_public: Vec<OracleF> = chain_public().into_iter().map(oracle_kb).collect();
    let mut up_challenger = spartan_whir::poseidon_challenger();
    let (up_instance, up_proof) = OracleProtocol::prove(
        &up_pk,
        &upstream_public,
        &upstream_witness,
        &mut up_challenger,
    )
    .unwrap();
    // A second identical upstream proof to consume for cross-verification
    // (the upstream proof type is not Clone).
    let mut up_challenger = spartan_whir::poseidon_challenger();
    let (_, up_proof_for_cross) = OracleProtocol::prove(
        &up_pk,
        &upstream_public,
        &spartan_whir::R1csWitness {
            w: chain_witness().into_iter().map(oracle_kb).collect(),
        },
        &mut up_challenger,
    )
    .unwrap();

    // Our proof is byte-identical to the upstream proof.
    assert_eq!(up_instance.public_inputs, upstream_public);
    assert_eq!(
        our_proof.proof.outer_sumcheck.rounds.len(),
        up_proof.outer_sumcheck.rounds.len()
    );
    for (ours, up) in our_proof
        .proof
        .outer_sumcheck
        .rounds
        .iter()
        .zip(&up_proof.outer_sumcheck.rounds)
    {
        for i in 0..3 {
            assert_eq!(
                BasedVectorSpace::<OracleF>::as_basis_coefficients_slice(&up.0[i])
                    .iter()
                    .map(|v| v.as_canonical_u64() as u32)
                    .collect::<Vec<_>>(),
                ours.0[i].canonical_coefficients().to_vec(),
                "outer round coefficient mismatch"
            );
        }
    }
    assert_eq!(
        our_proof.proof.inner_sumcheck.rounds.len(),
        up_proof.inner_sumcheck.rounds.len()
    );
    for (ours, up) in our_proof
        .proof
        .inner_sumcheck
        .rounds
        .iter()
        .zip(&up_proof.inner_sumcheck.rounds)
    {
        for i in 0..2 {
            assert_eq!(
                BasedVectorSpace::<OracleF>::as_basis_coefficients_slice(&up.0[i])
                    .iter()
                    .map(|v| v.as_canonical_u64() as u32)
                    .collect::<Vec<_>>(),
                ours.0[i].canonical_coefficients().to_vec(),
                "inner round coefficient mismatch"
            );
        }
    }
    assert_eq!(up_instance.witness_commitment.num_roots(), 1);
    for i in 0..8 {
        assert_eq!(
            our_proof.witness_commitment[i].canonical(),
            up_instance.witness_commitment[0][i].as_canonical_u64() as u32
        );
    }

    // Cross-verification: the upstream verifier accepts our proof. Rebuild
    // an upstream proof with our (byte-identical) field values.
    let spartan_whir::SpartanProof {
        pcs_proof: mut cross_pcs_proof,
        ..
    } = up_proof_for_cross;
    overwrite_upstream_whir_proof(&mut cross_pcs_proof, &our_proof.proof.pcs_proof);
    let cross_proof = spartan_whir::SpartanProof {
        pcs_proof: cross_pcs_proof,
        outer_sumcheck: spartan_whir::OuterSumcheckProof {
            rounds: our_proof
                .proof
                .outer_sumcheck
                .rounds
                .iter()
                .map(|round| spartan_whir::CubicRoundPoly(round.0.map(oracle_ext)))
                .collect(),
        },
        outer_claims: (
            oracle_ext(our_proof.proof.outer_claims.0),
            oracle_ext(our_proof.proof.outer_claims.1),
            oracle_ext(our_proof.proof.outer_claims.2),
        ),
        inner_sumcheck: spartan_whir::InnerSumcheckProof {
            rounds: our_proof
                .proof
                .inner_sumcheck
                .rounds
                .iter()
                .map(|round| spartan_whir::QuadraticRoundPoly(round.0.map(oracle_ext)))
                .collect(),
        },
        witness_eval: oracle_ext(our_proof.proof.witness_eval),
    };
    let mut up_challenger = spartan_whir::poseidon_challenger();
    OracleProtocol::verify(&up_vk, &up_instance, &cross_proof, &mut up_challenger)
        .expect("the upstream verifier must accept our proof");

    // Cross-verification: our verifier accepts the upstream proof.
    let converted = SpartanProof {
        witness_commitment: core::array::from_fn(|i| {
            KoalaBear::from_u64(up_instance.witness_commitment[0][i].as_canonical_u64())
        }),
        proof: cirrus_spartan_whir::DirectSparseProof {
            outer_sumcheck: cirrus_spartan_whir::OuterSumcheckProof {
                rounds: up_proof
                    .outer_sumcheck
                    .rounds
                    .iter()
                    .map(|round| {
                        cirrus_spartan_whir::CubicRoundPoly(core::array::from_fn(|i| {
                            QuinticExtension::new(core::array::from_fn(|j| {
                                KoalaBear::from_u64(
                                    BasedVectorSpace::<OracleF>::as_basis_coefficients_slice(
                                        &round.0[i],
                                    )[j]
                                        .as_canonical_u64(),
                                )
                            }))
                        }))
                    })
                    .collect(),
            },
            outer_claims: (
                oracle_ext_back(up_proof.outer_claims.0),
                oracle_ext_back(up_proof.outer_claims.1),
                oracle_ext_back(up_proof.outer_claims.2),
            ),
            inner_sumcheck: cirrus_spartan_whir::InnerSumcheckProof {
                rounds: up_proof
                    .inner_sumcheck
                    .rounds
                    .iter()
                    .map(|round| {
                        cirrus_spartan_whir::QuadraticRoundPoly(core::array::from_fn(|i| {
                            oracle_ext_back(round.0[i])
                        }))
                    })
                    .collect(),
            },
            witness_eval: oracle_ext_back(up_proof.witness_eval),
            pcs_proof: convert_upstream_whir_proof(&up_proof.pcs_proof),
        },
    };
    vk.verify(&chain_public(), &converted)
        .expect("our verifier must accept the upstream proof");

    // And both honest paths verify.
    vk.verify(&chain_public(), &our_proof).unwrap();
    let mut up_challenger = spartan_whir::poseidon_challenger();
    OracleProtocol::verify(&up_vk, &up_instance, &up_proof, &mut up_challenger).unwrap();
}

fn oracle_ext_back(value: OracleEF) -> QuinticExtension {
    QuinticExtension::new(core::array::from_fn(|i| {
        KoalaBear::from_u64(
            BasedVectorSpace::<OracleF>::as_basis_coefficients_slice(&value)[i].as_canonical_u64(),
        )
    }))
}

// Keep the p3-challenger trait imports used even if future edits remove a
// call site.
#[allow(dead_code)]
fn _trait_use(mut challenger: spartan_whir::PoseidonChallenger) {
    challenger.observe(OracleF::ZERO);
    let _: OracleF = challenger.sample();
    let _: OracleEF = challenger.sample_algebra_element();
}
