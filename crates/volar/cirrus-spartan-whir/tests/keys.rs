//! Deterministic tests for the setup/prove/verify key API, the composed
//! security budget, the domain-separator encoding, and the challenge-slot
//! schedule plumbing.

use cirrus_spartan_whir::security::{
    ComponentSecurity, derive_direct_component_security, spartan_algebraic_error_terms_no_zk,
};
use cirrus_spartan_whir::whir::params::SecurityAssumption;
use cirrus_spartan_whir::whir::spartan::{SecurityProfile, WhirParams};
use cirrus_spartan_whir::{
    ChallengeSchedule, DirectSparseError, FieldElement, KoalaBear, PoseidonTranscript,
    QuinticExtension, R1csShape, SparseMatEntry, SparseMatrix, SpartanKeyError, SpartanProof,
    setup, spartan_domain_separator,
};

fn kb(value: u32) -> KoalaBear {
    KoalaBear::from_u64(u64::from(value))
}

fn entry(row: usize, col: usize, val: u32) -> SparseMatEntry<KoalaBear> {
    SparseMatEntry {
        row,
        col,
        val: KoalaBear::from_u32(val),
    }
}

/// The chain shape from the sumcheck tests: `w1 = w0^2`, `w2 = w1 * x0`,
/// `w3 = w2 * x1`, one constant row; 4 constraints, 4 witness columns,
/// `num_io` public values.
fn chain_shape_with_io(num_io: usize) -> R1csShape<KoalaBear> {
    let num_cols = 4 + 1 + num_io;
    R1csShape {
        num_cons: 4,
        num_vars: 4,
        num_io,
        a: SparseMatrix {
            num_rows: 4,
            num_cols,
            entries: vec![
                entry(0, 0, 1),
                entry(1, 1, 1),
                entry(2, 2, 1),
                entry(3, 3, 1),
            ],
        },
        b: SparseMatrix {
            num_rows: 4,
            num_cols,
            entries: vec![
                entry(0, 0, 1),
                entry(1, 5, 1),
                entry(2, 6, 1),
                entry(3, 4, 1),
            ],
        },
        c: SparseMatrix {
            num_rows: 4,
            num_cols,
            entries: vec![
                entry(0, 1, 1),
                entry(1, 2, 1),
                entry(2, 3, 1),
                entry(3, 4, 60),
            ],
        },
    }
}

fn chain_shape() -> R1csShape<KoalaBear> {
    chain_shape_with_io(2)
}

fn chain_witness() -> Vec<KoalaBear> {
    vec![kb(2), kb(4), kb(12), kb(60)]
}

fn chain_public() -> Vec<KoalaBear> {
    vec![kb(3), kb(5)]
}

fn test_profile() -> SecurityProfile {
    SecurityProfile {
        security_level_bits: 80,
        merkle_security_bits: 80,
        soundness: SecurityAssumption::CapacityBound,
        whir: WhirParams {
            pow_bits: 0,
            folding_factor: 1,
            starting_log_inv_rate: 6,
            rs_domain_initial_reduction_factor: 1,
            folding_schedule: None,
            round_log_inv_rates: Vec::new(),
        },
    }
}

#[test]
fn setup_freezes_shape_profile_security_and_separator() {
    let (pk, vk) = setup(&chain_shape(), &test_profile()).unwrap();

    assert_eq!(pk.num_vars_unpadded(), 4);
    assert_eq!(pk.num_io(), 2);
    assert_eq!(vk.shape_canonical().num_cons, 4);
    assert_eq!(vk.shape_canonical().num_vars, 4);
    assert_eq!(vk.pcs_config().num_variables, 2);

    // Zero intermediate WHIR rounds for two variables with folding factor 1:
    // the algebraic budget is 4*2 + 2*3 + 2 = 16, one WHIR argument and one
    // commitment event give slack 2 on both components.
    assert_eq!(spartan_algebraic_error_terms_no_zk(2, 3).unwrap(), 16);
    let (component, budget) = derive_direct_component_security(80, 80, 0, 2, 3).unwrap();
    assert_eq!(budget.requested_bits, 80);
    assert_eq!(budget.whir_slack_bits, 2);
    assert_eq!(budget.merkle_slack_bits, 2);
    assert_eq!(budget.algebraic_error_terms, 16);
    assert_eq!(budget.commitment_binding_events, 1);
    assert_eq!(budget.attainable_bits, 121);
    assert_eq!(
        component,
        ComponentSecurity {
            security_level_bits: 82,
            merkle_security_bits: 82,
        }
    );
    assert_eq!(pk.component_security(), component);
    assert_eq!(vk.component_security(), component);

    // The domain separator bytes are the hand-built upstream encoding.
    let expected = {
        let mut bytes = b"spartan-whir-no-zk-v0".to_vec();
        bytes.push(0); // DirectSparse
        bytes.extend_from_slice(&4u64.to_le_bytes()); // num_cons
        bytes.extend_from_slice(&4u64.to_le_bytes()); // num_vars
        bytes.extend_from_slice(&2u64.to_le_bytes()); // num_io
        bytes.extend_from_slice(&80u32.to_le_bytes());
        bytes.extend_from_slice(&80u32.to_le_bytes());
        bytes.push(2); // CapacityBound
        bytes.extend_from_slice(&0u32.to_le_bytes()); // pow_bits
        bytes.extend_from_slice(&1u64.to_le_bytes()); // folding_factor
        bytes.extend_from_slice(&6u64.to_le_bytes()); // starting_log_inv_rate
        bytes.extend_from_slice(&1u64.to_le_bytes()); // rs_domain_initial_reduction_factor
        bytes
    };
    assert_eq!(pk.domain_separator(), expected.as_slice());
    assert_eq!(
        spartan_domain_separator(vk.shape_canonical(), &test_profile()),
        expected
    );

    vk.validate_key().unwrap();
}

#[test]
fn setup_rejects_out_of_range_and_unattainable_security() {
    let shape = chain_shape();

    let mut below = test_profile();
    below.security_level_bits = 79;
    assert!(matches!(
        setup(&shape, &below),
        Err(SpartanKeyError::Profile(
            cirrus_spartan_whir::whir::spartan::WhirConfigBuildError::SecurityBelowMinimum
        ))
    ));

    let mut above = test_profile();
    above.security_level_bits = 124;
    assert!(matches!(
        setup(&shape, &above),
        Err(SpartanKeyError::Profile(
            cirrus_spartan_whir::whir::spartan::WhirConfigBuildError::SecurityAboveMaximum
        ))
    ));

    // The attainable level for this tiny shape is 121 bits (WHIR argument
    // slack), so 121 passes and 122 fails.
    let mut boundary = test_profile();
    boundary.security_level_bits = 121;
    boundary.merkle_security_bits = 121;
    setup(&shape, &boundary).expect("121 bits is attainable");
    boundary.security_level_bits = 122;
    boundary.merkle_security_bits = 122;
    assert!(matches!(
        setup(&shape, &boundary),
        Err(SpartanKeyError::Security(
            cirrus_spartan_whir::security::SecurityBudgetError::ComposedSecurityUnavailable {
                requested_bits: 122,
                attainable_bits: 121,
                ..
            }
        ))
    ));
}

#[test]
fn end_to_end_prove_verify_through_the_key_api() {
    let (pk, vk) = setup(&chain_shape(), &test_profile()).unwrap();
    let proof = pk.prove(&chain_witness(), &chain_public()).unwrap();
    vk.verify(&chain_public(), &proof).unwrap();

    // Wrong public values are rejected.
    let mut wrong = chain_public();
    wrong[0] = kb(4);
    assert!(vk.verify(&wrong, &proof).is_err());

    // A tampered witness commitment is rejected.
    let mut tampered: SpartanProof = proof.clone();
    tampered.witness_commitment[0] = tampered.witness_commitment[0] + KoalaBear::ONE;
    assert!(vk.verify(&chain_public(), &tampered).is_err());

    // A tampered round polynomial is rejected.
    let mut tampered = proof.clone();
    let evals = &mut tampered.proof.outer_sumcheck.rounds[0].0;
    evals[2] = evals[2] + QuinticExtension::ONE;
    assert!(vk.verify(&chain_public(), &tampered).is_err());

    // A tampered witness evaluation is rejected.
    let mut tampered = proof.clone();
    tampered.proof.witness_eval = tampered.proof.witness_eval + QuinticExtension::ONE;
    assert!(vk.verify(&chain_public(), &tampered).is_err());
}

/// A synthetic two-slot schedule: derives two base-field challenges from
/// the post-commitment transcript state.
struct TwoSlotSchedule {
    recorded: Vec<KoalaBear>,
}

impl ChallengeSchedule for TwoSlotSchedule {
    fn challenge_slots(&self) -> usize {
        2
    }

    fn derive_challenges(&mut self, transcript: &mut PoseidonTranscript) -> Vec<KoalaBear> {
        let derived = vec![transcript.sample_base(), transcript.sample_base()];
        self.recorded = derived.clone();
        derived
    }
}

#[test]
fn challenge_slots_are_transcript_derived_and_checked_on_both_sides() {
    // The two challenge slots are unconstrained extra public inputs; binding
    // them to committed columns is the Mode-B storage milestone's job.
    let shape = chain_shape_with_io(4);
    let (pk, vk) = setup(&shape, &test_profile()).unwrap();

    // Trial run: learn the transcript-derived slot values. The driver
    // rejects the zero tail but the schedule records the derived values.
    let mut trial = TwoSlotSchedule {
        recorded: Vec::new(),
    };
    let zero_tail_public = vec![kb(3), kb(5), KoalaBear::ZERO, KoalaBear::ZERO];
    let result = pk.prove_with_schedule(&chain_witness(), &zero_tail_public, &mut trial);
    assert!(matches!(
        result,
        Err(SpartanKeyError::Protocol(
            DirectSparseError::ChallengeMismatch
        ))
    ));
    assert_eq!(trial.recorded.len(), 2);

    // Real run with the transcript-derived tail.
    let mut public = chain_public();
    public.extend_from_slice(&trial.recorded);
    let mut prover_schedule = TwoSlotSchedule {
        recorded: Vec::new(),
    };
    let proof = pk
        .prove_with_schedule(&chain_witness(), &public, &mut prover_schedule)
        .unwrap();
    assert_eq!(prover_schedule.recorded, trial.recorded);

    // The verifier recomputes the slots independently.
    let mut verifier_schedule = TwoSlotSchedule {
        recorded: Vec::new(),
    };
    vk.verify_with_schedule(&public, &proof, &mut verifier_schedule)
        .unwrap();
    assert_eq!(verifier_schedule.recorded, trial.recorded);

    // A wrong supplied tail is rejected on the verifier side.
    let mut wrong_public = public.clone();
    wrong_public[2] = wrong_public[2] + KoalaBear::ONE;
    let mut verifier_schedule = TwoSlotSchedule {
        recorded: Vec::new(),
    };
    assert!(matches!(
        vk.verify_with_schedule(&wrong_public, &proof, &mut verifier_schedule),
        Err(SpartanKeyError::Protocol(
            DirectSparseError::ChallengeMismatch
        ))
    ));

    // A schedule deriving a different number of slots is rejected.
    struct OneSlotSchedule;
    impl ChallengeSchedule for OneSlotSchedule {
        fn challenge_slots(&self) -> usize {
            1
        }
        fn derive_challenges(&mut self, transcript: &mut PoseidonTranscript) -> Vec<KoalaBear> {
            vec![transcript.sample_base()]
        }
    }
    let mut wrong_arity = OneSlotSchedule;
    assert!(
        pk.prove_with_schedule(&chain_witness(), &public, &mut wrong_arity)
            .is_err()
    );

    // The no-schedule entry points reject the slot-bearing shape's tail
    // interpretation: with zero slots the whole vector is observed up front,
    // so the slot-bearing proof does not verify without its schedule.
    assert!(vk.verify(&public, &proof).is_err());
}

#[cfg(feature = "oracle-tests")]
mod oracle {
    include!("keys/oracle.rs");
}
