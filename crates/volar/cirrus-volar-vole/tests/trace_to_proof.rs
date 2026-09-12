use cirrus_volar_vole::{
    ModeBRelationError, PrimeRamPermutationChallenges, RamChallengeInput, TraceToProofError,
    TraceToProofInput, build_trace_proof_artifacts, commit_boolar_trace,
};
use volar_ir::{
    boolar::{BIrStmt, LaneId},
    circuit::BCircuit,
    ir::IRVarId,
};
use volar_ir_common::{Node, StorageId};

fn half_adder() -> BCircuit {
    // Inputs: 0=a, 1=b. Statements: 2=sum, 3=carry.
    BCircuit {
        params: 2,
        stmts: vec![
            Node::new(BIrStmt::Xor(IRVarId(0), IRVarId(1)), (), None),
            Node::new(BIrStmt::And(IRVarId(0), IRVarId(1)), (), None),
        ],
        pre_init: vec![],
        outputs: vec![IRVarId(2), IRVarId(3)],
    }
}

fn storage_write_then_read() -> BCircuit {
    // Inputs: 0=one address bit, 1=value. Statement 2 is the write's
    // mandated false result; statement 3 is the subsequently read value.
    BCircuit {
        params: 2,
        stmts: vec![
            Node::new(
                BIrStmt::StorageWrite {
                    storage: StorageId(7),
                    lane: LaneId(0),
                    src: IRVarId(1),
                    addr: vec![IRVarId(0)],
                },
                (),
                None,
            ),
            Node::new(
                BIrStmt::StorageRead {
                    storage: StorageId(7),
                    lane: LaneId(0),
                    addr: vec![IRVarId(0)],
                },
                (),
                None,
            ),
        ],
        pre_init: vec![],
        outputs: vec![IRVarId(3)],
    }
}

fn challenges() -> PrimeRamPermutationChallenges {
    PrimeRamPermutationChallenges {
        gamma: [11, 12, 13, 14, 15],
        eta: [16, 17, 18, 19, 20],
    }
}

#[test]
fn storage_free_trace_builds_and_revalidates_all_artifacts() {
    let circuit = half_adder();
    let input = TraceToProofInput {
        audit: commit_boolar_trace(&circuit, &[true, false, true, false]).unwrap(),
        public_inputs: vec![true, false],
        claimed_outputs: vec![true, false],
    };
    let artifacts =
        build_trace_proof_artifacts(&circuit, &input, RamChallengeInput::NoStorage).unwrap();

    assert_eq!(artifacts.trace_values, vec![true, false, true, false]);
    assert_eq!(artifacts.ram_witness, None);
    assert_eq!(artifacts.relation.circuit_id, artifacts.circuit_id);
    assert_eq!(artifacts.public_instance.circuit_id, artifacts.circuit_id);
    assert_eq!(artifacts.unified.circuit_id, artifacts.circuit_id);
    assert_eq!(
        artifacts.witness.values.len(),
        artifacts.unified.variable_count
    );
    artifacts.validate_for_circuit(&circuit).unwrap();
    assert_eq!(
        artifacts.public_instance.canonical_bytes(),
        artifacts.public_instance.canonical_bytes()
    );
    assert_eq!(
        artifacts.relation.canonical_bytes(),
        artifacts.relation.canonical_bytes()
    );
}

#[test]
fn storage_trace_converts_mode_a_memory_to_mode_b_ram_witness() {
    let circuit = storage_write_then_read();
    let input = TraceToProofInput {
        audit: commit_boolar_trace(&circuit, &[true, true, false, true]).unwrap(),
        public_inputs: vec![true, true],
        claimed_outputs: vec![true],
    };
    let artifacts = build_trace_proof_artifacts(
        &circuit,
        &input,
        RamChallengeInput::DifferentialOracle(challenges()),
    )
    .unwrap();

    let ram = artifacts.ram_witness.as_ref().unwrap();
    assert_eq!(ram.execution.len(), 2);
    assert_eq!(ram.address_sorted.len(), 2);
    assert_eq!(ram.execution[0].address.len(), 32);
    assert!(artifacts.unified.ram.is_some());
    artifacts.validate_for_circuit(&circuit).unwrap();
}

#[test]
fn invalid_trace_is_rejected_before_relation_construction() {
    let circuit = half_adder();
    let input = TraceToProofInput {
        audit: commit_boolar_trace(&circuit, &[true, true, false, true]).unwrap(),
        public_inputs: vec![true, true],
        claimed_outputs: vec![true, true], // sum should be false
    };
    assert!(matches!(
        build_trace_proof_artifacts(&circuit, &input, RamChallengeInput::NoStorage),
        Err(TraceToProofError::Trace(
            cirrus_volar_vole::TraceAuditError::OutputMismatch { output: 0 }
        ))
    ));
}

#[test]
fn challenge_schedule_is_checked_against_storage_presence() {
    let circuit = half_adder();
    let input = TraceToProofInput {
        audit: commit_boolar_trace(&circuit, &[true, false, true, false]).unwrap(),
        public_inputs: vec![true, false],
        claimed_outputs: vec![true, false],
    };
    assert_eq!(
        build_trace_proof_artifacts(
            &circuit,
            &input,
            RamChallengeInput::DifferentialOracle(challenges()),
        ),
        Err(TraceToProofError::UnexpectedRamChallenges)
    );

    let circuit = storage_write_then_read();
    let input = TraceToProofInput {
        audit: commit_boolar_trace(&circuit, &[true, true, false, true]).unwrap(),
        public_inputs: vec![true, true],
        claimed_outputs: vec![true],
    };
    assert_eq!(
        build_trace_proof_artifacts(&circuit, &input, RamChallengeInput::NoStorage),
        Err(TraceToProofError::MissingRamChallenges)
    );
}

#[test]
fn artifact_context_validation_rejects_mixed_relations() {
    let circuit = half_adder();
    let input = TraceToProofInput {
        audit: commit_boolar_trace(&circuit, &[true, false, true, false]).unwrap(),
        public_inputs: vec![true, false],
        claimed_outputs: vec![true, false],
    };
    let mut artifacts =
        build_trace_proof_artifacts(&circuit, &input, RamChallengeInput::NoStorage).unwrap();
    artifacts.relation =
        cirrus_volar_vole::ModeBRelation::from_boolar(&storage_write_then_read()).unwrap();
    assert_eq!(
        artifacts.validate_for_circuit(&circuit),
        Err(TraceToProofError::CircuitIdMismatch)
    );
}

#[test]
fn artifact_context_validation_rejects_tampered_trace_values() {
    let circuit = half_adder();
    let input = TraceToProofInput {
        audit: commit_boolar_trace(&circuit, &[true, false, true, false]).unwrap(),
        public_inputs: vec![true, false],
        claimed_outputs: vec![true, false],
    };
    let mut artifacts =
        build_trace_proof_artifacts(&circuit, &input, RamChallengeInput::NoStorage).unwrap();
    artifacts.trace_values[2] = false;
    let error = artifacts.validate_for_circuit(&circuit).unwrap_err();
    panic_expected_trace_validation_error(error);
}

#[track_caller]
fn panic_expected_trace_validation_error(error: TraceToProofError) {
    assert!(
        matches!(
            error,
            TraceToProofError::ModeB(
                ModeBRelationError::UnsatisfiedUnifiedRow { .. }
                    | ModeBRelationError::InconsistentUnifiedWitness { .. }
                    | ModeBRelationError::UnsatisfiedRow { .. }
                    | ModeBRelationError::PublicMismatch { .. }
                    | ModeBRelationError::RamExecutionMismatch
            )
        ),
        "unexpected validation error: {error:?}"
    );
}
