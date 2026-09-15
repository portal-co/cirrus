use cirrus_volar_vole::{TraceAuditError, commit_boolar_trace};
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

#[test]
fn exhaustive_authenticated_trace_accepts_a_valid_circuit_evaluation() {
    let circuit = half_adder();
    let audit = commit_boolar_trace(&circuit, &[true, true, false, true]).unwrap();
    audit
        .verify(&circuit, &[true, true], &[false, true])
        .unwrap();
}

#[test]
fn rejects_an_authenticated_but_invalid_gate_trace() {
    let circuit = half_adder();
    // This root and all paths are honest for these values, but the XOR output
    // is false when the inputs are true and false.
    let audit = commit_boolar_trace(&circuit, &[true, false, false, false]).unwrap();
    assert_eq!(
        audit.verify(&circuit, &[true, false], &[false, false]),
        Err(TraceAuditError::GateMismatch { wire: 2 })
    );
}

#[test]
fn rejects_a_tampered_opening() {
    let circuit = half_adder();
    let mut audit = commit_boolar_trace(&circuit, &[true, false, true, false]).unwrap();
    audit.openings[2].value = false;
    assert_eq!(
        audit.verify(&circuit, &[true, false], &[true, false]),
        Err(TraceAuditError::InvalidAuthentication { wire: 2 })
    );
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

#[test]
fn storage_trace_uses_a_full_ram_permutation_and_checks_latest_write() {
    let circuit = storage_write_then_read();
    let audit = commit_boolar_trace(&circuit, &[true, true, false, true]).unwrap();
    audit.verify(&circuit, &[true, true], &[true]).unwrap();
    let memory = audit.memory.as_ref().unwrap();
    assert_eq!(memory.execution.len(), 2);
    assert_eq!(memory.address_sorted.len(), 2);
}

#[test]
fn storage_trace_rejects_a_non_permutation_address_table() {
    let circuit = storage_write_then_read();
    let mut audit = commit_boolar_trace(&circuit, &[true, true, false, true]).unwrap();
    audit.memory.as_mut().unwrap().address_sorted.swap(0, 1);
    assert_eq!(
        audit.verify(&circuit, &[true, true], &[true]),
        Err(TraceAuditError::MemoryNotPermutation)
    );
}

#[test]
fn storage_trace_rejects_a_read_that_does_not_match_the_prior_write() {
    let circuit = storage_write_then_read();
    let audit = commit_boolar_trace(&circuit, &[true, true, false, false]).unwrap();
    assert_eq!(
        audit.verify(&circuit, &[true, true], &[false]),
        Err(TraceAuditError::InvalidMemoryRead { time: 1 })
    );
}

#[test]
fn rejects_a_wrong_public_output() {
    let circuit = half_adder();
    let audit = commit_boolar_trace(&circuit, &[true, false, true, false]).unwrap();
    assert_eq!(
        audit.verify(&circuit, &[true, false], &[false, false]),
        Err(TraceAuditError::OutputMismatch { output: 0 })
    );
}
