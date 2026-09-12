use cirrus_volar_vole::{
    KOALABEAR_MODULUS, KOALABEAR_QUINTIC_DEGREE, ModeBRelation, ModeBRelationError,
    PRIME_RAM_ADDRESS_BITS, PrimeFieldRamConfig,
};
use volar_ir::{
    boolar::{BIrStmt, LaneId},
    circuit::BCircuit,
    ir::IRVarId,
};
use volar_ir_common::{Node, StorageId};

fn half_adder() -> BCircuit {
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
fn lowers_and_evaluates_boolar_over_field_independent_rows() {
    let relation = ModeBRelation::from_boolar(&half_adder()).unwrap();
    assert_eq!(relation.wire_count, 4);
    assert_eq!(relation.witness_count, 5); // XOR product helper
    relation
        .evaluate_bool(&[true, true, false, true], &[true, true], &[false, true])
        .unwrap();
}

#[test]
fn relation_rejects_invalid_gate_and_public_output() {
    let relation = ModeBRelation::from_boolar(&half_adder()).unwrap();
    assert!(matches!(
        relation.evaluate_bool(
            &[true, false, false, false],
            &[true, false],
            &[false, false]
        ),
        Err(ModeBRelationError::UnsatisfiedRow { .. })
    ));
    assert_eq!(
        relation.evaluate_bool(&[true, false, true, false], &[true, false], &[false, false]),
        Err(ModeBRelationError::PublicMismatch { binding: 2 })
    );
}

fn storage_write_then_read() -> BCircuit {
    BCircuit {
        params: 2,
        stmts: vec![
            Node::new(
                BIrStmt::StorageWrite {
                    storage: StorageId(9),
                    lane: LaneId(2),
                    src: IRVarId(1),
                    addr: vec![IRVarId(0)],
                },
                (),
                None,
            ),
            Node::new(
                BIrStmt::StorageRead {
                    storage: StorageId(9),
                    lane: LaneId(2),
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
fn ram_relation_accepts_canonical_witness_and_rejects_bad_read() {
    let circuit = storage_write_then_read();
    let relation = ModeBRelation::from_boolar(&circuit).unwrap();
    let witness = [true, true, false, true];
    let ram = cirrus_volar_vole::RamWitness::from_boolar(&circuit, &witness).unwrap();
    relation
        .evaluate_bool_with_ram(&circuit, &witness, &[true, true], &[true], Some(&ram))
        .unwrap();
    let mut bad = ram.clone();
    bad.address_sorted[1].value = false;
    assert_eq!(
        relation.evaluate_bool_with_ram(&circuit, &witness, &[true, true], &[true], Some(&bad)),
        Err(ModeBRelationError::RamNotPermutation)
    );
    let mut invalid_read = ram.clone();
    invalid_read.execution[1].value = false;
    invalid_read.address_sorted[1].value = false;
    assert_eq!(
        relation.evaluate_bool_with_ram(
            &circuit,
            &witness,
            &[true, true],
            &[true],
            Some(&invalid_read)
        ),
        Err(ModeBRelationError::RamExecutionMismatch)
    );
}

#[test]
fn koalabear_quintic_ram_format_has_a_full_32_bit_address_space() {
    let config = PrimeFieldRamConfig::default();
    assert!(config.validate());
    assert_eq!(config.base_modulus, KOALABEAR_MODULUS);
    assert_eq!(config.extension_degree, KOALABEAR_QUINTIC_DEGREE);
    assert_eq!(config.address_bits, PRIME_RAM_ADDRESS_BITS);
    // A quintic extension has ~155 bits; this format allocates 16+16+32+32+1
    // key bits, leaving ample headroom for the compression challenge/value.
    assert_eq!(
        config.storage_bits + config.lane_bits + config.address_bits + config.time_bits + 1,
        97
    );
    let mut invalid = config;
    invalid.address_bits = 31;
    assert!(!invalid.validate());
}

#[test]
fn canonical_serializer_is_stable_and_statement_bound() {
    let a = ModeBRelation::from_boolar(&half_adder()).unwrap();
    let instance =
        cirrus_volar_vole::ModeBPublicInstance::new(&a, vec![true, false], vec![true, false]);
    assert_eq!(a.canonical_bytes(), a.canonical_bytes());
    assert_eq!(instance.canonical_bytes(), instance.canonical_bytes());
    assert_ne!(
        a.canonical_bytes(),
        ModeBRelation::from_boolar(&storage_write_then_read())
            .unwrap()
            .canonical_bytes()
    );
}

#[test]
fn circuit_id_binds_more_than_the_gate_tags() {
    let a = half_adder();
    let mut b = half_adder();
    b.outputs.swap(0, 1);
    assert_ne!(
        ModeBRelation::from_boolar(&a).unwrap().circuit_id,
        ModeBRelation::from_boolar(&b).unwrap().circuit_id
    );
}
