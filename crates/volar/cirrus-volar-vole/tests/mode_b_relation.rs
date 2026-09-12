use cirrus_volar_vole::{
    KOALABEAR_MODULUS, KOALABEAR_QUINTIC_DEGREE, ModeBRelation, ModeBRelationError,
    PRIME_RAM_ADDRESS_BITS, PrimeFieldRamConfig, PrimeRamPermutationChallenges, PrimeRamR1cs,
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
    let materialized = cirrus_volar_vole::PrimeRamMaterialization::from_ram_witness(&ram).unwrap();
    assert_eq!(materialized.execution.len(), 2);
    assert_eq!(materialized.sorted.len(), 2);
    assert!(materialized.sorted[0].first_write);
    assert!(materialized.sorted[1].later_read);
    assert!(materialized.sorted[1].prior_latest);
    assert!(materialized.sorted[1].latest);
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

fn storage_bound_circuit(storage: StorageId, lane: LaneId, address_bits: usize) -> BCircuit {
    BCircuit {
        params: address_bits as u32 + 1,
        stmts: vec![Node::new(
            BIrStmt::StorageWrite {
                storage,
                lane,
                src: IRVarId(address_bits as u32),
                addr: (0..address_bits as u32).map(IRVarId).collect(),
            },
            (),
            None,
        )],
        pre_init: vec![],
        outputs: vec![],
    }
}

#[test]
fn relation_rejects_ram_values_outside_the_fixed_extension_abi() {
    let bad_storage = storage_bound_circuit(StorageId(1 << 16), LaneId(0), 1);
    assert!(matches!(
        ModeBRelation::from_boolar(&bad_storage),
        Err(ModeBRelationError::RamBoundsExceeded {
            field: "storage ID",
            ..
        })
    ));
    let bad_lane = storage_bound_circuit(StorageId(0), LaneId(1 << 16), 1);
    assert!(matches!(
        ModeBRelation::from_boolar(&bad_lane),
        Err(ModeBRelationError::RamBoundsExceeded {
            field: "lane ID",
            ..
        })
    ));
    let bad_address = storage_bound_circuit(StorageId(0), LaneId(0), 33);
    assert!(matches!(
        ModeBRelation::from_boolar(&bad_address),
        Err(ModeBRelationError::RamBoundsExceeded {
            field: "address width",
            ..
        })
    ));
}

#[test]
fn unified_exporter_has_one_non_overlapping_coordinate_system() {
    let circuit = storage_write_then_read();
    let relation = ModeBRelation::from_boolar(&circuit).unwrap();
    let challenges = PrimeRamPermutationChallenges {
        gamma: [1, 2, 3, 4, 5],
        eta: [6, 7, 8, 9, 10],
    };
    let export = relation
        .export_unified_r1cs(&circuit, Some(&challenges))
        .unwrap();
    let ram = export.ram.as_ref().unwrap();
    let permutation = export.permutation.as_ref().unwrap();
    assert_eq!(export.primary_offset, PrimeRamR1cs::new(2).variable_count);
    assert_eq!(permutation.variable_count, export.variable_count);
    assert!(export.public_bindings.iter().all(|binding| match binding {
        cirrus_volar_vole::PublicBinding::Input { wire, .. }
        | cirrus_volar_vole::PublicBinding::Output { wire, .. } => *wire >= export.primary_offset,
    }));
    assert!(ram.rows.iter().all(|row| {
        row.a
            .terms
            .iter()
            .chain(row.b.terms.iter())
            .chain(row.c.terms.iter())
            .all(|(variable, _)| *variable < export.variable_count)
    }));
    assert!(matches!(
        relation.export_unified_r1cs(&circuit, None),
        Err(ModeBRelationError::MissingRamPermutationChallenges)
    ));
}

#[test]
fn koalabear_lowering_normalizes_signed_and_duplicate_coefficients() {
    let circuit = storage_write_then_read();
    let relation = ModeBRelation::from_boolar(&circuit).unwrap();
    let export = relation
        .export_unified_r1cs(
            &circuit,
            Some(&PrimeRamPermutationChallenges {
                gamma: [-1, 2, 3, 4, 5],
                eta: [6, 7, 8, 9, 10],
            }),
        )
        .unwrap();
    let lowered = export.lower_koalabear().unwrap();
    assert_eq!(lowered.modulus, KOALABEAR_MODULUS);
    assert_eq!(lowered.variable_count, export.variable_count);
    assert_eq!(lowered.rows.len(), export.rows.len());
    assert!(
        lowered
            .rows
            .iter()
            .flat_map(|row| [&row.a, &row.b, &row.c])
            .all(|lc| {
                lc.constant < KOALABEAR_MODULUS
                    && lc.terms.windows(2).all(|pair| pair[0].0 < pair[1].0)
                    && lc.terms.iter().all(|(_, coefficient)| {
                        *coefficient != 0 && *coefficient < KOALABEAR_MODULUS
                    })
            })
    );
    // The relation contains `-1` coefficients, which become canonical field
    // representatives rather than host-language signed integers.
    assert!(lowered.rows.iter().any(|row| {
        [&row.a, &row.b, &row.c].into_iter().any(|lc| {
            lc.constant == KOALABEAR_MODULUS - 1
                || lc.terms.iter().any(|(_, c)| *c == KOALABEAR_MODULUS - 1)
        })
    }));
}

#[test]
fn spartan_whir_shape_uses_witness_one_public_column_order() {
    let circuit = half_adder();
    let relation = ModeBRelation::from_boolar(&circuit).unwrap();
    let shape = relation
        .export_unified_r1cs(&circuit, None)
        .unwrap()
        .lower_koalabear()
        .unwrap()
        .export_spartan_whir_shape()
        .unwrap();
    // Spartan-WHIR/Circom convention is claimed outputs, then public inputs.
    assert_eq!(shape.public_wires, vec![2, 3, 0, 1]);
    assert_eq!(shape.public_input_count, 4);
    assert_eq!(shape.witness_count, relation.witness_count - 4);
    let columns = shape
        .a
        .iter()
        .chain(shape.b.iter())
        .chain(shape.c.iter())
        .map(|entry| entry.column)
        .collect::<Vec<_>>();
    assert!(
        columns
            .iter()
            .all(|column| *column < shape.witness_count + 1 + shape.public_input_count)
    );
    assert!(columns.contains(&shape.witness_count)); // constant-one column
    assert!(columns.contains(&(shape.witness_count + 1))); // first public output
}

#[test]
fn unified_exporter_rejects_ram_challenges_without_storage() {
    let circuit = half_adder();
    let relation = ModeBRelation::from_boolar(&circuit).unwrap();
    let challenges = PrimeRamPermutationChallenges {
        gamma: [0; 5],
        eta: [0; 5],
    };
    assert!(matches!(
        relation.export_unified_r1cs(&circuit, Some(&challenges)),
        Err(ModeBRelationError::UnexpectedRamPermutationChallenges)
    ));
    let export = relation.export_unified_r1cs(&circuit, None).unwrap();
    assert_eq!(export.primary_offset, 0);
    assert_eq!(export.variable_count, relation.witness_count);
}

#[test]
fn prime_ram_r1cs_binds_execution_records_to_boolar_wires() {
    let circuit = storage_write_then_read();
    let static_layout = PrimeRamR1cs::new(2);
    let layout = PrimeRamR1cs::for_boolar(&circuit, static_layout.variable_count).unwrap();
    assert!(layout.variable_count >= static_layout.variable_count + 4);
    assert!(matches!(
        PrimeRamR1cs::for_boolar(&circuit, 0),
        Err(ModeBRelationError::RamVariableLayoutOverlap)
    ));
}

#[test]
fn prime_ram_r1cs_has_a_deterministic_static_scan_layout() {
    let a = PrimeRamR1cs::new(2);
    let b = PrimeRamR1cs::new(2);
    assert_eq!(a, b);
    assert_eq!(a.execution.len(), 2);
    assert_eq!(a.sorted.len(), 2);
    assert!(!a.rows.is_empty());
    assert!(a.variable_count > 2 * (16 + 16 + 32 + 32 + 2));
    assert_eq!(a.sorted[0].cell_bit_equal.len(), 0);
    assert_eq!(a.sorted[1].cell_bit_equal.len(), 64);
    assert_eq!(a.sorted[1].cell_first_difference.len(), 64);
    let permutation = a.permutation_rows(&PrimeRamPermutationChallenges {
        gamma: [1, 2, 3, 4, 5],
        eta: [6, 7, 8, 9, 10],
    });
    assert_eq!(permutation.z.len(), 3);
    assert!(permutation.variable_count > a.variable_count);
    assert!(!permutation.rows.is_empty());
    // The initial scan row has the required fixed zero predecessor.
    assert!(
        a.rows
            .iter()
            .any(|row| { row.c.terms == vec![(a.sorted[0].same_cell, 1)] && row.c.constant == 0 })
    );
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
