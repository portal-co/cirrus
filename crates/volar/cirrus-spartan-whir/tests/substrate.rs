use cirrus_spartan_whir::{
    EqPolynomial, FieldElement, KoalaBear, R1csError, R1csShape, R1csWitness, SparseMatEntry,
    SparseMatrix, evaluate_mle_table,
};
#[cfg(feature = "oracle-tests")]
use cirrus_spartan_whir::{QUINTIC_DEGREE, QuinticExtension};

fn kb(value: u32) -> KoalaBear {
    KoalaBear::from_u64(u64::from(value))
}

#[cfg(feature = "oracle-tests")]
fn ext(seed: u32) -> QuinticExtension {
    QuinticExtension::from_canonical_coefficients(core::array::from_fn(|index| {
        seed.wrapping_mul((index as u32) + 3)
            .wrapping_add(17 + index as u32)
    }))
    .unwrap()
}

fn entry<F: FieldElement>(row: usize, col: usize, val: u32) -> SparseMatEntry<F> {
    SparseMatEntry {
        row,
        col,
        val: F::from_u32(val),
    }
}

fn shape<F: FieldElement>() -> R1csShape<F> {
    R1csShape {
        num_cons: 3,
        num_vars: 2,
        num_io: 2,
        a: SparseMatrix {
            num_rows: 3,
            num_cols: 5,
            entries: vec![
                entry(0, 0, 1),
                entry(0, 2, 1),
                entry(1, 1, 1),
                entry(1, 2, 1),
                entry(2, 0, 1),
                entry(2, 1, 1),
            ],
        },
        b: SparseMatrix {
            num_rows: 3,
            num_cols: 5,
            entries: vec![entry(0, 2, 1), entry(1, 2, 1), entry(2, 4, 1)],
        },
        c: SparseMatrix {
            num_rows: 3,
            num_cols: 5,
            entries: vec![entry(0, 3, 1), entry(1, 4, 1), entry(2, 2, 35)],
        },
    }
}

#[test]
fn equality_table_and_mle_follow_boolean_folding_order() {
    let point = vec![kb(2), kb(3), kb(4)];
    let evals = EqPolynomial::evals_from_point(&point);
    assert_eq!(evals.len(), 8);
    assert_eq!(
        evals.iter().copied().fold(KoalaBear::ZERO, |a, b| a + b),
        KoalaBear::ONE
    );
    for (index, value) in evals.iter().enumerate() {
        let mut expected = KoalaBear::ONE;
        for bit in 0..point.len() {
            let coordinate = point[point.len() - 1 - bit];
            let selected = (index >> bit) & 1 == 1;
            expected = expected
                * if selected {
                    coordinate
                } else {
                    KoalaBear::ONE - coordinate
                };
        }
        assert_eq!(*value, expected, "wrong eq value at index {index}");
    }

    let table = [1, 2, 3, 4, 5, 6, 7, 8].map(kb);
    let mut expected = KoalaBear::ZERO;
    for (value, eq) in table.iter().zip(&evals) {
        expected = expected + *value * *eq;
    }
    assert_eq!(evaluate_mle_table(&table, &point).unwrap(), expected);
    assert!(evaluate_mle_table(&table[..7], &point).is_err());
    assert!(evaluate_mle_table(&table, &point[..2]).is_err());
}

#[test]
fn r1cs_shape_multiplies_pads_and_validates_satisfaction() {
    let shape = shape::<KoalaBear>();
    shape.validate().unwrap();
    let witness = R1csWitness {
        w: vec![kb(3), kb(4)],
    };
    let public = vec![kb(4), kb(5)];
    shape.validate_satisfaction(&witness, &public).unwrap();

    let z = vec![kb(3), kb(4), kb(1), kb(4), kb(5)];
    let (az, bz, cz) = shape.multiply_vec(&z).unwrap();
    assert_eq!(az, vec![kb(4), kb(5), kb(7)]);
    assert_eq!(bz, vec![kb(1), kb(1), kb(5)]);
    assert_eq!(cz, vec![kb(4), kb(5), kb(35)]);
    assert_eq!(
        shape.witness_to_mle(&witness.w).unwrap(),
        vec![kb(3), kb(4)]
    );

    let padded = shape.pad_regular().unwrap();
    assert_eq!(padded.num_cons, 4);
    assert_eq!(padded.num_vars, 4);
    assert_eq!(padded.num_io, 2);
    assert_eq!(padded.a.num_cols, 7);
    assert!(padded.a.entries.iter().all(|entry| entry.col < 7));
    assert!(
        padded
            .a
            .entries
            .iter()
            .any(|entry| entry.row == 0 && entry.col == 4)
    ); // constant-one column shifted

    let bad_public = vec![kb(4), kb(6)];
    assert_eq!(
        shape.validate_satisfaction(&witness, &bad_public),
        Err(R1csError::UnsatisfiedConstraint { row: 1 })
    );
    assert!(matches!(
        shape.validate_satisfaction(&witness, &public[..1]),
        Err(R1csError::InvalidWitnessLength { .. })
    ));

    let mut invalid = shape.clone();
    invalid.a.entries[0].col = invalid.a.num_cols;
    assert_eq!(invalid.validate(), Err(R1csError::InvalidShape));
}

#[cfg(feature = "oracle-tests")]
fn oracle_ext(value: QuinticExtension) -> spartan_whir::engine::QuinticExtension {
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
    use spartan_whir::engine::{F as OracleF, QuinticExtension as OracleEF};

    let coefficients = value.canonical_coefficients();
    OracleEF::from_basis_coefficients_fn(|index| OracleF::from_u32(coefficients[index]))
}

#[cfg(feature = "oracle-tests")]
fn assert_ext_eq(value: QuinticExtension, oracle: spartan_whir::engine::QuinticExtension) {
    use p3_field::{BasedVectorSpace, PrimeField32};
    use spartan_whir::engine::{F as OracleF, QuinticExtension as OracleEF};

    let ours = value.canonical_coefficients();
    let oracle = <OracleEF as BasedVectorSpace<OracleF>>::as_basis_coefficients_slice(&oracle);
    for index in 0..QUINTIC_DEGREE {
        assert_eq!(ours[index], oracle[index].as_canonical_u32());
    }
}

#[cfg(feature = "oracle-tests")]
#[test]
fn polynomial_tables_match_upstream_oracle() {
    use p3_field::{PrimeCharacteristicRing, PrimeField32};
    use spartan_whir::engine::F as OracleF;

    for seed in 1..24 {
        let point = (0..4)
            .map(|index| kb(seed * 17 + index as u32 * 31))
            .collect::<Vec<_>>();
        let oracle_point = point
            .iter()
            .map(|value| OracleF::from_u32(value.canonical()))
            .collect::<Vec<_>>();
        let ours = EqPolynomial::evals_from_point(&point);
        let oracle = spartan_whir::EqPolynomial::evals_from_point(&oracle_point);
        for (ours, oracle) in ours.iter().zip(oracle) {
            assert_eq!(ours.canonical(), oracle.as_canonical_u32());
        }

        let table = (0..16)
            .map(|index| kb(seed * 43 + index as u32 * 11))
            .collect::<Vec<_>>();
        let oracle_table = table
            .iter()
            .map(|value| OracleF::from_u32(value.canonical()))
            .collect::<Vec<_>>();
        let ours_eval = evaluate_mle_table(&table, &point).unwrap();
        let oracle_eval = spartan_whir::evaluate_mle_table(&oracle_table, &oracle_point).unwrap();
        assert_eq!(ours_eval.canonical(), oracle_eval.as_canonical_u32());
    }
}

#[cfg(feature = "oracle-tests")]
#[test]
fn extension_polynomial_tables_match_upstream_oracle() {
    for seed in 1..12 {
        let point = (0..3)
            .map(|index| ext(seed * 19 + index as u32 * 7))
            .collect::<Vec<_>>();
        let oracle_point = point.iter().copied().map(oracle_ext).collect::<Vec<_>>();
        let ours = EqPolynomial::evals_from_point(&point);
        let oracle = spartan_whir::EqPolynomial::evals_from_point(&oracle_point);
        for (ours, oracle) in ours.iter().copied().zip(oracle) {
            assert_ext_eq(ours, oracle);
        }

        let table = (0..8)
            .map(|index| ext(seed * 29 + index as u32 * 13))
            .collect::<Vec<_>>();
        let oracle_table = table.iter().copied().map(oracle_ext).collect::<Vec<_>>();
        assert_ext_eq(
            evaluate_mle_table(&table, &point).unwrap(),
            spartan_whir::evaluate_mle_table(&oracle_table, &oracle_point).unwrap(),
        );
    }
}

#[cfg(feature = "oracle-tests")]
#[test]
fn r1cs_substrate_matches_upstream_oracle() {
    use p3_field::{PrimeCharacteristicRing, PrimeField32};
    use spartan_whir::engine::F as OracleF;

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
                    val: OracleF::from_u32(entry.val.canonical()),
                })
                .collect(),
        }
    }

    let ours = shape::<KoalaBear>();
    let oracle = spartan_whir::R1csShape {
        num_cons: ours.num_cons,
        num_vars: ours.num_vars,
        num_io: ours.num_io,
        a: convert_matrix(&ours.a),
        b: convert_matrix(&ours.b),
        c: convert_matrix(&ours.c),
    };
    oracle.validate().unwrap();

    let z = vec![
        OracleF::from_u32(3),
        OracleF::from_u32(4),
        OracleF::from_u32(1),
        OracleF::from_u32(4),
        OracleF::from_u32(5),
    ];
    let ours_z = vec![kb(3), kb(4), kb(1), kb(4), kb(5)];
    let (oa, ob, oc) = oracle.multiply_vec(&z).unwrap();
    let (a, b, c) = ours.multiply_vec(&ours_z).unwrap();
    for (ours, oracle) in a
        .iter()
        .zip(oa)
        .chain(b.iter().zip(ob))
        .chain(c.iter().zip(oc))
    {
        assert_eq!(ours.canonical(), oracle.as_canonical_u32());
    }

    let padded_ours = ours.pad_regular().unwrap();
    let padded_oracle = oracle.pad_regular().unwrap();
    assert_eq!(padded_ours.num_cons, padded_oracle.num_cons);
    assert_eq!(padded_ours.num_vars, padded_oracle.num_vars);
    assert_eq!(padded_ours.num_io, padded_oracle.num_io);
    assert_eq!(padded_ours.a.entries.len(), padded_oracle.a.entries.len());
    for (ours, oracle) in padded_ours.a.entries.iter().zip(&padded_oracle.a.entries) {
        assert_eq!(ours.row, oracle.row);
        assert_eq!(ours.col, oracle.col);
        assert_eq!(ours.val.canonical(), oracle.val.as_canonical_u32());
    }

    let witness = spartan_whir::R1csWitness {
        w: vec![OracleF::from_u32(3), OracleF::from_u32(4)],
    };
    let public = vec![OracleF::from_u32(4), OracleF::from_u32(5)];
    spartan_whir::validate_satisfaction(&oracle, &witness, &public).unwrap();
}
