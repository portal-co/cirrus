use cirrus_spartan_whir::{
    KoalaBear, PoseidonMerkleTree, PoseidonTranscript, poseidon_compress2, poseidon_hash_fixed,
};

fn kb(value: u32) -> KoalaBear {
    KoalaBear::from_u64(u64::from(value))
}

fn row(seed: u32, width: usize) -> Vec<KoalaBear> {
    (0..width)
        .map(|index| kb(seed * 37 + index as u32 * 101 + 7))
        .collect()
}

#[test]
fn merkle_tree_commits_opens_and_rejects_mutations() {
    let rows = (0..8).map(|seed| row(seed, 5)).collect::<Vec<_>>();
    let tree = PoseidonMerkleTree::commit_rows(&rows).unwrap();
    let root = tree.root();
    for (index, row) in rows.iter().enumerate() {
        let path = tree.open(index).unwrap();
        PoseidonMerkleTree::verify_with_height(&root, index, row, &path, rows.len()).unwrap();
        let mut bad_path = path.clone();
        bad_path.siblings[0][0] = bad_path.siblings[0][0] + KoalaBear::ONE;
        assert!(
            PoseidonMerkleTree::verify_with_height(&root, index, row, &bad_path, rows.len())
                .is_err()
        );
    }
    assert!(PoseidonMerkleTree::commit_rows(&[]).is_err());
    assert!(PoseidonMerkleTree::commit_rows(&rows[..3]).is_err());
}

#[test]
fn labeled_transcript_labels_are_cryptographically_bound() {
    let mut a = PoseidonTranscript::new();
    let mut b = PoseidonTranscript::new();
    a.observe_labeled(b"gamma", &[kb(1), kb(2)]).unwrap();
    b.observe_labeled(b"eta", &[kb(1), kb(2)]).unwrap();
    assert_ne!(a.sample_base(), b.sample_base());

    let mut tagged = PoseidonTranscript::new();
    let challenge = tagged
        .sample_labeled_quintic(b"ram-permutation-challenge")
        .unwrap();
    for coefficient in challenge.canonical_coefficients() {
        assert!(coefficient < cirrus_spartan_whir::KOALABEAR_MODULUS);
    }
}

#[cfg(feature = "oracle-tests")]
fn oracle_values(values: &[KoalaBear]) -> Vec<spartan_whir::engine::F> {
    use p3_field::PrimeCharacteristicRing;
    values
        .iter()
        .map(|value| spartan_whir::engine::F::from_u32(value.canonical()))
        .collect()
}

#[cfg(feature = "oracle-tests")]
#[test]
fn poseidon_hash_and_compression_match_upstream_oracle() {
    use p3_field::PrimeField32;
    use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
    use spartan_whir::engine::{PoseidonFieldHash, PoseidonNodeCompress};

    let hasher = PoseidonFieldHash::new(p3_koala_bear::default_koalabear_poseidon2_24());
    let compressor = PoseidonNodeCompress::new(p3_koala_bear::default_koalabear_poseidon2_16());
    for length in [0, 1, 7, 16, 17, 39] {
        let input = row(length as u32 + 1, length);
        let ours = poseidon_hash_fixed(&input);
        let oracle = hasher.hash_iter(oracle_values(&input));
        for (ours, oracle) in ours.iter().zip(oracle) {
            assert_eq!(ours.canonical(), oracle.as_canonical_u32());
        }
    }

    let left = poseidon_hash_fixed(&row(100, 13));
    let right = poseidon_hash_fixed(&row(200, 14));
    let ours = poseidon_compress2([left, right]);
    let oracle_left: [_; 8] = oracle_values(&left).try_into().unwrap();
    let oracle_right: [_; 8] = oracle_values(&right).try_into().unwrap();
    let oracle = compressor.compress([oracle_left, oracle_right]);
    for (ours, oracle) in ours.iter().zip(oracle) {
        assert_eq!(ours.canonical(), oracle.as_canonical_u32());
    }
}

#[cfg(feature = "oracle-tests")]
#[test]
fn poseidon_merkle_root_matches_upstream_single_matrix_oracle() {
    use p3_commit::Mmcs;
    use p3_field::PrimeField32;
    use p3_matrix::dense::RowMajorMatrix;
    use spartan_whir::engine::{PoseidonFieldHash, PoseidonNodeCompress};
    use spartan_whir::plonky3_whir_pcs::InnerPoseidonMmcs;

    let rows = (0..8).map(|seed| row(seed + 300, 3)).collect::<Vec<_>>();
    let ours = PoseidonMerkleTree::commit_rows(&rows).unwrap();
    let flat = rows
        .iter()
        .flat_map(|row| oracle_values(row))
        .collect::<Vec<_>>();
    let matrix = RowMajorMatrix::new(flat, 3);
    let mmcs = InnerPoseidonMmcs::new(
        PoseidonFieldHash::new(p3_koala_bear::default_koalabear_poseidon2_24()),
        PoseidonNodeCompress::new(p3_koala_bear::default_koalabear_poseidon2_16()),
        0,
    );
    let (cap, _tree) = mmcs.commit(vec![matrix]);
    let oracle_root = cap[0];
    for (ours, oracle) in ours.root().iter().zip(oracle_root) {
        assert_eq!(ours.canonical(), oracle.as_canonical_u32());
    }
}

#[cfg(feature = "oracle-tests")]
#[test]
fn poseidon_transcript_matches_upstream_observe_and_sample_schedule() {
    use p3_challenger::{CanObserve, CanSample, CanSampleBits, FieldChallenger};
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32};
    use spartan_whir::engine::{F as OracleF, PoseidonChallenger, QuinticExtension as OracleEF};

    let mut ours = PoseidonTranscript::new();
    let mut oracle = PoseidonChallenger::new(p3_koala_bear::default_koalabear_poseidon2_16());
    let first = row(1, 19);
    ours.observe_slice(&first);
    oracle.observe_slice(&oracle_values(&first));
    let ext =
        cirrus_spartan_whir::QuinticExtension::from_canonical_coefficients([11, 22, 33, 44, 55])
            .unwrap();
    ours.observe_quintic(ext);
    let oracle_ext = OracleEF::from_basis_coefficients_fn(|index| {
        OracleF::from_u32(ext.canonical_coefficients()[index])
    });
    oracle.observe_algebra_element(oracle_ext);

    for _ in 0..23 {
        assert_eq!(
            ours.sample_base().canonical(),
            CanSample::<OracleF>::sample(&mut oracle).as_canonical_u32()
        );
    }
    let ours_ext = ours.sample_quintic();
    let oracle_sample: OracleEF = CanSample::<OracleEF>::sample(&mut oracle);
    for (ours, oracle) in ours_ext
        .canonical_coefficients()
        .iter()
        .zip(<OracleEF as BasedVectorSpace<OracleF>>::as_basis_coefficients_slice(&oracle_sample))
    {
        assert_eq!(*ours, oracle.as_canonical_u32());
    }
    assert_eq!(
        ours.sample_bits(17).unwrap(),
        oracle.sample_bits(17),
        "bit sampling schedule diverged"
    );
}
