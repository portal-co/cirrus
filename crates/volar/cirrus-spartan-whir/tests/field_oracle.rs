use cirrus_spartan_whir::{KoalaBear, KoalaBearError, QUINTIC_DEGREE, QuinticExtension};

fn sample_base(seed: u64) -> KoalaBear {
    // Deterministic splitmix-style sample; modulo reduction is intentional for
    // constructing test inputs, while parsing remains canonical.
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    KoalaBear::from_u64(z ^ (z >> 31))
}

fn sample_extension(seed: u64) -> QuinticExtension {
    QuinticExtension::new(core::array::from_fn(|index| {
        sample_base(seed.wrapping_mul(7).wrapping_add(index as u64))
    }))
}

#[test]
fn base_arithmetic_is_canonical_and_algebraic() {
    let zero = KoalaBear::ZERO;
    let one = KoalaBear::ONE;
    assert_eq!(zero + zero, zero);
    assert_eq!(zero * one, zero);
    assert_eq!(one * one, one);
    assert_eq!(one.neg() + one, zero);
    assert_eq!(
        KoalaBear::from_le_bytes(KoalaBear::from_u64(123456789).to_le_bytes()).unwrap(),
        KoalaBear::from_u64(123456789)
    );
    assert_eq!(
        KoalaBear::from_le_bytes(cirrus_spartan_whir::KOALABEAR_MODULUS.to_le_bytes()),
        Err(KoalaBearError::Noncanonical {
            value: cirrus_spartan_whir::KOALABEAR_MODULUS
        })
    );

    for seed in 0..64 {
        let a = sample_base(seed);
        let b = sample_base(seed + 1000);
        assert_eq!(a + b, b + a);
        assert_eq!(a * b, b * a);
        assert_eq!((a - b) + b, a);
        if !a.is_zero() {
            assert_eq!(a * a.inverse().unwrap(), one);
        }
    }
}

#[test]
fn quintic_reduction_matches_defining_polynomial() {
    let x = QuinticExtension::new([
        KoalaBear::ZERO,
        KoalaBear::ONE,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
    ]);
    let x2 = x.square();
    let x5 = x2.square() * x;
    // X^5 = 1 - X^2.
    assert_eq!(x5 + x2, QuinticExtension::ONE);

    for seed in 0..32 {
        let a = sample_extension(seed);
        let b = sample_extension(seed + 10_000);
        let c = sample_extension(seed + 20_000);
        assert_eq!(a * b, b * a);
        assert_eq!((a + b) * c, a * c + b * c);
        if !a.is_zero() {
            assert_eq!(a * a.inverse().unwrap(), QuinticExtension::ONE);
        }
    }
    assert_eq!(
        QuinticExtension::ZERO.inverse(),
        Err(cirrus_spartan_whir::QuinticExtensionError::ZeroInverse)
    );
    assert_eq!(QUINTIC_DEGREE, 5);
}

#[cfg(feature = "oracle-tests")]
#[test]
fn base_arithmetic_matches_plonky3_oracle() {
    use p3_field::{Field, PrimeCharacteristicRing, PrimeField32};
    use spartan_whir::engine::F as OracleF;

    for seed in 0..256 {
        let a = sample_base(seed);
        let b = sample_base(seed + 0x1000_0000);
        let oa = OracleF::from_u32(a.canonical());
        assert_eq!(
            oa.as_canonical_u32(),
            a.canonical(),
            "oracle construction differs"
        );
        let ob = OracleF::from_u32(b.canonical());
        let ours = (a + b).canonical();
        let oracle = (oa + ob).as_canonical_u32();
        assert_eq!(ours, oracle);
        assert_eq!((a - b).canonical(), (oa - ob).as_canonical_u32());
        assert_eq!((a * b).canonical(), (oa * ob).as_canonical_u32());
        if !a.is_zero() {
            assert_eq!(
                a.inverse().unwrap().canonical(),
                oa.inverse().as_canonical_u32()
            );
        }
    }
}

#[cfg(feature = "oracle-tests")]
#[test]
fn quintic_arithmetic_matches_plonky3_oracle() {
    use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32};
    use spartan_whir::engine::{F as OracleF, QuinticExtension as OracleEF};

    for seed in 0..128 {
        let a = sample_extension(seed);
        let b = sample_extension(seed + 0x5555_0000);
        let oa =
            OracleEF::from_basis_coefficients_fn(|index| OracleF::from_u32(a.0[index].canonical()));
        let ob =
            OracleEF::from_basis_coefficients_fn(|index| OracleF::from_u32(b.0[index].canonical()));
        let ours_add = (a + b).canonical_coefficients();
        let oracle_add: OracleEF = oa + ob;
        let ours_sub = (a - b).canonical_coefficients();
        let oracle_sub: OracleEF = oa - ob;
        let ours_mul = (a * b).canonical_coefficients();
        let oracle_mul: OracleEF = oa * ob;
        for index in 0..QUINTIC_DEGREE {
            assert_eq!(
                ours_add[index],
                <OracleEF as BasedVectorSpace<OracleF>>::as_basis_coefficients_slice(&oracle_add)
                    [index]
                    .as_canonical_u32()
            );
            assert_eq!(
                ours_sub[index],
                <OracleEF as BasedVectorSpace<OracleF>>::as_basis_coefficients_slice(&oracle_sub)
                    [index]
                    .as_canonical_u32()
            );
            assert_eq!(
                ours_mul[index],
                <OracleEF as BasedVectorSpace<OracleF>>::as_basis_coefficients_slice(&oracle_mul)
                    [index]
                    .as_canonical_u32()
            );
        }
        if !a.is_zero() {
            let ours_inverse = a.inverse().unwrap().canonical_coefficients();
            let oracle_inverse: OracleEF = oa.inverse();
            for index in 0..QUINTIC_DEGREE {
                assert_eq!(
                    ours_inverse[index],
                    <OracleEF as BasedVectorSpace<OracleF>>::as_basis_coefficients_slice(
                        &oracle_inverse
                    )[index]
                        .as_canonical_u32()
                );
            }
        }
    }
}
