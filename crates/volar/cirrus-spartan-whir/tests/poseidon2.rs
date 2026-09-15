use cirrus_spartan_whir::{KoalaBear, Poseidon2KoalaBear16, Poseidon2KoalaBear24};

fn values<const N: usize>(seed: u32) -> [KoalaBear; N] {
    core::array::from_fn(|index| {
        KoalaBear::from_u64(u64::from(
            seed.wrapping_mul(0x45d9f3b)
                .wrapping_add(index as u32 * 0x119d_e1),
        ))
    })
}

#[test]
fn poseidon2_width_16_matches_the_frozen_vector() {
    let mut state = [
        894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831, 120589055,
        19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920, 1783754913,
    ]
    .map(KoalaBear::from_u64);
    Poseidon2KoalaBear16::permute_mut(&mut state);
    assert_eq!(
        state.map(|value| value.canonical()),
        [
            1934285469, 604889435, 133449501, 1026180808, 1830659359, 176667110, 1391183747,
            351743874, 1238264085, 1292768839, 2023573270, 1201586780, 1360691759, 1230682461,
            748270449, 651545025,
        ]
    );
}

#[test]
fn poseidon2_width_24_matches_the_frozen_vector() {
    let mut state = [
        886409618, 1327899896, 1902407911, 591953491, 648428576, 1844789031, 1198336108, 355597330,
        1799586834, 59617783, 790334801, 1968791836, 559272107, 31054313, 1042221543, 474748436,
        135686258, 263665994, 1962340735, 1741539604, 2026927696, 449439011, 1131357108, 50869465,
    ]
    .map(KoalaBear::from_u64);
    Poseidon2KoalaBear24::permute_mut(&mut state);
    assert_eq!(
        state.map(|value| value.canonical()),
        [
            382801106, 82839311, 1503190615, 1987418517, 854076995, 1862291425, 262755189,
            1050814217, 722724562, 741265943, 1026879332, 754316749, 1966025564, 1518878196,
            502200188, 1368172258, 845459257, 1711434837, 724453836, 171032289, 655223446,
            1098636135, 407832555, 1707498914,
        ]
    );
}

#[cfg(feature = "oracle-tests")]
#[test]
fn poseidon2_permutations_match_plonky3_oracle() {
    use p3_field::{PrimeCharacteristicRing, PrimeField32};
    use p3_symmetric::Permutation;
    use spartan_whir::engine::F as OracleF;

    let oracle16 = p3_koala_bear::default_koalabear_poseidon2_16();
    let oracle24 = p3_koala_bear::default_koalabear_poseidon2_24();
    for seed in 0..32_u32 {
        let ours16 = values::<16>(seed * 17 + 1);
        let mut state16 = ours16;
        Poseidon2KoalaBear16::permute_mut(&mut state16);
        let mut expected16 = ours16.map(|value| OracleF::from_u32(value.canonical()));
        oracle16.permute_mut(&mut expected16);
        for (ours, oracle) in state16.iter().zip(expected16) {
            assert_eq!(ours.canonical(), oracle.as_canonical_u32());
        }

        let ours24 = values::<24>(seed * 29 + 3);
        let mut state24 = ours24;
        Poseidon2KoalaBear24::permute_mut(&mut state24);
        let mut expected24 = ours24.map(|value| OracleF::from_u32(value.canonical()));
        oracle24.permute_mut(&mut expected24);
        for (ours, oracle) in state24.iter().zip(expected24) {
            assert_eq!(ours.canonical(), oracle.as_canonical_u32());
        }
    }
}
