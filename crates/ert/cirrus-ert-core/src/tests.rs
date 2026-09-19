extern crate std;

use crate::{
    ComparePredicate, RawMemory, Shift, add_bits_with_carry_out, add_overflow, arm_condition,
    arm_condition_value, arm_runtime_shift_with_carry, compare_word, subtract_overflow,
    subtract_word_with_carry_out, zero_word,
};

fn word(value: u32) -> [bool; 32] {
    core::array::from_fn(|bit| (value >> bit) & 1 != 0)
}

fn eval(left: u32, right: u32, predicate: ComparePredicate) -> bool {
    compare_word(&mut (), &word(left), &word(right), predicate, &true).unwrap()
}

#[test]
fn eq_and_ne_agree_with_equality() {
    assert!(eval(42, 42, ComparePredicate::Eq));
    assert!(!eval(42, 42, ComparePredicate::Ne));
    assert!(!eval(42, 43, ComparePredicate::Eq));
    assert!(eval(42, 43, ComparePredicate::Ne));
    assert!(eval(0, u32::MAX, ComparePredicate::Ne));
}

#[test]
fn unsigned_comparisons_match_u32_ordering() {
    for (left, right) in [(0, 0), (1, 0), (0, 1), (u32::MAX, 0), (0, u32::MAX), (5, 5)] {
        assert_eq!(
            eval(left, right, ComparePredicate::GeU),
            left >= right,
            "GeU({left}, {right})"
        );
        assert_eq!(
            eval(left, right, ComparePredicate::LtU),
            left < right,
            "LtU({left}, {right})"
        );
    }
}

#[test]
fn signed_comparisons_match_i32_ordering() {
    let values = [0i32, 1, -1, i32::MIN, i32::MAX, -100, 100];
    for &left in &values {
        for &right in &values {
            let (l, r) = (left as u32, right as u32);
            assert_eq!(
                eval(l, r, ComparePredicate::GeS),
                left >= right,
                "GeS({left}, {right})"
            );
            assert_eq!(
                eval(l, r, ComparePredicate::LtS),
                left < right,
                "LtS({left}, {right})"
            );
        }
    }
}

#[test]
fn arithmetic_status_primitives_report_carry_and_zero() {
    let (sum, carry) = add_bits_with_carry_out(&mut (), &word(u32::MAX), &word(1), false)
        .expect("native Boolean addition cannot fail");
    assert_eq!(sum, word(0));
    assert!(carry);
    assert!(zero_word(&mut (), &sum, &true).unwrap());

    let (difference, no_borrow) =
        subtract_word_with_carry_out(&mut (), &word(0), &word(1), &true).unwrap();
    assert_eq!(difference, word(u32::MAX));
    assert!(!no_borrow);

    let (_, no_borrow) =
        subtract_word_with_carry_out(&mut (), &word(u32::MAX), &word(1), &true).unwrap();
    assert!(no_borrow);
}

#[test]
fn overflow_and_arm_conditions_match_the_nzcv_truth_table() {
    assert!(add_overflow(&mut (), false, false, true).unwrap());
    assert!(subtract_overflow(&mut (), false, true, true).unwrap());
    assert!(!add_overflow(&mut (), false, true, true).unwrap());
    assert!(!subtract_overflow(&mut (), true, true, false).unwrap());

    for bits in 0..16u8 {
        let n = bits & 8 != 0;
        let z = bits & 4 != 0;
        let c = bits & 2 != 0;
        let v = bits & 1 != 0;
        for condition in 0..=14 {
            let expected = match condition {
                0 => z,
                1 => !z,
                2 => c,
                3 => !c,
                4 => n,
                5 => !n,
                6 => v,
                7 => !v,
                8 => c && !z,
                9 => !c || z,
                10 => n == v,
                11 => n != v,
                12 => !z && n == v,
                13 => z || n != v,
                14 => true,
                _ => unreachable!(),
            };
            assert_eq!(
                arm_condition_value(Some(n), Some(z), Some(c), Some(v), condition),
                Some(Some(expected)),
                "NZCV={n}{z}{c}{v}, condition={condition}"
            );
            assert_eq!(
                arm_condition(&mut (), n, z, c, v, condition, &true).unwrap(),
                Some(expected),
                "NZCV={n}{z}{c}{v}, condition={condition}"
            );
        }
    }
    assert_eq!(
        arm_condition_value(Some(false), None, None, None, 0),
        Some(None)
    );
    assert_eq!(
        arm_condition_value(None, None, None, None, 14),
        Some(Some(true))
    );
    assert_eq!(arm_condition_value(None, None, None, None, 15), None);
}

#[test]
fn arm_shift_carry_uses_register_count_rules() {
    let (result, carry) = arm_runtime_shift_with_carry(
        &mut (),
        &word(0x8000_0000),
        &word(1),
        Shift::Left,
        &false,
        false,
    )
    .unwrap();
    assert_eq!(result, word(0));
    assert!(carry);

    // A nonzero rotate count that is a multiple of 32 leaves the data in
    // place but still writes C from the top output bit.
    let (result, carry) = arm_runtime_shift_with_carry(
        &mut (),
        &word(0x4000_0001),
        &word(32),
        Shift::RotateRight,
        &false,
        true,
    )
    .unwrap();
    assert_eq!(result, word(0x4000_0001));
    assert!(!carry);

    let (result, carry) = arm_runtime_shift_with_carry(
        &mut (),
        &word(0x8000_0000),
        &word(33),
        Shift::ArithmeticRight,
        &false,
        false,
    )
    .unwrap();
    assert_eq!(result, word(u32::MAX));
    assert!(carry);
}

#[test]
fn a_word_read_at_the_detect_address_returns_the_overridden_value() {
    let memory = RawMemory::from(&[0u8; 8][..]).with_ert_detect(4, 0x1234_5678);

    assert_eq!(memory.read::<4>(4), Some(0x1234_5678u32.to_le_bytes()));
}

#[test]
fn a_word_read_elsewhere_falls_through_to_the_backing_bytes() {
    let bytes = [1, 2, 3, 4, 5, 6, 7, 8];
    let memory = RawMemory::from(&bytes[..]).with_ert_detect(4, 0x1234_5678);

    assert_eq!(memory.read::<4>(0), Some([1, 2, 3, 4]));
}

#[test]
fn a_non_word_read_at_the_detect_address_falls_through_to_the_backing_bytes() {
    let bytes = [1, 2, 3, 4, 5, 6, 7, 8];
    let memory = RawMemory::from(&bytes[..]).with_ert_detect(4, 0x1234_5678);

    assert_eq!(memory.read::<1>(4), Some([5]));
    assert_eq!(memory.read::<2>(4), Some([5, 6]));
}

#[test]
fn mutable_raw_memory_bounds_concrete_writes() {
    let mut bytes = [0u8; 8];
    let memory = RawMemory::from_mut_slice(&mut bytes);
    assert_eq!(memory.write64(4, &0xdead_beefu32.to_le_bytes()), Some(()));
    assert_eq!(memory.read64::<4>(4), Some(0xdead_beefu32.to_le_bytes()));
    assert_eq!(memory.write64(5, &[0u8; 4]), None);
    assert_eq!(
        memory.read64::<8>(0),
        Some([0, 0, 0, 0, 0xef, 0xbe, 0xad, 0xde])
    );
}

#[test]
fn without_a_detect_override_reads_are_unchanged() {
    let bytes = [1, 2, 3, 4];
    let memory = RawMemory::from(&bytes[..]);

    assert_eq!(memory.read::<4>(0), Some(bytes));
}
