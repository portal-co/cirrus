extern crate std;

use crate::{ComparePredicate, RawMemory, compare_word};

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
fn without_a_detect_override_reads_are_unchanged() {
    let bytes = [1, 2, 3, 4];
    let memory = RawMemory::from(&bytes[..]);

    assert_eq!(memory.read::<4>(0), Some(bytes));
}
