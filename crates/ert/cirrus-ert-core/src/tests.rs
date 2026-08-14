extern crate std;

use crate::RawMemory;

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
