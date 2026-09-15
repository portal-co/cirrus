#![no_std]
#![warn(missing_docs)]

//! Cirrus adapter for the shared TinyLabels construction.
//!
//! The construction-neutral implementation lives in
//! [`volar_spec::tinylabels`]. Cirrus deliberately re-exports it instead of
//! maintaining a Ring-LWE fork: its own future responsibility is to bind that
//! shared core to fixed interpreter input manifests and bounded coroutine
//! framing. It does not change Cirrus's default direct-label delivery, table
//! stream formats, label width, or embedded admission policy.
//!
//! See [`../TINYLABELS_CROSS_REPO_PLAN.md`](../TINYLABELS_CROSS_REPO_PLAN.md)
//! for the ownership split and deployment gates.

extern crate alloc;

pub use volar_spec::tinylabels::*;

#[cfg(test)]
mod tests {
    use super::{EncodedLabelBatch, LabelBatch, LabelPair};

    const OFFSET: [u8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    #[test]
    fn cirrus_adapter_uses_the_shared_volar_label_core() {
        let pairs = [
            LabelPair {
                zero: [0x20; 16],
                one: core::array::from_fn(|i| 0x20 ^ OFFSET[i]),
            },
            LabelPair {
                zero: [0x40; 16],
                one: core::array::from_fn(|i| 0x40 ^ OFFSET[i]),
            },
        ];
        let batch = LabelBatch::new(&pairs, OFFSET).expect("shared free-XOR validation");
        assert_eq!(batch.len(), 2);
        let encoded = EncodedLabelBatch::from_pairs(&pairs, OFFSET).expect("shared encoding");
        let choices = EncodedLabelBatch::expanded_choices(&[false, true]);
        assert_eq!(choices, alloc::vec![false, false, false, true, true, true]);
        assert_eq!(encoded.label_count(), 2);
    }
}
