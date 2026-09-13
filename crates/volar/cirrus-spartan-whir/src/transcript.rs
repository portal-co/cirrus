//! Upstream-compatible Poseidon duplex transcript.
//!
//! The raw sponge mirrors `DuplexChallenger<KoalaBear, Poseidon2-16, 16, 8>`.
//! Typed label helpers use the same explicit tag encoding as upstream Spark:
//! first the byte length, then each byte as one base-field element.

use alloc::vec::Vec;
use core::fmt;

use crate::{
    KoalaBear, POSEIDON_CHALLENGER_RATE, POSEIDON2_WIDTH_16, Poseidon2KoalaBear16, QUINTIC_DEGREE,
    QuinticExtension,
};

/// Raw Poseidon2 duplex challenger state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoseidonTranscript {
    sponge_state: [KoalaBear; POSEIDON2_WIDTH_16],
    input_buffer: Vec<KoalaBear>,
    output_buffer: Vec<KoalaBear>,
}

/// Why a transcript operation is invalid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TranscriptError {
    /// A tag or label cannot be encoded as a `u32` length.
    LabelTooLong,
    /// The requested bit count is invalid for this field/platform.
    InvalidBitCount,
}

impl fmt::Display for TranscriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LabelTooLong => f.write_str("transcript label is too long"),
            Self::InvalidBitCount => f.write_str("invalid transcript bit-sampling request"),
        }
    }
}

impl core::error::Error for TranscriptError {}

impl Default for PoseidonTranscript {
    fn default() -> Self {
        Self::new()
    }
}

impl PoseidonTranscript {
    /// Create an empty upstream-compatible challenger.
    pub fn new() -> Self {
        Self {
            sponge_state: [KoalaBear::ZERO; POSEIDON2_WIDTH_16],
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
        }
    }

    /// Observe one base-field element.
    pub fn observe(&mut self, value: KoalaBear) {
        self.output_buffer.clear();
        self.input_buffer.push(value);
        if self.input_buffer.len() == POSEIDON_CHALLENGER_RATE {
            self.duplex();
        }
    }

    /// Observe a slice of base-field elements in order.
    pub fn observe_slice(&mut self, values: &[KoalaBear]) {
        for &value in values {
            self.observe(value);
        }
    }

    /// Observe a quintic-extension element in canonical polynomial-basis
    /// coefficient order.
    pub fn observe_quintic(&mut self, value: QuinticExtension) {
        for coefficient in value.canonical_coefficients() {
            self.observe(KoalaBear::from_u64(u64::from(coefficient)));
        }
    }

    /// Observe an explicit byte tag as `len || bytes`, matching upstream's
    /// Spark tag convention.
    pub fn observe_tag(&mut self, tag: &[u8]) -> Result<(), TranscriptError> {
        let length = u32::try_from(tag.len()).map_err(|_| TranscriptError::LabelTooLong)?;
        self.observe(KoalaBear::from_u64(u64::from(length)));
        for &byte in tag {
            self.observe(KoalaBear::from_u64(u64::from(byte)));
        }
        Ok(())
    }

    /// Observe a typed label followed by base-field values. The label is
    /// cryptographically absorbed using [`Self::observe_tag`].
    pub fn observe_labeled(
        &mut self,
        label: &[u8],
        values: &[KoalaBear],
    ) -> Result<(), TranscriptError> {
        self.observe_tag(label)?;
        self.observe_slice(values);
        Ok(())
    }

    /// Sample one base-field challenge.
    pub fn sample_base(&mut self) -> KoalaBear {
        if !self.input_buffer.is_empty() || self.output_buffer.is_empty() {
            self.duplex();
        }
        self.output_buffer
            .pop()
            .expect("duplex refill leaves a nonempty output buffer")
    }

    /// Sample one quintic-extension challenge in basis-coefficient order.
    pub fn sample_quintic(&mut self) -> QuinticExtension {
        QuinticExtension::new(core::array::from_fn(|_| self.sample_base()))
    }

    /// Sample a typed base-field challenge by first absorbing its label.
    pub fn sample_labeled_base(&mut self, label: &[u8]) -> Result<KoalaBear, TranscriptError> {
        self.observe_tag(label)?;
        Ok(self.sample_base())
    }

    /// Sample a typed quintic challenge by first absorbing its label.
    pub fn sample_labeled_quintic(
        &mut self,
        label: &[u8],
    ) -> Result<QuinticExtension, TranscriptError> {
        self.observe_tag(label)?;
        Ok(self.sample_quintic())
    }

    /// Sample the low `bits` of one base-field challenge, matching upstream's
    /// bounded-bias `sample_bits` helper.
    pub fn sample_bits(&mut self, bits: usize) -> Result<usize, TranscriptError> {
        if bits >= usize::BITS as usize || (bits < 64 && (1_u64 << bits) >= 2_130_706_433) {
            return Err(TranscriptError::InvalidBitCount);
        }
        Ok((self.sample_base().canonical() as usize) & ((1_usize << bits) - 1))
    }

    fn duplex(&mut self) {
        let absorbed = self.input_buffer.len();
        debug_assert!(absorbed <= POSEIDON_CHALLENGER_RATE);
        for (index, value) in self.input_buffer.drain(..).enumerate() {
            self.sponge_state[index] = value;
        }
        if absorbed > 0 {
            for value in &mut self.sponge_state[absorbed..POSEIDON_CHALLENGER_RATE] {
                *value = KoalaBear::ZERO;
            }
            self.sponge_state[POSEIDON_CHALLENGER_RATE] =
                self.sponge_state[POSEIDON_CHALLENGER_RATE] + KoalaBear::from_u64(absorbed as u64);
        }
        Poseidon2KoalaBear16::permute_mut(&mut self.sponge_state);
        self.output_buffer.clear();
        self.output_buffer
            .extend_from_slice(&self.sponge_state[..POSEIDON_CHALLENGER_RATE]);
    }
}

/// Number of coefficients sampled for one quintic transcript challenge.
pub const QUINTIC_TRANSCRIPT_SAMPLES: usize = QUINTIC_DEGREE;
