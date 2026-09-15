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

    /// Observe a slice of quintic-extension elements in order, matching
    /// upstream's `observe_algebra_slice`.
    pub fn observe_quintic_slice(&mut self, values: &[QuinticExtension]) {
        for &value in values {
            self.observe_quintic(value);
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

    /// Sample `bits` uniformly via upstream's rejection-sampled
    /// `sample_uniform_bits::<true>`.
    ///
    /// For `bits <= 24` one field element is drawn (and redrawn while its
    /// canonical value is at least `PRIME & !((1 << bits) - 1)`); wider
    /// requests draw two half-width chunks and combine them little-endian,
    /// exactly as upstream's slow path.
    pub fn sample_uniform_bits(&mut self, bits: usize) -> Result<usize, TranscriptError> {
        if bits == 0 {
            return Ok(0);
        }
        if bits >= usize::BITS as usize || (bits < 64 && (1_u64 << bits) >= 2_130_706_433) {
            return Err(TranscriptError::InvalidBitCount);
        }
        if bits <= MAX_SINGLE_SAMPLE_BITS {
            let m = sampling_bits_m(bits);
            let mut value = self.sample_base().canonical() as u64;
            while value >= m {
                value = self.sample_base().canonical() as u64;
            }
            Ok(value as usize & ((1_usize << bits) - 1))
        } else {
            let half1 = bits / 2;
            let half2 = bits - half1;
            let m1 = sampling_bits_m(half1);
            let mut v1 = self.sample_base().canonical() as u64;
            while v1 >= m1 {
                v1 = self.sample_base().canonical() as u64;
            }
            let chunk1 = v1 as usize & ((1_usize << half1) - 1);
            let m2 = sampling_bits_m(half2);
            let mut v2 = self.sample_base().canonical() as u64;
            while v2 >= m2 {
                v2 = self.sample_base().canonical() as u64;
            }
            let chunk2 = v2 as usize & ((1_usize << half2) - 1);
            Ok(chunk1 | (chunk2 << half1))
        }
    }

    /// Verify a proof-of-work witness: absorb it, then require the low
    /// `bits` of the next sample to be zero. Matches upstream's
    /// `GrindingChallenger::check_witness`.
    pub fn check_witness(
        &mut self,
        bits: usize,
        witness: KoalaBear,
    ) -> Result<bool, TranscriptError> {
        if bits == 0 {
            return Ok(true);
        }
        self.observe(witness);
        Ok(self.sample_bits(bits)? == 0)
    }

    /// Grind a proof-of-work witness: find the smallest canonical candidate
    /// whose check passes, then absorb it and advance the transcript exactly
    /// as [`Self::check_witness`] does. Matches upstream's serial grind
    /// semantics (candidates are tried in order `0, 1, 2, ...`).
    pub fn grind(&mut self, bits: usize) -> Result<KoalaBear, TranscriptError> {
        if bits == 0 {
            return Ok(KoalaBear::ZERO);
        }
        if bits >= 31 {
            // Upstream asserts (1 << bits) < ORDER.
            return Err(TranscriptError::InvalidBitCount);
        }
        let mut candidate = 0_u64;
        loop {
            let witness = KoalaBear::from_u64(candidate);
            let mut probe = self.clone();
            if probe.check_witness(bits, witness)? {
                // Commit the winning witness to the real transcript.
                let accepted = self.check_witness(bits, witness)?;
                debug_assert!(accepted);
                return Ok(witness);
            }
            candidate += 1;
            if candidate >= 2_130_706_433 {
                return Err(TranscriptError::InvalidBitCount);
            }
        }
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

/// Widest single-sample uniform bit draw for KoalaBear, matching upstream's
/// `UniformSamplingField::MAX_SINGLE_SAMPLE_BITS`.
const MAX_SINGLE_SAMPLE_BITS: usize = 24;

/// Rejection threshold for `bits`-wide uniform draws, matching upstream's
/// `SAMPLING_BITS_M`: the prime with its low `bits` cleared.
const fn sampling_bits_m(bits: usize) -> u64 {
    (2_130_706_433_u64) & !((1_u64 << bits) - 1)
}

/// Number of coefficients sampled for one quintic transcript challenge.
pub const QUINTIC_TRANSCRIPT_SAMPLES: usize = QUINTIC_DEGREE;
