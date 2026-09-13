//! WHIR PCS Fiat-Shamir domain separator.
//!
//! Mirrors the pinned upstream `p3-whir` `DomainSeparator`: the protocol
//! structure (parameters, round counts, query counts) is encoded as field
//! elements and absorbed into the challenger before any protocol-specific
//! transcript operations. Evaluation claims are NOT absorbed here; the PCS
//! layer binds those separately.

use alloc::vec::Vec;

use crate::KoalaBear;
use crate::whir::params::{FoldingFactor, SecurityAssumption, WhirConfig};

// Pattern tags (upstream discriminant order).
const TAG_SAMPLE: usize = 0;
const TAG_OBSERVE: usize = 1;
const TAG_HINT: usize = 2;

// Sample sub-labels.
const SAMPLE_INITIAL_COMBINATION: u8 = 0;
const SAMPLE_FOLDING_RANDOMNESS: u8 = 1;
const SAMPLE_COMBINATION: u8 = 2;
const SAMPLE_STIR_QUERIES: u8 = 3;
const SAMPLE_FINAL_QUERIES: u8 = 4;
const SAMPLE_POW_QUERIES: u8 = 5;
const SAMPLE_OOD_QUERY: u8 = 6;
const SAMPLE_TRANSCRIPT_CHECKPOINT: u8 = 7;

// Observe sub-labels.
const OBSERVE_MERKLE_DIGEST: u8 = 0;
const OBSERVE_OOD_ANSWERS: u8 = 1;
const OBSERVE_SUMCHECK_POLY: u8 = 2;
const OBSERVE_FINAL_COEFFS: u8 = 3;
const OBSERVE_POW_NONCE: u8 = 4;
const OBSERVE_PROTOCOL_PARAM: u8 = 5;

// Hint sub-labels.
const HINT_STIR_QUERIES: u8 = 0;
const HINT_STIR_ANSWERS: u8 = 1;
const HINT_MERKLE_PROOF: u8 = 2;
const HINT_DEFERRED_WEIGHT_EVALUATIONS: u8 = 3;

const COUNT_BITS: u32 = 16;
const LABEL_BITS: u32 = 8;

fn encode_entry(tag: usize, label: u8, count: usize) -> KoalaBear {
    debug_assert!(count < (1 << COUNT_BITS));
    let packed = (tag << (COUNT_BITS + LABEL_BITS)) | ((label as usize) << COUNT_BITS) | count;
    KoalaBear::from_u64(packed as u64)
}

/// The WHIR PCS domain separator pattern for one configuration.
#[derive(Clone, Debug)]
pub struct WhirDomainSeparator {
    pattern: Vec<KoalaBear>,
}

impl WhirDomainSeparator {
    fn observe(&mut self, count: usize, label: u8) {
        self.pattern.push(encode_entry(TAG_OBSERVE, label, count));
    }

    fn sample(&mut self, count: usize, label: u8) {
        self.pattern.push(encode_entry(TAG_SAMPLE, label, count));
    }

    fn hint(&mut self, label: u8) {
        self.pattern.push(encode_entry(TAG_HINT, label, 0));
    }

    fn protocol_param(&mut self, value: usize) {
        self.pattern
            .push(encode_entry(TAG_OBSERVE, OBSERVE_PROTOCOL_PARAM, 0));
        self.pattern.push(KoalaBear::from_u64(value as u64));
    }

    fn pow(&mut self, bits: usize) {
        if bits > 0 {
            self.sample(32, SAMPLE_POW_QUERIES);
            self.observe(8, OBSERVE_POW_NONCE);
        }
    }

    fn add_ood(&mut self, num_samples: usize) {
        if num_samples > 0 {
            self.sample(num_samples, SAMPLE_OOD_QUERY);
            self.observe(num_samples, OBSERVE_OOD_ANSWERS);
        }
    }

    fn add_sumcheck(&mut self, rounds: usize, pow_bits: usize) {
        for _ in 0..rounds {
            self.observe(2, OBSERVE_SUMCHECK_POLY);
            self.sample(1, SAMPLE_FOLDING_RANDOMNESS);
            self.pow(pow_bits);
        }
    }

    fn bind_config_params(&mut self, config: &WhirConfig) {
        self.protocol_param(config.num_variables);
        self.protocol_param(config.params.security_level);
        self.protocol_param(config.params.starting_log_inv_rate);
        self.protocol_param(config.params.pow_bits);
        self.protocol_param(config.round_parameters.len());
        for round in &config.round_parameters {
            self.protocol_param(round.log_inv_rate);
        }

        let soundness_discriminant = match config.params.soundness_type {
            SecurityAssumption::UniqueDecoding => 0,
            SecurityAssumption::JohnsonBound => 1,
            SecurityAssumption::CapacityBound => 2,
        };
        self.protocol_param(soundness_discriminant);

        match &config.params.folding_factor {
            FoldingFactor::Constant(f) => {
                self.protocol_param(0);
                self.protocol_param(*f);
            }
            FoldingFactor::ConstantFromSecondRound(first, rest) => {
                self.protocol_param(1);
                self.protocol_param(*first);
                self.protocol_param(*rest);
            }
            FoldingFactor::PerRound(factors) => {
                self.protocol_param(2);
                self.protocol_param(factors.len());
                for &factor in factors {
                    self.protocol_param(factor);
                }
            }
        }
    }

    /// Build the full domain separator for one WHIR PCS instance, matching
    /// upstream's `add_domain_separator`: `commit_statement` followed by
    /// `add_whir_proof`.
    pub fn new(config: &WhirConfig) -> Self {
        let mut ds = Self {
            pattern: Vec::new(),
        };

        // commit_statement
        ds.bind_config_params(config);
        ds.observe(8, OBSERVE_MERKLE_DIGEST);
        ds.add_ood(config.commitment_ood_samples);

        // add_whir_proof
        ds.sample(1, SAMPLE_INITIAL_COMBINATION);
        ds.add_sumcheck(
            config.round_folding_factor(0),
            config.starting_folding_pow_bits,
        );

        let mut domain_size = config.starting_domain_size();
        for (round, r) in config.round_parameters.iter().enumerate() {
            let folded_domain_size = domain_size >> config.round_folding_factor(round);
            let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

            ds.observe(8, OBSERVE_MERKLE_DIGEST);
            ds.add_ood(r.ood_samples);
            ds.pow(r.pow_bits);
            ds.sample(1, SAMPLE_TRANSCRIPT_CHECKPOINT);
            ds.sample(r.num_queries * domain_size_bytes, SAMPLE_STIR_QUERIES);
            ds.hint(HINT_STIR_QUERIES);
            ds.hint(HINT_MERKLE_PROOF);
            ds.sample(1, SAMPLE_COMBINATION);
            ds.add_sumcheck(config.round_folding_factor(round + 1), r.folding_pow_bits);
            domain_size >>= config.rs_reduction_factor(round);
        }

        let folded_domain_size =
            domain_size >> config.round_folding_factor(config.round_parameters.len());
        let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

        ds.observe(1 << config.final_sumcheck_rounds, OBSERVE_FINAL_COEFFS);
        ds.pow(config.final_pow_bits);
        ds.sample(
            domain_size_bytes * config.final_queries,
            SAMPLE_FINAL_QUERIES,
        );
        ds.hint(HINT_STIR_ANSWERS);
        ds.hint(HINT_MERKLE_PROOF);
        ds.add_sumcheck(config.final_sumcheck_rounds, config.final_folding_pow_bits);
        ds.hint(HINT_DEFERRED_WEIGHT_EVALUATIONS);

        ds
    }

    /// Absorb the entire pattern into the challenger, matching upstream's
    /// `observe_domain_separator`.
    pub fn observe_into(&self, transcript: &mut crate::PoseidonTranscript) {
        transcript.observe_slice(&self.pattern);
    }

    /// The encoded pattern entries, for inspection and frozen tests.
    pub fn pattern(&self) -> &[KoalaBear] {
        &self.pattern
    }
}
