//! Spartan-facing WHIR PCS adapter.
//!
//! Mirrors the pinned upstream `spartan-whir` `Plonky3WhirPcs` layer:
//! parameter mapping from the Spartan-facing `WhirParams` shape, the typed
//! [`SecurityProfile`], the commit/open/verify-parse/verify-finalize
//! transcript schedule, and the [`DirectSparsePcs`] boundary implementation
//! used by the Slice-4 DirectSparse driver.

use alloc::vec::Vec;
use core::fmt;

use crate::spartan::DirectSparsePcs;
use crate::whir::domain_separator::WhirDomainSeparator;
use crate::whir::params::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
use crate::whir::pcs::{
    WhirError, WhirProof, WhirProverData, commit_base_prefix, expand_from_univariate,
    parse_initial_ood, whir_prove, whir_verify,
};
use crate::whir::sumcheck::{Constraint, EqStatement, Statements};
use crate::{KoalaBear, PoseidonDigest, PoseidonTranscript, QuinticExtension};

/// Spartan-facing WHIR folding schedule, mirroring upstream
/// `spartan-whir`'s `WhirFoldingSchedule`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WhirFoldingSchedule {
    /// Use the same folding factor in every WHIR round.
    Constant(usize),
    /// Use one folding factor for the first round and another for the rest.
    ConstantFromSecondRound {
        /// First-round folding factor.
        first: usize,
        /// Folding factor for every later round.
        rest: usize,
    },
    /// Use an explicit per-round folding factor list.
    PerRound(Vec<usize>),
}

impl WhirFoldingSchedule {
    /// The first round's folding factor.
    pub fn first_round(&self) -> usize {
        match self {
            Self::Constant(factor) => *factor,
            Self::ConstantFromSecondRound { first, .. } => *first,
            Self::PerRound(factors) => factors.first().copied().unwrap_or(0),
        }
    }
}

/// Spartan-facing WHIR parameters, mirroring upstream `spartan-whir`'s
/// `WhirParams` (without its serde surface).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WhirParams {
    /// Grinding budget in bits.
    pub pow_bits: u32,
    /// Legacy constant folding factor (used when `folding_schedule` is
    /// `None`).
    pub folding_factor: usize,
    /// Initial logarithmic inverse rate.
    pub starting_log_inv_rate: usize,
    /// Domain reduction applied at the first round.
    pub rs_domain_initial_reduction_factor: usize,
    /// Optional explicit folding schedule.
    pub folding_schedule: Option<WhirFoldingSchedule>,
    /// Optional explicit per-round log-inverse rates.
    pub round_log_inv_rates: Vec<usize>,
}

impl Default for WhirParams {
    fn default() -> Self {
        Self {
            pow_bits: 16,
            folding_factor: 4,
            starting_log_inv_rate: 1,
            rs_domain_initial_reduction_factor: 1,
            folding_schedule: None,
            round_log_inv_rates: Vec::new(),
        }
    }
}

impl WhirParams {
    /// The effective folding schedule, mirroring upstream
    /// `WhirParams::effective_folding_schedule`.
    pub fn effective_folding_schedule(&self) -> WhirFoldingSchedule {
        self.folding_schedule
            .clone()
            .unwrap_or(WhirFoldingSchedule::Constant(self.folding_factor))
    }
}

/// Typed security profile freezing the security level, Merkle security
/// level, soundness assumption, and WHIR parameters, mirroring upstream's
/// `SecurityConfig` + `WhirPcsConfig` validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SecurityProfile {
    /// Target protocol security level in bits.
    pub security_level_bits: u32,
    /// Target Merkle (hash) security level in bits.
    pub merkle_security_bits: u32,
    /// The proximity soundness regime.
    pub soundness: SecurityAssumption,
    /// WHIR protocol parameters.
    pub whir: WhirParams,
}

/// Minimum and maximum supported security levels, matching upstream's
/// `MIN_SECURITY_BITS`/`MAX_SECURITY_BITS`.
pub const MIN_SECURITY_BITS: u32 = 80;
/// Maximum target supported by the eight-element KoalaBear Poseidon digest.
pub const MAX_SECURITY_BITS: u32 = 123;

/// Why a security profile or WHIR configuration is invalid.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WhirConfigBuildError {
    /// The protocol security level is below the supported minimum.
    SecurityBelowMinimum,
    /// The protocol security level exceeds the digest-supported maximum.
    SecurityAboveMaximum,
    /// The Merkle security level is below the supported minimum.
    MerkleSecurityBelowMinimum,
    /// The Merkle security level exceeds the digest-supported maximum.
    MerkleSecurityAboveMaximum,
    /// The first folding factor is zero.
    ZeroFoldingFactor,
    /// The RS domain initial reduction factor is zero.
    ZeroRsDomainInitialReductionFactor,
    /// The RS domain initial reduction factor exceeds the first folding
    /// factor.
    RsDomainInitialReductionFactorExceedsFirstFoldingFactor,
    /// The explicit round-rate count does not match the derived round count.
    RoundRateCountMismatch,
    /// A rate derivation would grow the RS domain.
    RateGrowsDomain,
    /// The WHIR configuration derivation failed.
    Whir(crate::whir::params::WhirConfigError),
}

impl fmt::Display for WhirConfigBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SecurityBelowMinimum => f.write_str("security level below the 80-bit minimum"),
            Self::SecurityAboveMaximum => {
                f.write_str("security level above the 123-bit digest maximum")
            }
            Self::MerkleSecurityBelowMinimum => {
                f.write_str("Merkle security below the 80-bit minimum")
            }
            Self::MerkleSecurityAboveMaximum => {
                f.write_str("Merkle security above the 123-bit digest maximum")
            }
            Self::ZeroFoldingFactor => f.write_str("first folding factor must be positive"),
            Self::ZeroRsDomainInitialReductionFactor => {
                f.write_str("RS domain initial reduction factor must be positive")
            }
            Self::RsDomainInitialReductionFactorExceedsFirstFoldingFactor => {
                f.write_str("RS domain initial reduction factor exceeds the first folding factor")
            }
            Self::RoundRateCountMismatch => {
                f.write_str("explicit round-rate count does not match the round count")
            }
            Self::RateGrowsDomain => f.write_str("requested rate would grow the RS domain"),
            Self::Whir(error) => write!(f, "{error}"),
        }
    }
}

impl core::error::Error for WhirConfigBuildError {}

impl SecurityProfile {
    /// Explicit test profile matching upstream phase-3 no-ZK fixtures.
    pub fn capacity_bound_80_test() -> Self {
        Self {
            security_level_bits: 80,
            merkle_security_bits: 80,
            soundness: SecurityAssumption::CapacityBound,
            whir: WhirParams {
                pow_bits: 0,
                folding_factor: 1,
                starting_log_inv_rate: 6,
                rs_domain_initial_reduction_factor: 1,
                folding_schedule: None,
                round_log_inv_rates: Vec::new(),
            },
        }
    }

    /// Validate the profile's security levels, mirroring upstream
    /// `SecurityConfig::validate` and the `WhirPcsConfig::validate`
    /// structural checks.
    pub fn validate(&self, num_variables: usize) -> Result<(), WhirConfigBuildError> {
        if self.security_level_bits < MIN_SECURITY_BITS {
            return Err(WhirConfigBuildError::SecurityBelowMinimum);
        }
        if self.security_level_bits > MAX_SECURITY_BITS {
            return Err(WhirConfigBuildError::SecurityAboveMaximum);
        }
        if self.merkle_security_bits < MIN_SECURITY_BITS {
            return Err(WhirConfigBuildError::MerkleSecurityBelowMinimum);
        }
        if self.merkle_security_bits > MAX_SECURITY_BITS {
            return Err(WhirConfigBuildError::MerkleSecurityAboveMaximum);
        }
        let schedule = self.whir.effective_folding_schedule();
        let first = schedule.first_round();
        if first == 0 {
            return Err(WhirConfigBuildError::ZeroFoldingFactor);
        }
        if self.whir.rs_domain_initial_reduction_factor == 0 {
            return Err(WhirConfigBuildError::ZeroRsDomainInitialReductionFactor);
        }
        if self.whir.rs_domain_initial_reduction_factor > first {
            return Err(
                WhirConfigBuildError::RsDomainInitialReductionFactorExceedsFirstFoldingFactor,
            );
        }
        let _ = num_variables;
        Ok(())
    }

    /// Derive the explicit per-round log-inverse rates, mirroring upstream
    /// `poseidon_round_log_inv_rates`.
    fn round_log_inv_rates(
        &self,
        num_variables: usize,
        folding: &FoldingFactor,
        num_rounds: usize,
    ) -> Result<Vec<usize>, WhirConfigBuildError> {
        if !self.whir.round_log_inv_rates.is_empty() {
            if self.whir.round_log_inv_rates.len() != num_rounds {
                return Err(WhirConfigBuildError::RoundRateCountMismatch);
            }
            return Ok(self.whir.round_log_inv_rates.clone());
        }
        let schedule = self.whir.effective_folding_schedule();
        let concrete = folding
            .compute_folding_schedule(num_variables)
            .map_err(crate::whir::params::WhirConfigError::FoldingFactor)
            .map_err(WhirConfigBuildError::Whir)?;
        let _ = schedule;
        let mut rates = Vec::with_capacity(num_rounds);
        let mut rate = self.whir.starting_log_inv_rate;
        for round in 0..num_rounds {
            let reduction = if round == 0 {
                self.whir.rs_domain_initial_reduction_factor
            } else {
                1
            };
            let folding_factor = concrete[round];
            if reduction > rate + folding_factor {
                return Err(WhirConfigBuildError::RateGrowsDomain);
            }
            rate += folding_factor - reduction;
            rates.push(rate);
        }
        Ok(rates)
    }

    /// Derive the full WHIR configuration for `num_variables`, mirroring
    /// upstream `build_poseidon_plain_pcs_parts`.
    pub fn derive_config(&self, num_variables: usize) -> Result<WhirConfig, WhirConfigBuildError> {
        self.validate(num_variables)?;
        let schedule = self.whir.effective_folding_schedule();
        let folding = match &schedule {
            WhirFoldingSchedule::Constant(factor) => FoldingFactor::Constant(*factor),
            WhirFoldingSchedule::ConstantFromSecondRound { first, rest } => {
                FoldingFactor::ConstantFromSecondRound(*first, *rest)
            }
            WhirFoldingSchedule::PerRound(factors) => FoldingFactor::PerRound(factors.clone()),
        };
        let concrete = folding
            .compute_folding_schedule(num_variables)
            .map_err(crate::whir::params::WhirConfigError::FoldingFactor)
            .map_err(WhirConfigBuildError::Whir)?;
        let num_rounds = concrete.len() - 1;
        let round_log_inv_rates = self.round_log_inv_rates(num_variables, &folding, num_rounds)?;
        let protocol_params = ProtocolParameters {
            starting_log_inv_rate: self.whir.starting_log_inv_rate,
            round_log_inv_rates,
            folding_factor: folding,
            soundness_type: self.soundness,
            security_level: self.security_level_bits as usize,
            pow_bits: self.whir.pow_bits as usize,
        };
        WhirConfig::new(num_variables, protocol_params).map_err(WhirConfigBuildError::Whir)
    }
}

/// The plain WHIR PCS bound to a [`SecurityProfile`], implementing the
/// Slice-4 [`DirectSparsePcs`] boundary. A single value serves both sides:
/// the prover calls [`DirectSparsePcs::commit`] then
/// [`DirectSparsePcs::open`]; the verifier uses a fresh value with only the
/// configuration.
pub struct WhirPcs {
    config: WhirConfig,
    domain_separator: WhirDomainSeparator,
    prover_data: Option<WhirProverData>,
}

impl WhirPcs {
    /// Build the PCS for `num_variables` under `profile`.
    pub fn new(
        num_variables: usize,
        profile: &SecurityProfile,
    ) -> Result<Self, WhirConfigBuildError> {
        let config = profile.derive_config(num_variables)?;
        let domain_separator = WhirDomainSeparator::new(&config);
        Ok(Self {
            config,
            domain_separator,
            prover_data: None,
        })
    }

    /// The derived WHIR configuration.
    pub fn config(&self) -> &WhirConfig {
        &self.config
    }

    fn observe_domain_separator(&self, transcript: &mut PoseidonTranscript) {
        self.domain_separator.observe_into(transcript);
    }
}

/// Verifier-side parsed commitment: the Merkle root plus the replayed
/// commitment-phase OOD statement, mirroring upstream `PlainParsedCommitment`.
pub struct WhirParsedCommitment {
    root: PoseidonDigest,
    ood_statement: EqStatement,
}

impl DirectSparsePcs for WhirPcs {
    type Commitment = PoseidonDigest;
    type Proof = WhirProof;
    type ParsedCommitment = WhirParsedCommitment;
    type Error = WhirError;

    fn commit(
        &mut self,
        witness: &[KoalaBear],
        transcript: &mut PoseidonTranscript,
    ) -> Result<Self::Commitment, Self::Error> {
        if witness.len() != 1 << self.config.num_variables {
            return Err(WhirError::InvalidPolynomialLength {
                expected: 1 << self.config.num_variables,
                actual: witness.len(),
            });
        }
        self.observe_domain_separator(transcript);

        let (rows, tree) = commit_base_prefix(
            witness,
            self.config.round_folding_factor(0),
            self.config.params.starting_log_inv_rate,
        )?;
        let root = tree.root();
        transcript.observe_slice(&root);

        let mut initial_ood_answers = Vec::with_capacity(self.config.commitment_ood_samples);
        let mut ood_pairs = Vec::with_capacity(self.config.commitment_ood_samples);
        for _ in 0..self.config.commitment_ood_samples {
            let point =
                expand_from_univariate(transcript.sample_quintic(), self.config.num_variables);
            let eval = crate::whir::pcs::eval_mle_base_at_ext(witness, &point);
            transcript.observe_quintic(eval);
            initial_ood_answers.push(eval);
            ood_pairs.push((point, eval));
        }

        self.prover_data = Some(WhirProverData {
            polynomial: witness.to_vec(),
            tree,
            encoded_rows: rows,
            initial_ood_answers,
            ood_pairs,
            num_variables: self.config.num_variables,
        });
        Ok(root)
    }

    fn open(
        &mut self,
        point: &[QuinticExtension],
        value: QuinticExtension,
        transcript: &mut PoseidonTranscript,
    ) -> Result<Self::Proof, Self::Error> {
        let prover_data = self
            .prover_data
            .take()
            .ok_or(WhirError::MerkleProofInvalid)?;
        if point.len() != self.config.num_variables {
            return Err(WhirError::InvalidPolynomialLength {
                expected: self.config.num_variables,
                actual: point.len(),
            });
        }

        // Bind every claimed opening before the batching challenge is
        // sampled, mirroring upstream `observe_statement_point_claims`.
        observe_statement_point_claims(&[(point.to_vec(), value)], transcript);

        let mut claims = alloc::vec![(point.to_vec(), value)];
        claims.extend(prover_data.ood_pairs.iter().cloned());

        whir_prove(
            &self.config,
            prover_data.initial_ood_answers.clone(),
            claims,
            &prover_data.polynomial,
            prover_data.tree,
            prover_data.encoded_rows,
            transcript,
        )
    }

    fn verify_commitment(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        transcript: &mut PoseidonTranscript,
    ) -> Result<Self::ParsedCommitment, Self::Error> {
        self.observe_domain_separator(transcript);
        transcript.observe_slice(commitment);
        let ood_statement = parse_initial_ood(&self.config, proof, transcript)?;
        Ok(WhirParsedCommitment {
            root: *commitment,
            ood_statement,
        })
    }

    fn verify_opening(
        &self,
        parsed: &Self::ParsedCommitment,
        point: &[QuinticExtension],
        value: QuinticExtension,
        proof: &Self::Proof,
        transcript: &mut PoseidonTranscript,
    ) -> Result<(), Self::Error> {
        if point.len() != self.config.num_variables {
            return Err(WhirError::InvalidPolynomialLength {
                expected: self.config.num_variables,
                actual: point.len(),
            });
        }

        // Bind every claimed opening before the batching challenge,
        // mirroring upstream `verify_finalize`.
        observe_statement_point_claims(&[(point.to_vec(), value)], transcript);

        let mut eq_statement = EqStatement::initialize(self.config.num_variables);
        eq_statement.add_evaluated_constraint(point.to_vec(), value);
        for (ood_point, &ood_eval) in parsed.ood_statement.iter() {
            eq_statement.add_evaluated_constraint(ood_point.to_vec(), ood_eval);
        }
        let initial_constraint = Constraint::new(
            transcript.sample_quintic(),
            self.config.num_variables,
            alloc::vec![Statements::Eq(eq_statement)],
        );
        let mut claimed_eval = QuinticExtension::ZERO;
        initial_constraint.combine_evals(&mut claimed_eval);

        whir_verify(
            &self.config,
            proof,
            transcript,
            &parsed.root,
            initial_constraint,
            claimed_eval,
        )?;
        Ok(())
    }
}

/// Absorb the statement's claimed opening points and values, mirroring
/// upstream `observe_statement_point_claims`.
fn observe_statement_point_claims(
    claims: &[(Vec<QuinticExtension>, QuinticExtension)],
    transcript: &mut PoseidonTranscript,
) {
    transcript.observe(KoalaBear::from_u64(claims.len() as u64));
    for (point, value) in claims {
        transcript.observe_quintic_slice(point);
        transcript.observe_quintic(*value);
    }
}
