//! Setup/prove/verify key API for the no-ZK DirectSparse profile.
//!
//! Mirrors the pinned upstream `spartan-whir` key lifecycle
//! (`SpartanProtocol::setup_with_config` with
//! `MatrixClosingMode::DirectSparse`, `prove`, and `verify`):
//!
//! - [`setup`] pads the shape, derives the composed per-component security
//!   levels ([`crate::security`]), derives the canonical WHIR configuration,
//!   and freezes the Spartan domain separator bytes.
//! - [`SpartanProvingKey::prove`] runs the Slice-4 DirectSparse driver over
//!   the Slice-5 WHIR PCS with a fresh transcript.
//! - [`SpartanVerifyingKey::verify`] re-authenticates the key (mirroring
//!   upstream's `validate_verifying_key`) and replays the verifier schedule
//!   against caller-supplied expected public values; public inputs bundled
//!   with a proof are never trusted.
//!
//! Profiles with transcript-derived public challenge slots (the Mode-B RAM
//! profile) use [`ChallengeSchedule`] with the `_with_schedule` entry
//! points. The schedule fires after the witness commitment has been
//! absorbed, so the challenge values are transcript-derived by construction
//! rather than caller-selected; see
//! [`crate::spartan::ChallengeSlotSchedule`] for the exact transcript
//! schedule. Binding the derived challenges to the committed RAM columns
//! additionally requires the two-phase commitment from the Mode-B storage
//! milestone; until that lands, slot-bearing profiles must not be used for
//! soundness claims.

use alloc::vec::Vec;
use core::fmt;

use crate::error::SpartanError;
use crate::r1cs::{R1csShape, R1csWitness};
use crate::security::{
    ComponentSecurity, ComposedSecurityBudget, SecurityBudgetError,
    derive_direct_component_security,
};
use crate::spartan::{
    ChallengeSlotSchedule, DirectSparseError, DirectSparseProof, R1csInstance,
    prove_direct_sparse_with_schedule, verify_direct_sparse_with_schedule,
};
use crate::whir::params::{FoldingFactor, SecurityAssumption, WhirConfig};
use crate::whir::pcs::WhirError;
use crate::whir::spartan::{
    SecurityProfile, WhirConfigBuildError, WhirFoldingSchedule, WhirParams, WhirPcs,
};
use crate::{KoalaBear, PoseidonDigest, PoseidonTranscript};

/// Protocol identifier bound into the no-ZK domain separator, mirroring
/// upstream `NO_ZK_PROTOCOL_ID`.
pub const SPARTAN_NO_ZK_PROTOCOL_ID: &[u8] = b"spartan-whir-no-zk-v0";

/// Version of the proof object layout. Versioning becomes load-bearing
/// with the Slice-7 serialization; proofs in memory are already typed.
pub const SPARTAN_PROOF_VERSION: u16 = 1;

/// Encode the Spartan domain separator bytes exactly as upstream
/// `DomainSeparator::to_bytes` does for the no-ZK DirectSparse profile.
pub fn spartan_domain_separator(
    shape: &R1csShape<KoalaBear>,
    profile: &SecurityProfile,
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(SPARTAN_NO_ZK_PROTOCOL_ID);
    out.push(0); // MatrixClosingMode::DirectSparse
    out.extend_from_slice(&(shape.num_cons as u64).to_le_bytes());
    out.extend_from_slice(&(shape.num_vars as u64).to_le_bytes());
    out.extend_from_slice(&(shape.num_io as u64).to_le_bytes());
    out.extend_from_slice(&profile.security_level_bits.to_le_bytes());
    out.extend_from_slice(&profile.merkle_security_bits.to_le_bytes());
    out.push(match profile.soundness {
        SecurityAssumption::UniqueDecoding => 0,
        SecurityAssumption::JohnsonBound => 1,
        SecurityAssumption::CapacityBound => 2,
    });
    encode_whir_params(&profile.whir, &mut out);
    out
}

/// Encode `WhirParams` exactly as upstream `encode_whir_params`.
fn encode_whir_params(params: &WhirParams, out: &mut Vec<u8>) {
    out.extend_from_slice(&params.pow_bits.to_le_bytes());
    out.extend_from_slice(&(params.folding_factor as u64).to_le_bytes());
    out.extend_from_slice(&(params.starting_log_inv_rate as u64).to_le_bytes());
    out.extend_from_slice(&(params.rs_domain_initial_reduction_factor as u64).to_le_bytes());
    if should_encode_schedule_suffix(params) {
        encode_folding_schedule(params, out);
    }
}

fn should_encode_schedule_suffix(params: &WhirParams) -> bool {
    if !params.round_log_inv_rates.is_empty() {
        return true;
    }
    match &params.folding_schedule {
        None => false,
        Some(WhirFoldingSchedule::Constant(factor)) => *factor != params.folding_factor,
        Some(_) => true,
    }
}

fn encode_folding_schedule(params: &WhirParams, out: &mut Vec<u8>) {
    match params.effective_folding_schedule() {
        WhirFoldingSchedule::Constant(factor) => {
            out.push(0);
            out.extend_from_slice(&(factor as u64).to_le_bytes());
        }
        WhirFoldingSchedule::ConstantFromSecondRound { first, rest } => {
            out.push(1);
            out.extend_from_slice(&(first as u64).to_le_bytes());
            out.extend_from_slice(&(rest as u64).to_le_bytes());
        }
        WhirFoldingSchedule::PerRound(factors) => {
            out.push(2);
            out.extend_from_slice(&(factors.len() as u64).to_le_bytes());
            for factor in factors {
                out.extend_from_slice(&(factor as u64).to_le_bytes());
            }
        }
    }
    out.extend_from_slice(&(params.round_log_inv_rates.len() as u64).to_le_bytes());
    for rate in &params.round_log_inv_rates {
        out.extend_from_slice(&(*rate as u64).to_le_bytes());
    }
}

/// A post-commitment challenge schedule for profiles whose public vector
/// ends in transcript-derived challenge slots.
///
/// The schedule fires after the witness PCS commitment has been absorbed
/// and before `tau` is sampled; both sides derive the slot values from the
/// transcript state, so the challenges are Fiat-Shamir outputs, never
/// caller-selected. The driver rejects the proof when the supplied public
/// tail differs from the derived values.
pub trait ChallengeSchedule {
    /// Number of trailing public-input slots derived after the commitment.
    fn challenge_slots(&self) -> usize {
        0
    }

    /// Derive the slot values from the post-commitment transcript state.
    /// Called only when [`Self::challenge_slots`] is nonzero.
    fn derive_challenges(&mut self, transcript: &mut PoseidonTranscript) -> Vec<KoalaBear>;
}

/// The empty schedule: no challenge slots, exact upstream transcript
/// schedule.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoChallengeSchedule;

impl ChallengeSchedule for NoChallengeSchedule {
    fn derive_challenges(&mut self, _transcript: &mut PoseidonTranscript) -> Vec<KoalaBear> {
        Vec::new()
    }
}

/// A DirectSparse Spartan proof with the WHIR PCS opening. The witness
/// commitment travels with the proof (mirroring upstream's `R1csInstance`);
/// the public inputs do not — the verifier always supplies the expected
/// public values itself.
#[derive(Clone, Debug, PartialEq)]
pub struct SpartanProof {
    /// PCS commitment to the zero-padded witness MLE.
    pub witness_commitment: PoseidonDigest,
    /// The DirectSparse reduction proof with the WHIR opening.
    pub proof: DirectSparseProof<WhirProof>,
}

/// The WHIR PCS proof type used by this profile.
pub use crate::whir::pcs::WhirProof;

/// Why setup, proving, or verification through the key API failed.
#[derive(Clone, Debug, PartialEq)]
pub enum SpartanKeyError {
    /// The R1CS shape or its canonical padding is invalid.
    InvalidShape,
    /// The security profile or WHIR parameter mapping is invalid.
    Profile(WhirConfigBuildError),
    /// The composed security budget cannot support the requested level.
    Security(SecurityBudgetError),
    /// A restored or hand-built verifying key is inconsistent.
    InvalidKey(&'static str),
    /// Proving or verification failed inside the protocol.
    Protocol(DirectSparseError<WhirError>),
}

impl fmt::Display for SpartanKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => f.write_str("invalid R1CS shape"),
            Self::Profile(error) => write!(f, "invalid security profile: {error}"),
            Self::Security(error) => write!(f, "security budget error: {error}"),
            Self::InvalidKey(reason) => write!(f, "invalid key: {reason}"),
            Self::Protocol(error) => write!(f, "protocol error: {error}"),
        }
    }
}

impl core::error::Error for SpartanKeyError {}

impl From<SecurityBudgetError> for SpartanKeyError {
    fn from(error: SecurityBudgetError) -> Self {
        Self::Security(error)
    }
}

impl From<WhirConfigBuildError> for SpartanKeyError {
    fn from(error: WhirConfigBuildError) -> Self {
        Self::Profile(error)
    }
}

/// Key fields shared by the proving and verifying keys.
#[derive(Clone, Debug, PartialEq)]
struct KeyCore {
    shape_canonical: R1csShape<KoalaBear>,
    num_cons_unpadded: usize,
    num_vars_unpadded: usize,
    num_io: usize,
    profile: SecurityProfile,
    component_security: ComponentSecurity,
    budget: ComposedSecurityBudget,
    pcs_config: WhirConfig,
    domain_separator: Vec<u8>,
}

impl KeyCore {
    fn validate(&self) -> Result<(), SpartanKeyError> {
        self.shape_canonical.validate().map_err(|_| {
            SpartanKeyError::InvalidKey("canonical shape fails validation")
        })?;
        self.profile
            .validate(self.pcs_config.num_variables)
            .map_err(SpartanKeyError::Profile)?;
        if self.shape_canonical.num_vars.ilog2() as usize != self.pcs_config.num_variables {
            return Err(SpartanKeyError::InvalidKey(
                "pcs config num_variables does not match the canonical shape",
            ));
        }
        if self.shape_canonical.num_cons != self.shape_canonical.num_cons.next_power_of_two()
            || self.shape_canonical.num_vars != self.shape_canonical.num_vars.next_power_of_two()
        {
            return Err(SpartanKeyError::InvalidKey(
                "canonical shape is not power-of-two padded",
            ));
        }

        // Recompute the composed component security from the requested
        // profile and the WHIR round count.
        let num_variables = self.pcs_config.num_variables;
        let folding = folding_factor_of(&self.profile.whir);
        let schedule = folding
            .compute_folding_schedule(num_variables)
            .map_err(|_| SpartanKeyError::InvalidKey("folding schedule does not compute"))?;
        let whir_rounds = schedule.len() - 1;
        let (component, budget) = derive_direct_component_security(
            self.profile.security_level_bits,
            self.profile.merkle_security_bits,
            whir_rounds,
            self.shape_canonical.num_cons.ilog2() as usize,
            num_variables + 1,
        )?;
        if component != self.component_security || budget != self.budget {
            return Err(SpartanKeyError::InvalidKey(
                "component security does not match the requested profile",
            ));
        }

        // Recompute the canonical WHIR configuration from the component
        // profile.
        let component_profile = SecurityProfile {
            security_level_bits: component.security_level_bits,
            merkle_security_bits: component.merkle_security_bits,
            soundness: self.profile.soundness,
            whir: self.profile.whir.clone(),
        };
        let pcs_config = component_profile
            .derive_config(num_variables)
            .map_err(SpartanKeyError::Profile)?;
        if pcs_config != self.pcs_config {
            return Err(SpartanKeyError::InvalidKey(
                "pcs config does not match the component profile",
            ));
        }

        // Recompute the domain separator bytes.
        let domain_separator = spartan_domain_separator(&self.shape_canonical, &self.profile);
        if domain_separator != self.domain_separator {
            return Err(SpartanKeyError::InvalidKey(
                "domain separator does not match the key fields",
            ));
        }
        Ok(())
    }
}

fn folding_factor_of(params: &WhirParams) -> FoldingFactor {
    match params.effective_folding_schedule() {
        WhirFoldingSchedule::Constant(factor) => FoldingFactor::Constant(factor),
        WhirFoldingSchedule::ConstantFromSecondRound { first, rest } => {
            FoldingFactor::ConstantFromSecondRound(first, rest)
        }
        WhirFoldingSchedule::PerRound(factors) => FoldingFactor::PerRound(factors),
    }
}

/// A shape- and profile-specific proving key.
#[derive(Clone, Debug, PartialEq)]
pub struct SpartanProvingKey {
    core: KeyCore,
}

/// A shape- and profile-specific verifying key.
#[derive(Clone, Debug, PartialEq)]
pub struct SpartanVerifyingKey {
    core: KeyCore,
}

/// Build the proving and verifying keys for a shape and security profile,
/// mirroring upstream `SpartanProtocol::setup_with_config` with
/// `MatrixClosingMode::DirectSparse`.
pub fn setup(
    shape: &R1csShape<KoalaBear>,
    profile: &SecurityProfile,
) -> Result<(SpartanProvingKey, SpartanVerifyingKey), SpartanKeyError> {
    shape.validate().map_err(|_| SpartanKeyError::InvalidShape)?;
    let shape_canonical = shape
        .pad_regular()
        .map_err(|_| SpartanKeyError::InvalidShape)?;
    let num_variables = shape_canonical.num_vars.ilog2() as usize;

    profile.validate(num_variables)?;

    let folding = folding_factor_of(&profile.whir);
    let schedule = folding
        .compute_folding_schedule(num_variables)
        .map_err(|_| SpartanKeyError::InvalidKey("folding schedule does not compute"))?;
    let whir_rounds = schedule.len() - 1;

    let (component_security, budget) = derive_direct_component_security(
        profile.security_level_bits,
        profile.merkle_security_bits,
        whir_rounds,
        shape_canonical.num_cons.ilog2() as usize,
        num_variables + 1,
    )?;

    let component_profile = SecurityProfile {
        security_level_bits: component_security.security_level_bits,
        merkle_security_bits: component_security.merkle_security_bits,
        soundness: profile.soundness,
        whir: profile.whir.clone(),
    };
    let pcs_config = component_profile.derive_config(num_variables)?;
    let domain_separator = spartan_domain_separator(&shape_canonical, profile);

    let core = KeyCore {
        shape_canonical,
        num_cons_unpadded: shape.num_cons,
        num_vars_unpadded: shape.num_vars,
        num_io: shape.num_io,
        profile: profile.clone(),
        component_security,
        budget,
        pcs_config,
        domain_separator,
    };
    Ok((
        SpartanProvingKey { core: core.clone() },
        SpartanVerifyingKey { core },
    ))
}

impl SpartanProvingKey {
    /// The canonical (padded) shape this key proves.
    pub fn shape_canonical(&self) -> &R1csShape<KoalaBear> {
        &self.core.shape_canonical
    }

    /// The unpadded witness length expected by [`Self::prove`].
    pub fn num_vars_unpadded(&self) -> usize {
        self.core.num_vars_unpadded
    }

    /// The number of public inputs expected by [`Self::prove`].
    pub fn num_io(&self) -> usize {
        self.core.num_io
    }

    /// The requested security profile.
    pub fn profile(&self) -> &SecurityProfile {
        &self.core.profile
    }

    /// The derived per-component security levels.
    pub fn component_security(&self) -> ComponentSecurity {
        self.core.component_security
    }

    /// The derived canonical WHIR configuration.
    pub fn pcs_config(&self) -> &WhirConfig {
        &self.core.pcs_config
    }

    /// The frozen Spartan domain separator bytes.
    pub fn domain_separator(&self) -> &[u8] {
        &self.core.domain_separator
    }

    /// Prove with the empty challenge schedule (the exact upstream
    /// transcript schedule).
    pub fn prove(
        &self,
        witness: &[KoalaBear],
        public_inputs: &[KoalaBear],
    ) -> Result<SpartanProof, SpartanKeyError> {
        self.prove_with_schedule(witness, public_inputs, &mut NoChallengeSchedule)
    }

    /// Prove with a post-commitment [`ChallengeSchedule`]. The schedule
    /// derives the trailing challenge-slot public values from the
    /// post-commitment transcript state; the supplied `public_inputs` tail
    /// must match the derived values.
    pub fn prove_with_schedule<S: ChallengeSchedule + ?Sized>(
        &self,
        witness: &[KoalaBear],
        public_inputs: &[KoalaBear],
        schedule: &mut S,
    ) -> Result<SpartanProof, SpartanKeyError> {
        let mut pcs = WhirPcs::from_config(self.core.pcs_config.clone());
        let mut transcript = PoseidonTranscript::new();
        let slots = schedule.challenge_slots();
        let mut derive = |transcript: &mut PoseidonTranscript| schedule.derive_challenges(transcript);
        let witness = R1csWitness {
            w: witness.to_vec(),
        };
        let (instance, proof) = prove_direct_sparse_with_schedule(
            &self.core.shape_canonical,
            &self.core.domain_separator,
            public_inputs,
            &witness,
            &mut pcs,
            &mut transcript,
            Some(ChallengeSlotSchedule {
                slots,
                derive: &mut derive,
            }),
        )
        .map_err(SpartanKeyError::Protocol)?;
        Ok(SpartanProof {
            witness_commitment: instance.witness_commitment,
            proof,
        })
    }
}

impl SpartanVerifyingKey {
    /// The canonical (padded) shape this key verifies.
    pub fn shape_canonical(&self) -> &R1csShape<KoalaBear> {
        &self.core.shape_canonical
    }

    /// The number of public inputs expected by [`Self::verify`].
    pub fn num_io(&self) -> usize {
        self.core.num_io
    }

    /// The requested security profile.
    pub fn profile(&self) -> &SecurityProfile {
        &self.core.profile
    }

    /// The derived per-component security levels.
    pub fn component_security(&self) -> ComponentSecurity {
        self.core.component_security
    }

    /// The derived canonical WHIR configuration.
    pub fn pcs_config(&self) -> &WhirConfig {
        &self.core.pcs_config
    }

    /// The frozen Spartan domain separator bytes.
    pub fn domain_separator(&self) -> &[u8] {
        &self.core.domain_separator
    }

    /// Re-authenticate the key fields, mirroring upstream
    /// `validate_verifying_key`: the canonical shape, the composed
    /// component security, the WHIR configuration, and the domain
    /// separator must all recompute to the stored values.
    pub fn validate_key(&self) -> Result<(), SpartanKeyError> {
        self.core.validate()
    }

    /// Verify with the empty challenge schedule against caller-supplied
    /// expected public values.
    pub fn verify(
        &self,
        expected_public_inputs: &[KoalaBear],
        proof: &SpartanProof,
    ) -> Result<(), SpartanKeyError> {
        self.verify_with_schedule(expected_public_inputs, proof, &mut NoChallengeSchedule)
    }

    /// Verify with a post-commitment [`ChallengeSchedule`]. The verifier
    /// recomputes the challenge slots independently and rejects the proof
    /// when the supplied public vector differs from the derived vector.
    pub fn verify_with_schedule<S: ChallengeSchedule + ?Sized>(
        &self,
        expected_public_inputs: &[KoalaBear],
        proof: &SpartanProof,
        schedule: &mut S,
    ) -> Result<(), SpartanKeyError> {
        self.validate_key()?;
        let pcs = WhirPcs::from_config(self.core.pcs_config.clone());
        let mut transcript = PoseidonTranscript::new();
        let instance = R1csInstance {
            public_inputs: expected_public_inputs.to_vec(),
            witness_commitment: proof.witness_commitment,
        };
        let slots = schedule.challenge_slots();
        let mut derive = |transcript: &mut PoseidonTranscript| schedule.derive_challenges(transcript);
        verify_direct_sparse_with_schedule(
            &self.core.shape_canonical,
            &self.core.domain_separator,
            &instance,
            &proof.proof,
            &pcs,
            &mut transcript,
            Some(ChallengeSlotSchedule {
                slots,
                derive: &mut derive,
            }),
        )
        .map_err(SpartanKeyError::Protocol)
    }
}

/// Convenience re-export for the no-schedule error conversion used by the
/// DirectSparse driver.
impl From<SpartanError> for SpartanKeyError {
    fn from(error: SpartanError) -> Self {
        Self::Protocol(DirectSparseError::Spartan(error))
    }
}
