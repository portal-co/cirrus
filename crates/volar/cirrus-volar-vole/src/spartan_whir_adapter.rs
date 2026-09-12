//! `spartan-whir` conversion boundary for Mode B.
//!
//! This module is deliberately feature-gated: the core VOLE crate remains
//! `no_std`, while the upstream proving implementation is a std-only
//! application dependency.  It only performs the frozen R1CS/witness/public
//! vector conversion.  In particular, deriving RAM permutation challenges
//! from committed RAM columns belongs to the application transcript lifecycle
//! and is not performed here.

use alloc::{vec, vec::Vec};
use core::fmt;

use p3_field::PrimeCharacteristicRing;
use spartan_whir::{
    MatrixClosingMode, PoseidonProof, PoseidonProvingKey, PoseidonSetupConfig,
    PoseidonVerifyingKey, QuinticExtension, R1csShape, R1csWitness, SecurityConfig,
    SoundnessAssumption, SparseMatEntry, SparseMatrix, SpartanSnarkConfig, SpartanWhirError,
    WhirParams, engine::F, recommended_quintic_whir_params,
};

use crate::{
    ModeBRelationError, SpartanWhirMatrixEntry, SpartanWhirR1csShape, TraceProofArtifacts,
    UnifiedR1csWitness,
};

/// A `spartan-whir` R1CS shape together with the Mode-B circuit binding that
/// selected it.
#[derive(Clone, Debug)]
pub struct SpartanWhirAdapterShape {
    /// Structural Mode-B circuit binding. Keep this alongside setup/proving
    /// keys; upstream keys are shape-specific but do not expose this ID.
    pub circuit_id: crate::CircuitId,
    /// Upstream sparse R1CS shape in `[ private | one | public ]` order.
    pub shape: R1csShape<F>,
    /// Unified wire indices in upstream public-vector order.
    pub public_wires: Vec<usize>,
}

/// The witness and verifier-selected public values for one adapted instance.
#[derive(Clone, Debug)]
pub struct SpartanWhirAdapterWitness {
    /// Private witness columns only.
    pub witness: R1csWitness<F>,
    /// Public values in `claimed_outputs || public_inputs` order.
    ///
    /// This vector must be independently supplied by the verifier to upstream
    /// verification; it must never be trusted merely because a proof carries a
    /// copy of it.
    pub public_values: Vec<F>,
}

/// Security and PCS profile for the initial no-ZK trace-to-proof lifecycle.
///
/// The default is the explicit 80-bit CapacityBound profile used by upstream
/// phase-3 tests. It is convenient for integration testing, but it is **not**
/// a production-security recommendation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceProofSecurityProfile {
    /// Spartan/Merkle target and soundness assumption.
    pub security: SecurityConfig,
    /// Plain-WHIR parameters selected for the padded witness size.
    pub whir_params: WhirParams,
}

impl TraceProofSecurityProfile {
    /// Explicit test profile matching upstream phase-3 no-ZK fixtures.
    pub fn capacity_bound_80_test() -> Self {
        Self {
            security: SecurityConfig {
                security_level_bits: 80,
                merkle_security_bits: 80,
                soundness_assumption: SoundnessAssumption::CapacityBound,
            },
            whir_params: WhirParams {
                pow_bits: 0,
                folding_factor: 1,
                starting_log_inv_rate: 6,
                rs_domain_initial_reduction_factor: 1,
                ..WhirParams::default()
            },
        }
    }

    /// Upstream-recommended quintic parameters for `num_variables`.
    pub fn recommended_quintic(num_variables: usize, security: SecurityConfig) -> Self {
        Self {
            security,
            whir_params: recommended_quintic_whir_params(num_variables),
        }
    }
}

/// A proving key and the verifying key bound to the same Mode-B circuit ID.
pub struct SpartanWhirTraceKeys {
    /// Structural Mode-B circuit binding.
    pub circuit_id: crate::CircuitId,
    /// Upstream circuit-specific proving key.
    pub proving: PoseidonProvingKey<QuinticExtension>,
    /// Upstream circuit-specific verifying key.
    pub verifying: PoseidonVerifyingKey<QuinticExtension>,
}

/// One no-ZK Poseidon/Quintic Spartan-WHIR proof plus its circuit binding.
pub struct SpartanWhirTraceProof {
    /// Structural Mode-B circuit binding.
    pub circuit_id: crate::CircuitId,
    /// Upstream proof object. Its embedded public-input vector is an untrusted
    /// copy; verification always receives expected public values separately.
    pub proof: PoseidonProof<QuinticExtension>,
}

/// Why trace artifacts cannot enter the initial upstream proving lifecycle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TraceProofBackendError {
    /// The initial lifecycle supports only storage-free circuits.
    StorageRequiresChallengeSlots,
    /// Upstream setup, proving, or verification failed.
    Upstream(&'static str),
}

impl fmt::Display for TraceProofBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StorageRequiresChallengeSlots => {
                f.write_str("storage-bearing trace proofs require transcript challenge slots")
            }
            Self::Upstream(operation) => write!(f, "upstream spartan-whir {operation} failed"),
        }
    }
}

impl core::error::Error for TraceProofBackendError {}

impl From<SpartanWhirError> for TraceProofBackendError {
    fn from(_: SpartanWhirError) -> Self {
        // Keep the no_std-facing error small and non-String; callers needing
        // diagnostics can call upstream directly.
        Self::Upstream("operation")
    }
}

impl TraceProofArtifacts {
    /// Convert validated no-storage artifacts to upstream keys.
    pub fn setup_spartan_whir_keys(
        &self,
        profile: &TraceProofSecurityProfile,
    ) -> Result<SpartanWhirTraceKeys, TraceProofBackendError> {
        if self.unified.ram.is_some() || self.ram_witness.is_some() {
            return Err(TraceProofBackendError::StorageRequiresChallengeSlots);
        }
        let adapter = self
            .unified
            .clone()
            .lower_koalabear()
            .and_then(|lowered| lowered.export_spartan_whir_shape())
            .map_err(|_| TraceProofBackendError::Upstream("shape export"))?
            .to_spartan_whir_adapter();
        adapter
            .validate()
            .map_err(|_| TraceProofBackendError::Upstream("shape validation"))?;
        let num_variables = adapter.shape.num_vars.next_power_of_two().ilog2() as usize;
        let config: PoseidonSetupConfig = SpartanSnarkConfig {
            matrix_closing: MatrixClosingMode::DirectSparse,
            security: profile.security,
            whir_params: if profile.whir_params == WhirParams::default() {
                recommended_quintic_whir_params(num_variables)
            } else {
                profile.whir_params.clone()
            },
            spark_whir_params: None,
        };
        let (proving, verifying) =
            PoseidonProvingKey::<QuinticExtension>::setup(adapter.shape.clone(), config)
                .map_err(|_| TraceProofBackendError::Upstream("setup"))?;
        Ok(SpartanWhirTraceKeys {
            circuit_id: adapter.circuit_id,
            proving,
            verifying,
        })
    }

    /// Prove a validated no-storage trace with the initial upstream backend.
    pub fn prove_spartan_whir(
        &self,
        keys: &SpartanWhirTraceKeys,
    ) -> Result<SpartanWhirTraceProof, TraceProofBackendError> {
        if keys.circuit_id != self.circuit_id {
            return Err(TraceProofBackendError::Upstream("circuit binding"));
        }
        let adapter = self
            .unified
            .clone()
            .lower_koalabear()
            .and_then(|lowered| lowered.export_spartan_whir_shape())
            .map_err(|_| TraceProofBackendError::Upstream("shape export"))?
            .to_spartan_whir_adapter();
        let split = adapter
            .split_witness(&self.witness)
            .map_err(|_| TraceProofBackendError::Upstream("witness split"))?;
        let proof = keys
            .proving
            .prove(split.witness, split.public_values)
            .map_err(|_| TraceProofBackendError::Upstream("prove"))?;
        Ok(SpartanWhirTraceProof {
            circuit_id: self.circuit_id,
            proof,
        })
    }
}

impl SpartanWhirTraceKeys {
    /// Verify a proof against verifier-selected public values.
    pub fn verify(
        &self,
        expected_public_values: &[F],
        proof: &SpartanWhirTraceProof,
    ) -> Result<(), TraceProofBackendError> {
        if proof.circuit_id != self.circuit_id {
            return Err(TraceProofBackendError::Upstream("circuit binding"));
        }
        self.verifying
            .verify(expected_public_values, &proof.proof)
            .map_err(|_| TraceProofBackendError::Upstream("verify"))
    }
}

impl SpartanWhirR1csShape {
    /// Convert this frozen, dependency-free export to the upstream Rust type.
    pub fn to_spartan_whir_adapter(&self) -> SpartanWhirAdapterShape {
        let columns = self.witness_count + 1 + self.public_input_count;
        SpartanWhirAdapterShape {
            circuit_id: self.circuit_id,
            shape: R1csShape {
                num_cons: self.constraint_count,
                num_vars: self.witness_count,
                num_io: self.public_input_count,
                a: convert_matrix(&self.a, self.constraint_count, columns),
                b: convert_matrix(&self.b, self.constraint_count, columns),
                c: convert_matrix(&self.c, self.constraint_count, columns),
            },
            public_wires: self.public_wires.clone(),
        }
    }
}

impl SpartanWhirAdapterShape {
    /// Validate this converted shape with the upstream implementation before
    /// circuit-specific setup.
    pub fn validate(&self) -> Result<(), spartan_whir::SpartanWhirError> {
        self.shape.validate()
    }

    /// Split a complete unified assignment into the exact upstream witness and
    /// public-vector layout, validating the assignment size and matrix shape.
    pub fn split_witness(
        &self,
        unified: &UnifiedR1csWitness,
    ) -> Result<SpartanWhirAdapterWitness, ModeBRelationError> {
        let expected = self.shape.num_vars.checked_add(self.shape.num_io).ok_or(
            ModeBRelationError::WrongUnifiedWitnessCount {
                expected: usize::MAX,
                found: unified.values.len(),
            },
        )?;
        if unified.values.len() != expected {
            return Err(ModeBRelationError::WrongUnifiedWitnessCount {
                expected,
                found: unified.values.len(),
            });
        }

        let mut is_public = vec![false; unified.values.len()];
        for &wire in &self.public_wires {
            let Some(public) = is_public.get_mut(wire) else {
                return Err(ModeBRelationError::WrongUnifiedWitnessCount {
                    expected,
                    found: unified.values.len(),
                });
            };
            if *public {
                return Err(ModeBRelationError::DuplicatePublicWire { wire });
            }
            *public = true;
        }

        let witness = unified
            .values
            .iter()
            .enumerate()
            .filter_map(|(wire, &value)| (!is_public[wire]).then_some(F::from_u32(value)))
            .collect();
        let public_values = self
            .public_wires
            .iter()
            .map(|&wire| F::from_u32(unified.values[wire]))
            .collect();

        Ok(SpartanWhirAdapterWitness {
            witness: R1csWitness { w: witness },
            public_values,
        })
    }
}

fn convert_matrix(
    entries: &[SpartanWhirMatrixEntry],
    rows: usize,
    columns: usize,
) -> SparseMatrix<F> {
    SparseMatrix {
        num_rows: rows,
        num_cols: columns,
        entries: entries
            .iter()
            .map(|entry| SparseMatEntry {
                row: entry.row,
                col: entry.column,
                val: F::from_u32(entry.value),
            })
            .collect(),
    }
}
