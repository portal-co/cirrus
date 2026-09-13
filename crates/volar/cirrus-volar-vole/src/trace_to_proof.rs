//! Canonical trace-to-Mode-B-proof artifact pipeline.
//!
//! This module is the `no_std` preparation boundary for a proof backend. It
//! validates a complete canonical Boolar trace with the transparent Mode-A
//! auditor, converts that audit to the backend-neutral Mode-B RAM witness,
//! constructs the relation and unified KoalaBear witness, and returns every
//! artifact a backend needs. It deliberately does **not** choose Fiat--Shamir
//! challenges, perform setup, or emit a backend proof.
//!
//! Storage-bearing circuits are prepared only when the caller supplies RAM
//! permutation challenges. Those values are suitable for differential-oracle
//! tests only until the challenge-slot/transcript schedule in
//! `trace-to-no-std-spartan-whir-plan.md` is implemented.

use alloc::vec::Vec;
use core::fmt;

use volar_ir::circuit::BCircuit;

use crate::{
    BoolarTraceAudit, CircuitId, MemoryAccess, ModeBPublicInstance, ModeBRelation,
    ModeBRelationError, PrimeRamPermutationChallenges, RamAccess, RamWitness, TraceAuditError,
    UnifiedR1cs, UnifiedR1csWitness,
};

/// Complete canonical trace and public statement accepted by the wrapper.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceToProofInput {
    /// Exhaustive authenticated trace in canonical wire order.
    pub audit: BoolarTraceAudit,
    /// Verifier-selected public inputs.
    pub public_inputs: Vec<bool>,
    /// Verifier-selected claimed outputs.
    pub claimed_outputs: Vec<bool>,
}

/// RAM challenge values for the current differential-oracle witness builder.
///
/// This is **not** a deployed proving lifecycle. The relation now contains
/// public challenge slots in a static R1CS shape; a real proof transcript must
/// derive these values after committing the relevant witness columns and the
/// verifier must independently recompute them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RamChallengeInput {
    /// The circuit is storage-free.
    NoStorage,
    /// Test-only externally supplied challenge-slot values.
    DifferentialOracle(PrimeRamPermutationChallenges),
}

/// Fully validated artifacts consumed by a proving backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceProofArtifacts {
    /// Structural circuit binding shared by every artifact.
    pub circuit_id: CircuitId,
    /// Backend-neutral Mode-B relation descriptor.
    pub relation: ModeBRelation,
    /// Canonical public statement.
    pub public_instance: ModeBPublicInstance,
    /// Unified signed-coefficient R1CS in the frozen variable order.
    pub unified: UnifiedR1cs,
    /// Canonical KoalaBear assignment for every unified variable.
    pub witness: UnifiedR1csWitness,
    /// Primary trace values in canonical wire order.
    pub trace_values: Vec<bool>,
    /// Canonical RAM witness used by storage-bearing circuits.
    pub ram_witness: Option<RamWitness>,
}

impl TraceProofArtifacts {
    /// Validate that all returned artifacts belong to `circuit`.
    ///
    /// Construction already validates the trace and relation; this method is
    /// for a backend that received artifacts from an untrusted serializer or
    /// another process and must reject mixed statements before setup/proving.
    pub fn validate_for_circuit<P: Clone>(
        &self,
        circuit: &BCircuit<P>,
    ) -> Result<(), TraceToProofError> {
        let expected_wire_count = circuit.params as usize + circuit.stmts.len();
        if self.trace_values.len() != expected_wire_count {
            return Err(TraceToProofError::WrongWireCount {
                expected: expected_wire_count,
                found: self.trace_values.len(),
            });
        }
        if self.public_instance.circuit_id != self.circuit_id
            || self.relation.circuit_id != self.circuit_id
            || self.unified.circuit_id != self.circuit_id
        {
            return Err(TraceToProofError::CircuitIdMismatch);
        }
        if self.public_instance.public_inputs.len() != circuit.params as usize
            || self.public_instance.claimed_outputs.len() != circuit.outputs.len()
        {
            return Err(TraceToProofError::WrongWireCount {
                expected: circuit.params as usize,
                found: self.public_instance.public_inputs.len(),
            });
        }
        if self.witness.values.len() != self.unified.variable_count {
            return Err(TraceToProofError::ModeB(
                ModeBRelationError::WrongUnifiedWitnessCount {
                    expected: self.unified.variable_count,
                    found: self.witness.values.len(),
                },
            ));
        }
        self.unified.evaluate_koalabear_witness(&self.witness)?;
        // The field witness is authoritative for proving, but trace values
        // are returned separately. Rebind them explicitly so a caller cannot
        // mix a valid witness with a stale or unrelated primary trace.
        self.relation.evaluate_bool(
            &self.trace_values,
            &self.public_instance.public_inputs,
            &self.public_instance.claimed_outputs,
        )?;
        let expected_circuit_id = ModeBRelation::from_boolar(circuit)?.circuit_id;
        if expected_circuit_id != self.circuit_id {
            return Err(TraceToProofError::CircuitIdMismatch);
        }
        match (self.unified.ram.is_some(), &self.ram_witness) {
            (false, None) => Ok(()),
            (false, Some(_)) => Err(TraceToProofError::ModeB(
                ModeBRelationError::UnexpectedRamWitness,
            )),
            (true, None) => Err(TraceToProofError::ModeB(
                ModeBRelationError::MissingRamWitness,
            )),
            (true, Some(ram)) => {
                let expected = RamWitness::from_boolar(circuit, &self.trace_values)?;
                if expected != *ram {
                    Err(TraceToProofError::ModeB(
                        ModeBRelationError::RamExecutionMismatch,
                    ))
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// Why a trace cannot be converted to proof artifacts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TraceToProofError {
    /// The transparent authenticated trace replay failed.
    Trace(TraceAuditError),
    /// The Mode-B relation or witness construction failed.
    ModeB(ModeBRelationError),
    /// The audit does not expose exactly one value per canonical wire.
    WrongWireCount {
        /// Required number of trace values.
        expected: usize,
        /// Number present.
        found: usize,
    },
    /// An artifact's circuit binding does not match the wrapper statement.
    CircuitIdMismatch,
    /// A Mode-A memory record uses a noncanonical or overwide address.
    InvalidMemoryAddress {
        /// Execution time of the offending access.
        time: usize,
    },
    /// The audit is missing the RAM witness required by the circuit.
    MissingMemoryAudit,
    /// A RAM witness was present for a storage-free circuit.
    UnexpectedMemoryAudit,
    /// A storage-bearing witness materialization requires challenge-slot
    /// values from the (currently test-only) caller.
    MissingRamChallenges,
    /// Challenges were supplied for a circuit without storage.
    UnexpectedRamChallenges,
}

impl fmt::Display for TraceToProofError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Trace(error) => write!(f, "trace audit failed: {error}"),
            Self::ModeB(error) => write!(f, "Mode-B artifact construction failed: {error}"),
            Self::WrongWireCount { expected, found } => {
                write!(f, "trace has {found} wire values, expected {expected}")
            }
            Self::CircuitIdMismatch => f.write_str("proof artifacts have mixed circuit IDs"),
            Self::InvalidMemoryAddress { time } => {
                write!(f, "memory access at time {time} has an invalid address")
            }
            Self::MissingMemoryAudit => f.write_str("storage circuit is missing its RAM audit"),
            Self::UnexpectedMemoryAudit => f.write_str("storage-free circuit received a RAM audit"),
            Self::MissingRamChallenges => {
                f.write_str("storage-bearing proof artifacts require RAM permutation challenges")
            }
            Self::UnexpectedRamChallenges => {
                f.write_str("storage-free proof artifacts received RAM challenges")
            }
        }
    }
}

impl core::error::Error for TraceToProofError {}

impl From<TraceAuditError> for TraceToProofError {
    fn from(value: TraceAuditError) -> Self {
        Self::Trace(value)
    }
}

impl From<ModeBRelationError> for TraceToProofError {
    fn from(value: ModeBRelationError) -> Self {
        Self::ModeB(value)
    }
}

/// Validate `input` against `circuit` and construct all backend-neutral proof
/// artifacts.
///
/// For a storage-bearing circuit, `ram_challenges` currently must be
/// [`RamChallengeInput::DifferentialOracle`]. The caller is responsible for
/// ensuring those values came from the appropriate transcript commitment in a
/// test harness; do not use prover-selected values in a deployed proof.
pub fn build_trace_proof_artifacts<P: Clone>(
    circuit: &BCircuit<P>,
    input: &TraceToProofInput,
    ram_challenges: RamChallengeInput,
) -> Result<TraceProofArtifacts, TraceToProofError> {
    // The transparent exhaustive replay is the reference semantics oracle. It
    // also proves that the openings are in canonical order and that public
    // inputs/outputs match the committed trace.
    input
        .audit
        .verify(circuit, &input.public_inputs, &input.claimed_outputs)?;

    let wire_count = circuit.params as usize + circuit.stmts.len();
    if input.audit.openings.len() != wire_count {
        return Err(TraceToProofError::WrongWireCount {
            expected: wire_count,
            found: input.audit.openings.len(),
        });
    }
    let trace_values = input
        .audit
        .openings
        .iter()
        .map(|opening| opening.value)
        .collect::<Vec<_>>();

    let ram_witness = ram_witness_from_audit(circuit, &input.audit)?;
    let challenges = match (&ram_witness, ram_challenges) {
        (None, RamChallengeInput::NoStorage) => None,
        (Some(_), RamChallengeInput::DifferentialOracle(challenges)) => Some(challenges),
        (None, RamChallengeInput::DifferentialOracle(_)) => {
            return Err(TraceToProofError::UnexpectedRamChallenges);
        }
        (Some(_), RamChallengeInput::NoStorage) => {
            return Err(TraceToProofError::MissingRamChallenges);
        }
    };

    let relation = ModeBRelation::from_boolar(circuit)?;
    relation.evaluate_bool_with_ram(
        circuit,
        &trace_values,
        &input.public_inputs,
        &input.claimed_outputs,
        ram_witness.as_ref(),
    )?;
    let public_instance = ModeBPublicInstance::new(
        &relation,
        input.public_inputs.clone(),
        input.claimed_outputs.clone(),
    );
    let unified = relation.export_unified_r1cs(circuit)?;
    let witness = relation.materialize_unified_witness(
        circuit,
        &trace_values,
        &input.public_inputs,
        &input.claimed_outputs,
        ram_witness.as_ref(),
        challenges.as_ref(),
    )?;
    unified.evaluate_koalabear_witness(&witness)?;

    Ok(TraceProofArtifacts {
        circuit_id: relation.circuit_id,
        relation,
        public_instance,
        unified,
        witness,
        trace_values,
        ram_witness,
    })
}

fn ram_witness_from_audit<P: Clone>(
    circuit: &BCircuit<P>,
    audit: &BoolarTraceAudit,
) -> Result<Option<RamWitness>, TraceToProofError> {
    let has_storage = !circuit.pre_init.is_empty()
        || circuit.stmts.iter().any(|statement| {
            matches!(
                statement.kind,
                volar_ir::boolar::BIrStmt::StorageRead { .. }
                    | volar_ir::boolar::BIrStmt::StorageWrite { .. }
            )
        });
    let Some(memory) = &audit.memory else {
        return if has_storage {
            Err(TraceToProofError::MissingMemoryAudit)
        } else {
            Ok(None)
        };
    };
    if !has_storage {
        return Err(TraceToProofError::UnexpectedMemoryAudit);
    }

    let execution = memory
        .execution
        .iter()
        .map(convert_memory_access)
        .collect::<Result<Vec<_>, _>>()?;
    let address_sorted = memory
        .address_sorted
        .iter()
        .map(convert_memory_access)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(RamWitness {
        execution,
        address_sorted,
    }))
}

fn convert_memory_access(access: &MemoryAccess) -> Result<RamAccess, TraceToProofError> {
    use crate::{PRIME_RAM_ADDRESS_BITS, RamAccessKind};

    if access.address.len() > PRIME_RAM_ADDRESS_BITS {
        return Err(TraceToProofError::InvalidMemoryAddress { time: access.time });
    }
    let mut address = access.address.clone();
    address.resize(PRIME_RAM_ADDRESS_BITS, false);
    Ok(RamAccess {
        storage: access.storage,
        lane: access.lane,
        address,
        time: access.time,
        kind: match access.kind {
            crate::MemoryAccessKind::Read => RamAccessKind::Read,
            crate::MemoryAccessKind::Write => RamAccessKind::Write,
        },
        value: access.value,
    })
}
