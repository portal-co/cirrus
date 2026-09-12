//! `spartan-whir` conversion boundary for Mode B.
//!
//! This module is deliberately feature-gated: the core VOLE crate remains
//! `no_std`, while the upstream proving implementation is a std-only
//! application dependency.  It only performs the frozen R1CS/witness/public
//! vector conversion.  In particular, deriving RAM permutation challenges
//! from committed RAM columns belongs to the application transcript lifecycle
//! and is not performed here.

use alloc::{vec, vec::Vec};

use p3_field::PrimeCharacteristicRing;
use spartan_whir::{R1csShape, R1csWitness, SparseMatEntry, SparseMatrix, engine::F};

use crate::{ModeBRelationError, SpartanWhirMatrixEntry, SpartanWhirR1csShape, UnifiedR1csWitness};

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
