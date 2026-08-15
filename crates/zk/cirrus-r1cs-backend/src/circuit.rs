//! A proof-system-agnostic [`ConstraintSynthesizer`] wrapper around a
//! recorded [`Program`], for use as the circuit type any `ark-relations`
//! SNARK's setup/prove entry points take (see `cirrus-groth16` for a
//! concrete Groth16 instantiation).

use alloc::vec::Vec;
use core::marker::PhantomData;

use ark_ff::PrimeField;
use ark_r1cs_std::prelude::*;
use ark_relations::gr1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use cirrus_recompile_core::Program;

use crate::R1csBackend;

/// Replays `program` through a fresh [`R1csBackend`], proving "I know
/// inputs that make `program` compute these public outputs."
///
/// `program.inputs` slots are allocated as private witnesses;
/// `program.outputs` slots are allocated as public instance variables and
/// constrained equal to the wires `program` actually computes.
pub struct ProgramCircuit<'a, F: PrimeField> {
    /// The recorded program to replay as R1CS constraints.
    pub program: &'a Program,
    /// The private witness for `program.inputs`, one bit per input slot in
    /// order. `None` during a SNARK's setup pass: `ark-relations` never
    /// invokes an allocation closure while `cs.is_in_setup_mode()`, so a
    /// `None` witness during setup never triggers `AssignmentMissing` --
    /// setup only needs the circuit's shape, never real values.
    pub private_inputs: Option<&'a [bool]>,
    /// The public output values for `program.outputs`, one bit per output
    /// slot in order. Same `None`-during-setup / `Some`-during-proving
    /// convention as `private_inputs`.
    pub public_outputs: Option<&'a [bool]>,
    /// Ties this circuit to a specific scalar field without needing an
    /// owned `F` value.
    pub _marker: PhantomData<F>,
}

impl<'a, F: PrimeField> ConstraintSynthesizer<F> for ProgramCircuit<'a, F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let private_inputs = self.private_inputs;
        let witness_wires: Vec<Boolean<F>> = (0..self.program.inputs.len())
            .map(|i| {
                Boolean::new_witness(cs.clone(), || {
                    private_inputs
                        .map(|inputs| inputs[i])
                        .ok_or(SynthesisError::AssignmentMissing)
                })
            })
            .collect::<Result<_, _>>()?;

        let mut backend = R1csBackend::new(cs.clone());
        let computed = cirrus_recompile_rt::execute(&mut backend, self.program, &witness_wires)?;

        let public_outputs = self.public_outputs;
        for (i, computed_bit) in computed.iter().enumerate() {
            let public_bit = Boolean::new_input(cs.clone(), || {
                public_outputs
                    .map(|outputs| outputs[i])
                    .ok_or(SynthesisError::AssignmentMissing)
            })?;
            computed_bit.enforce_equal(&public_bit)?;
        }
        Ok(())
    }
}
