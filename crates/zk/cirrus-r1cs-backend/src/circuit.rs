//! A proof-system-agnostic [`ConstraintSynthesizer`] wrapper around a
//! recorded [`Program`], for use as the circuit type any `ark-relations`
//! SNARK's setup/prove entry points take (see `cirrus-groth16` for a
//! concrete Groth16 instantiation).

use alloc::vec::Vec;
use core::marker::PhantomData;

use ark_ff::PrimeField;
use ark_r1cs_std::{fields::fp::FpVar, prelude::*};
use ark_relations::gr1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use cirrus_core::{ContextWithMux, ContextWithStorage, StorageAddressBit};
use cirrus_recompile_core::Program;
use cirrus_volar_boolar::ExternalBitRegistry;
use volar_ir::boolar::LaneId;
use volar_ir_common::StorageId;

use crate::R1csBackend;

/// Pluggable constrained gadgets for external Boolean primitives.
///
/// The plugin is responsible for allocating and constraining every returned
/// bit. `occurrence` is the stable token emitted by Boolar lowering; RNG
/// plugins should treat it as a transcript/PRF coordinate, never as a request
/// for fresh prover-only entropy. The returned [`Boolean`] is therefore a
/// normal R1CS value that can feed later circuit gates.
pub trait ExternalPrimitiveGadgets<F: PrimeField> {
    /// Stable field elements that identify this primitive implementation and
    /// its parameterization. They are public inputs in
    /// [`ProgramCircuitWithPlugins`], constrained to these constants, so a
    /// proving key cannot be reused with a silently different plugin set.
    fn external_binding(&self) -> Vec<F>;

    /// Constrain and return one oracle output bit.
    fn oracle_bit(
        &self,
        cs: ConstraintSystemRef<F>,
        name: &str,
        args: &[Boolean<F>],
        bit: usize,
        occurrence: u64,
    ) -> Result<Boolean<F>, SynthesisError>;

    /// Constrain and return one replayable RNG output bit.
    fn rng_bit(
        &self,
        cs: ConstraintSystemRef<F>,
        name: &str,
        bit: usize,
        occurrence: u64,
    ) -> Result<Boolean<F>, SynthesisError>;

    /// Constrain and return one action result bit before its caller applies
    /// the action's direct storage effect.
    fn action_bit(
        &self,
        cs: ConstraintSystemRef<F>,
        name: &str,
        args: &[Boolean<F>],
        bit: usize,
        occurrence: u64,
    ) -> Result<Boolean<F>, SynthesisError>;
}

/// Plugin-supplied storage commitment layout and root relation.
///
/// Implementations define how the external action/storage state is committed
/// and may allocate any private storage witness they need. Both roots are
/// public field elements; the gadget must constrain them to the initial and
/// final storage witnesses under its declared layout.
pub trait StorageCommitmentGadget<F: PrimeField> {
    /// Stable field elements describing the commitment scheme and storage
    /// layout. As with external bindings, these are fixed public inputs.
    fn storage_layout_binding(&self) -> Vec<F>;

    /// Enforce the relation between the plugin's storage witness and the
    /// public initial/final storage roots.
    fn enforce_storage_roots(
        &self,
        cs: ConstraintSystemRef<F>,
        initial_root: &FpVar<F>,
        final_root: &FpVar<F>,
    ) -> Result<(), SynthesisError>;
}

/// A complete ZK plugin set: primitive gadgets plus a storage commitment
/// layout. Kept as one trait so setup/proving/verification bind exactly the
/// same ordered public statement prefix.
pub trait ZkPluginSet<F: PrimeField>:
    ExternalPrimitiveGadgets<F> + StorageCommitmentGadget<F>
{
}

/// Bridge a constrained external gadget plugin into Cirrus's Boolar executor.
///
/// Use this with `cirrus_volar_boolar::execute_with_externals` and an
/// [`R1csBackend`]. Oracle/RNG callback results are gadget-constrained, while
/// action callbacks select the produced bit or fallback under `guard` and
/// write it directly through the symbolic storage gadget.
pub struct R1csExternalBitRegistry<'a, P> {
    /// Caller-owned constrained primitive gadget implementation.
    pub plugins: &'a P,
}

impl<'a, P> R1csExternalBitRegistry<'a, P> {
    /// Wrap a gadget plugin for Boolar execution.
    pub fn new(plugins: &'a P) -> Self {
        Self { plugins }
    }
}

impl<F: PrimeField, P: ExternalPrimitiveGadgets<F>> ExternalBitRegistry<R1csBackend<F>>
    for R1csExternalBitRegistry<'_, P>
{
    fn oracle_bit(
        &mut self,
        context: &mut R1csBackend<F>,
        name: &str,
        args: &[Boolean<F>],
        bit: usize,
        occurrence: u64,
    ) -> Result<Boolean<F>, SynthesisError> {
        self.plugins
            .oracle_bit(context.cs.clone(), name, args, bit, occurrence)
    }

    fn rng_bit(
        &mut self,
        context: &mut R1csBackend<F>,
        name: &str,
        bit: usize,
        occurrence: u64,
    ) -> Result<Boolean<F>, SynthesisError> {
        self.plugins
            .rng_bit(context.cs.clone(), name, bit, occurrence)
    }

    fn action_store_bit(
        &mut self,
        context: &mut R1csBackend<F>,
        name: &str,
        guard: Boolean<F>,
        args: &[Boolean<F>],
        fallback: Boolean<F>,
        _storage: StorageId,
        _lane: LaneId,
        address: &[StorageAddressBit<Boolean<F>>],
        bit: usize,
        occurrence: u64,
        cells: &mut [Boolean<F>],
    ) -> Result<(), SynthesisError> {
        let action = self
            .plugins
            .action_bit(context.cs.clone(), name, args, bit, occurrence)?;
        let selected = context.mux(guard, action, fallback)?;
        context.storage_write(cells, address, selected)
    }
}

/// The same gadget adapter also resolves the external metadata carried by
/// Cirrus's recorded [`Program`] and [`cirrus_recompile_core::PreparedProgram`].
///
/// Unlike direct Boolar actions, a recorded program's action operation is a
/// value-producing bit only; its storage mutation has already been modeled
/// by the Boolar host that constructed the recorded program. The direct
/// Boolar adapter above remains the storage-owning path.
impl<F: PrimeField, P: ExternalPrimitiveGadgets<F>>
    cirrus_recompile_rt::ExternalRegistry<R1csBackend<F>> for R1csExternalBitRegistry<'_, P>
{
    fn oracle_bit(
        &mut self,
        context: &mut R1csBackend<F>,
        name: &str,
        args: &[Boolean<F>],
        bit: usize,
        occurrence: u64,
    ) -> Result<Boolean<F>, SynthesisError> {
        self.plugins
            .oracle_bit(context.cs.clone(), name, args, bit, occurrence)
    }

    fn rng_bit(
        &mut self,
        context: &mut R1csBackend<F>,
        name: &str,
        bit: usize,
        occurrence: u64,
    ) -> Result<Boolean<F>, SynthesisError> {
        self.plugins
            .rng_bit(context.cs.clone(), name, bit, occurrence)
    }

    fn action_bit(
        &mut self,
        context: &mut R1csBackend<F>,
        name: &str,
        args: &[Boolean<F>],
        bit: usize,
        occurrence: u64,
    ) -> Result<Boolean<F>, SynthesisError> {
        self.plugins
            .action_bit(context.cs.clone(), name, args, bit, occurrence)
    }
}

impl<F: PrimeField, T> ZkPluginSet<F> for T where
    T: ExternalPrimitiveGadgets<F> + StorageCommitmentGadget<F>
{
}

/// Public values used by a plugin-enabled proof.
///
/// The serialized field ordering is: external binding, storage-layout
/// binding, initial root, final root, then program outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginPublicStatement<F: PrimeField> {
    /// Plugin implementation and parameter binding.
    pub external_binding: Vec<F>,
    /// Storage commitment/layout binding.
    pub storage_layout_binding: Vec<F>,
    /// Public commitment to storage before execution.
    pub initial_storage_root: F,
    /// Public commitment to storage after execution.
    pub final_storage_root: F,
    /// Public Boolean program outputs in their declared order.
    pub public_outputs: Vec<bool>,
}

impl<F: PrimeField> PluginPublicStatement<F> {
    /// Flatten into the primary-input order expected by a SNARK verifier.
    pub fn to_field_elements(&self) -> Vec<F> {
        let mut fields = self.external_binding.clone();
        fields.extend_from_slice(&self.storage_layout_binding);
        fields.push(self.initial_storage_root);
        fields.push(self.final_storage_root);
        fields.extend(
            self.public_outputs
                .iter()
                .map(|bit| if *bit { F::ONE } else { F::ZERO }),
        );
        fields
    }
}

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

/// [`ProgramCircuit`] with a pluggable external-gadget and storage-root
/// statement prefix.
///
/// This preserves the portable `Program` execution path while giving a ZK
/// application one explicit, binding-checked place to attach constrained
/// external primitive relations and its own storage commitment layout.
pub struct ProgramCircuitWithPlugins<'a, F: PrimeField, P: ZkPluginSet<F>> {
    /// Recorded Boolean program to constrain.
    pub program: &'a Program,
    /// Private Boolean inputs for `program`.
    pub private_inputs: Option<&'a [bool]>,
    /// Public Boolean output values for `program`.
    pub public_outputs: Option<&'a [bool]>,
    /// Public root before all storage/action effects.
    pub initial_storage_root: Option<F>,
    /// Public root after all storage/action effects.
    pub final_storage_root: Option<F>,
    /// Constrained external primitive and storage commitment plugins.
    pub plugins: &'a P,
    /// Ties this circuit to its scalar field.
    pub _marker: PhantomData<F>,
}

impl<'a, F: PrimeField, P: ZkPluginSet<F>> ProgramCircuitWithPlugins<'a, F, P> {
    /// Form the exact public input statement needed by verification.
    pub fn public_statement(
        plugins: &P,
        initial_storage_root: F,
        final_storage_root: F,
        public_outputs: &[bool],
    ) -> PluginPublicStatement<F> {
        PluginPublicStatement {
            external_binding: plugins.external_binding(),
            storage_layout_binding: plugins.storage_layout_binding(),
            initial_storage_root,
            final_storage_root,
            public_outputs: public_outputs.to_vec(),
        }
    }
}

impl<'a, F: PrimeField, P: ZkPluginSet<F>> ConstraintSynthesizer<F>
    for ProgramCircuitWithPlugins<'a, F, P>
{
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
        let mut externals = R1csExternalBitRegistry::new(self.plugins);
        let computed = cirrus_recompile_rt::execute_with_externals(
            &mut backend,
            self.program,
            &witness_wires,
            &mut externals,
        )?;

        // The plugin/layout values are public yet fixed: equality to a
        // constant binds them into setup, proving, and verification alike.
        for binding in self
            .plugins
            .external_binding()
            .into_iter()
            .chain(self.plugins.storage_layout_binding())
        {
            let public = FpVar::new_input(cs.clone(), || Ok(binding))?;
            public.enforce_equal(&FpVar::Constant(binding))?;
        }
        let initial_root = FpVar::new_input(cs.clone(), || {
            self.initial_storage_root
                .ok_or(SynthesisError::AssignmentMissing)
        })?;
        let final_root = FpVar::new_input(cs.clone(), || {
            self.final_storage_root
                .ok_or(SynthesisError::AssignmentMissing)
        })?;
        self.plugins
            .enforce_storage_roots(cs.clone(), &initial_root, &final_root)?;

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
