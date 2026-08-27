#![warn(missing_docs)]

//! Thin Groth16 setup/prove/verify wrappers over
//! [`cirrus_r1cs_backend::circuit::ProgramCircuit`]: a recorded
//! `cirrus_recompile_core::Program` becomes the arithmetic circuit Groth16
//! proves knowledge of a satisfying witness for.
//!
//! This crate is deliberately `std`-only (Groth16 setup/proving needs an
//! RNG); the R1CS-synthesis backend itself
//! (`cirrus-r1cs-backend`) stays `no_std`+`alloc`.

use ark_ec::AdditiveGroup;
use ark_ec::pairing::Pairing;
use ark_ff::Field;
pub use ark_groth16::{Proof, ProvingKey, VerifyingKey};
use ark_relations::gr1cs::SynthesisError;
use ark_snark::{CircuitSpecificSetupSNARK, SNARK};
use ark_std::rand::{CryptoRng, RngCore};
use cirrus_r1cs_backend::circuit::{
    PluginPublicStatement, ProgramCircuit, ProgramCircuitWithPlugins, ZkPluginSet,
};
use cirrus_recompile_core::Program;
use core::marker::PhantomData;

/// Derive a Groth16 proving/verifying key pair from `program`'s shape.
///
/// This is a one-time, per-circuit setup: it does not take a witness, only
/// `program` itself (see [`ProgramCircuit`]'s `None`-during-setup
/// convention).
pub fn setup<E: Pairing, R: RngCore + CryptoRng>(
    program: &Program,
    rng: &mut R,
) -> Result<(ProvingKey<E>, VerifyingKey<E>), SynthesisError> {
    let circuit = ProgramCircuit::<E::ScalarField> {
        program,
        private_inputs: None,
        public_outputs: None,
        _marker: PhantomData,
    };
    ark_groth16::Groth16::<E>::setup(circuit, rng)
}

/// Prove that `private_inputs` makes `program` compute `public_outputs`.
pub fn prove<E: Pairing, R: RngCore + CryptoRng>(
    pk: &ProvingKey<E>,
    program: &Program,
    private_inputs: &[bool],
    public_outputs: &[bool],
    rng: &mut R,
) -> Result<Proof<E>, SynthesisError> {
    let circuit = ProgramCircuit::<E::ScalarField> {
        program,
        private_inputs: Some(private_inputs),
        public_outputs: Some(public_outputs),
        _marker: PhantomData,
    };
    ark_groth16::Groth16::<E>::prove(pk, circuit, rng)
}

/// Verify a proof that some private input makes `program` compute
/// `public_outputs`.
///
/// `public_outputs` is converted `bool -> E::ScalarField` (`ONE`/`ZERO`),
/// matching how [`ProgramCircuit`] allocates each output as a
/// `Boolean::new_input`.
pub fn verify<E: Pairing>(
    vk: &VerifyingKey<E>,
    public_outputs: &[bool],
    proof: &Proof<E>,
) -> Result<bool, SynthesisError> {
    let public_inputs: Vec<E::ScalarField> = public_outputs
        .iter()
        .map(|&bit| {
            if bit {
                E::ScalarField::ONE
            } else {
                E::ScalarField::ZERO
            }
        })
        .collect();
    ark_groth16::Groth16::<E>::verify(vk, &public_inputs, proof)
}

/// Derive Groth16 keys with constrained external primitive gadgets and a
/// plugin-supplied storage commitment layout.
///
/// The plugin's external and layout bindings are public, constant-constrained
/// prefix values, so this setup key is tied to that exact configuration.
pub fn setup_with_plugins<E: Pairing, R: RngCore + CryptoRng, P: ZkPluginSet<E::ScalarField>>(
    program: &Program,
    plugins: &P,
    rng: &mut R,
) -> Result<(ProvingKey<E>, VerifyingKey<E>), SynthesisError> {
    let circuit = ProgramCircuitWithPlugins::<E::ScalarField, P> {
        program,
        private_inputs: None,
        public_outputs: None,
        initial_storage_root: None,
        final_storage_root: None,
        plugins,
        _marker: PhantomData,
    };
    ark_groth16::Groth16::<E>::setup(circuit, rng)
}

/// Prove a plugin-bound statement with public initial and final storage roots.
pub fn prove_with_plugins<E: Pairing, R: RngCore + CryptoRng, P: ZkPluginSet<E::ScalarField>>(
    pk: &ProvingKey<E>,
    program: &Program,
    plugins: &P,
    private_inputs: &[bool],
    initial_storage_root: E::ScalarField,
    final_storage_root: E::ScalarField,
    public_outputs: &[bool],
    rng: &mut R,
) -> Result<Proof<E>, SynthesisError> {
    let circuit = ProgramCircuitWithPlugins::<E::ScalarField, P> {
        program,
        private_inputs: Some(private_inputs),
        public_outputs: Some(public_outputs),
        initial_storage_root: Some(initial_storage_root),
        final_storage_root: Some(final_storage_root),
        plugins,
        _marker: PhantomData,
    };
    ark_groth16::Groth16::<E>::prove(pk, circuit, rng)
}

/// Construct the public plugin-bound statement in verifier input order.
pub fn plugin_public_statement<E: Pairing, P: ZkPluginSet<E::ScalarField>>(
    plugins: &P,
    initial_storage_root: E::ScalarField,
    final_storage_root: E::ScalarField,
    public_outputs: &[bool],
) -> PluginPublicStatement<E::ScalarField> {
    ProgramCircuitWithPlugins::public_statement(
        plugins,
        initial_storage_root,
        final_storage_root,
        public_outputs,
    )
}

/// Verify a proof with the exact plugin and storage-layout public prefix used
/// for setup and proving.
pub fn verify_with_plugins<E: Pairing, P: ZkPluginSet<E::ScalarField>>(
    vk: &VerifyingKey<E>,
    plugins: &P,
    initial_storage_root: E::ScalarField,
    final_storage_root: E::ScalarField,
    public_outputs: &[bool],
    proof: &Proof<E>,
) -> Result<bool, SynthesisError> {
    let statement = plugin_public_statement::<E, P>(
        plugins,
        initial_storage_root,
        final_storage_root,
        public_outputs,
    );
    ark_groth16::Groth16::<E>::verify(vk, &statement.to_field_elements(), proof)
}
