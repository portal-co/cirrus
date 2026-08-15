#![warn(missing_docs)]

//! Thin Groth16 setup/prove/verify wrappers over
//! [`cirrus_r1cs_backend::circuit::ProgramCircuit`]: a recorded
//! `cirrus_recompile_core::Program` becomes the arithmetic circuit Groth16
//! proves knowledge of a satisfying witness for.
//!
//! This crate is deliberately `std`-only (Groth16 setup/proving needs an
//! RNG); the R1CS-synthesis backend itself
//! (`cirrus-r1cs-backend`) stays `no_std`+`alloc`.

use ark_ec::pairing::Pairing;
use ark_ec::AdditiveGroup;
use ark_ff::Field;
pub use ark_groth16::{Proof, ProvingKey, VerifyingKey};
use ark_relations::gr1cs::SynthesisError;
use ark_snark::{CircuitSpecificSetupSNARK, SNARK};
use ark_std::rand::{CryptoRng, RngCore};
use cirrus_r1cs_backend::circuit::ProgramCircuit;
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
        .map(|&bit| if bit { E::ScalarField::ONE } else { E::ScalarField::ZERO })
        .collect();
    ark_groth16::Groth16::<E>::verify(vk, &public_inputs, proof)
}
