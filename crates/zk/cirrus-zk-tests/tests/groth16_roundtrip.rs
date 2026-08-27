//! End-to-end Groth16 round-trip over [`sample_program`]: setup -> prove ->
//! verify succeeds for the honest witness/outputs, and fails once the
//! public outputs handed to `verify` are corrupted.

use ark_bn254::Bn254;
use ark_ff::PrimeField;
use ark_r1cs_std::{eq::EqGadget, fields::fp::FpVar, prelude::Boolean};
use ark_relations::gr1cs::{ConstraintSystemRef, SynthesisError};
use ark_std::rand::{SeedableRng, rngs::StdRng};
use cirrus_core::ContextWithCreate;
use cirrus_r1cs_backend::circuit::{ExternalPrimitiveGadgets, StorageCommitmentGadget};
use cirrus_recompile_core::{ExternalKind, ExternalOp, Program, Recorder};
use cirrus_zk_tests::sample_program;

struct TestPlugins {
    binding: u64,
}

impl<F: PrimeField> ExternalPrimitiveGadgets<F> for TestPlugins {
    fn external_binding(&self) -> Vec<F> {
        vec![F::from(self.binding)]
    }

    fn oracle_bit(
        &self,
        _cs: ConstraintSystemRef<F>,
        _name: &str,
        args: &[Boolean<F>],
        _bit: usize,
        _occurrence: u64,
    ) -> Result<Boolean<F>, SynthesisError> {
        Ok(args[0].clone())
    }

    fn rng_bit(
        &self,
        _cs: ConstraintSystemRef<F>,
        _name: &str,
        _bit: usize,
        _occurrence: u64,
    ) -> Result<Boolean<F>, SynthesisError> {
        Ok(Boolean::constant(true))
    }

    fn action_bit(
        &self,
        _cs: ConstraintSystemRef<F>,
        _name: &str,
        args: &[Boolean<F>],
        _bit: usize,
        _occurrence: u64,
    ) -> Result<Boolean<F>, SynthesisError> {
        Ok(&args[0] & &args[1])
    }
}

fn external_program() -> Program {
    let mut recorder = Recorder::new();
    let input = recorder.create(false).unwrap();
    let oracle = recorder.external_bit(ExternalOp {
        kind: ExternalKind::Oracle,
        name: "lookup".into(),
        args: vec![input],
        bit: 0,
        occurrence: 1,
    });
    let rng = recorder.external_bit(ExternalOp {
        kind: ExternalKind::Rng,
        name: "nonce".into(),
        args: vec![],
        bit: 0,
        occurrence: 2,
    });
    let action = recorder.external_bit(ExternalOp {
        kind: ExternalKind::Action,
        name: "commit".into(),
        args: vec![oracle, rng],
        bit: 0,
        occurrence: 3,
    });
    recorder.finish(vec![input], vec![action])
}

impl<F: PrimeField> StorageCommitmentGadget<F> for TestPlugins {
    fn storage_layout_binding(&self) -> Vec<F> {
        vec![F::from(0x51u64)]
    }

    fn enforce_storage_roots(
        &self,
        _cs: ConstraintSystemRef<F>,
        initial_root: &FpVar<F>,
        final_root: &FpVar<F>,
    ) -> Result<(), SynthesisError> {
        // A tiny test layout: the no-op program leaves its root unchanged.
        initial_root.enforce_equal(final_root)
    }
}

#[test]
fn setup_prove_verify_round_trips_and_rejects_a_corrupted_output() {
    let program = sample_program();
    let mut rng = StdRng::seed_from_u64(0xC1_2C_55);

    let (pk, vk) = cirrus_groth16::setup::<Bn254, _>(&program, &mut rng)
        .expect("setup only needs the circuit's shape");

    let private_inputs = [false, true, true, false]; // zero, one, a=true, b=false
    let public_outputs = cirrus_recompile_core::interpret(&program, &private_inputs);

    let proof = cirrus_groth16::prove::<Bn254, _>(
        &pk,
        &program,
        &private_inputs,
        &public_outputs,
        &mut rng,
    )
    .expect("the honest witness satisfies every constraint");

    assert!(
        cirrus_groth16::verify::<Bn254>(&vk, &public_outputs, &proof).unwrap(),
        "an honest proof against its own public outputs must verify"
    );

    let mut corrupted_outputs = public_outputs.clone();
    corrupted_outputs[0] = !corrupted_outputs[0];
    assert!(
        !cirrus_groth16::verify::<Bn254>(&vk, &corrupted_outputs, &proof).unwrap(),
        "a proof must not verify against a corrupted public output"
    );
}

#[test]
fn plugin_bound_roots_round_trip_and_reject_layout_or_root_changes() {
    let program = external_program();
    let plugins = TestPlugins { binding: 0xAA };
    let mut rng = StdRng::seed_from_u64(0x51_0A);
    let (pk, vk) = cirrus_groth16::setup_with_plugins::<Bn254, _, _>(&program, &plugins, &mut rng)
        .expect("setup fixes plugin and layout bindings");
    let private_inputs = [true];
    let outputs = [true];
    let root = ark_bn254::Fr::from(42u64);
    let proof = cirrus_groth16::prove_with_plugins::<Bn254, _, _>(
        &pk,
        &program,
        &plugins,
        &private_inputs,
        root,
        root,
        &outputs,
        &mut rng,
    )
    .expect("honest roots and outputs satisfy the plugin gadget");
    assert!(
        cirrus_groth16::verify_with_plugins::<Bn254, _>(
            &vk, &plugins, root, root, &outputs, &proof
        )
        .unwrap()
    );
    assert!(
        !cirrus_groth16::verify_with_plugins::<Bn254, _>(
            &vk,
            &plugins,
            root,
            ark_bn254::Fr::from(43u64),
            &outputs,
            &proof,
        )
        .unwrap(),
        "a changed public final root must fail verification"
    );
    let different_plugins = TestPlugins { binding: 0xBB };
    assert!(
        !cirrus_groth16::verify_with_plugins::<Bn254, _>(
            &vk,
            &different_plugins,
            root,
            root,
            &outputs,
            &proof
        )
        .unwrap(),
        "a different external-plugin binding must fail verification"
    );
}
