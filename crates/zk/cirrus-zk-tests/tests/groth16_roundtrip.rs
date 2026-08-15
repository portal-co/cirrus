//! End-to-end Groth16 round-trip over [`sample_program`]: setup -> prove ->
//! verify succeeds for the honest witness/outputs, and fails once the
//! public outputs handed to `verify` are corrupted.

use ark_bn254::Bn254;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use cirrus_zk_tests::sample_program;

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
