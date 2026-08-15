//! Cross-checks [`R1csBackend`] against `cirrus_recompile_core::interpret`
//! (the reference oracle every backend in this workspace is validated
//! against), through both `cirrus_recompile_rt::execute` and
//! `execute_prepared` -- mirroring
//! `cirrus-recompile-tests/tests/multi_backend.rs`'s cross-backend pattern.

use ark_bn254::Fr;
use ark_r1cs_std::prelude::*;
use ark_relations::gr1cs::ConstraintSystem;
use cirrus_r1cs_backend::R1csBackend;
use cirrus_recompile_core::OptimizationOptions;
use cirrus_zk_tests::sample_program;

const CASES: [(bool, bool); 4] = [(false, false), (false, true), (true, false), (true, true)];

#[test]
fn execute_matches_the_interpret_oracle() {
    let program = sample_program();
    for &(av, bv) in &CASES {
        let raw_inputs = [false, true, av, bv];
        let expected = cirrus_recompile_core::interpret(&program, &raw_inputs);

        let cs = ConstraintSystem::<Fr>::new_ref();
        let mut backend = R1csBackend::new(cs.clone());
        let inputs: Vec<Boolean<Fr>> = raw_inputs
            .iter()
            .map(|&bit| Boolean::new_witness(cs.clone(), || Ok(bit)).unwrap())
            .collect();

        let outputs = cirrus_recompile_rt::execute(&mut backend, &program, &inputs).unwrap();

        assert!(cs.is_satisfied().unwrap(), "unsatisfied for ({av}, {bv})");
        let actual: Vec<bool> = outputs.iter().map(|wire| wire.value().unwrap()).collect();
        assert_eq!(actual, expected, "execute mismatch for ({av}, {bv})");
    }
}

#[test]
fn execute_prepared_matches_the_interpret_oracle() {
    let program = sample_program();
    let prepared = program.prepare(&OptimizationOptions::default());
    for &(av, bv) in &CASES {
        let raw_inputs = [false, true, av, bv];
        let expected = cirrus_recompile_core::interpret_prepared(&prepared, &raw_inputs);

        let cs = ConstraintSystem::<Fr>::new_ref();
        let mut backend = R1csBackend::new(cs.clone());
        let inputs: Vec<Boolean<Fr>> = raw_inputs
            .iter()
            .map(|&bit| Boolean::new_witness(cs.clone(), || Ok(bit)).unwrap())
            .collect();

        let outputs =
            cirrus_recompile_rt::execute_prepared(&mut backend, &prepared, &inputs).unwrap();

        assert!(cs.is_satisfied().unwrap(), "unsatisfied for ({av}, {bv})");
        let actual: Vec<bool> = outputs.iter().map(|wire| wire.value().unwrap()).collect();
        assert_eq!(
            actual, expected,
            "execute_prepared mismatch for ({av}, {bv})"
        );
    }
}
