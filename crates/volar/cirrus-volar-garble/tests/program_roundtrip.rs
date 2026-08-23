//! Records one `Program` covering every `Op` variant, garbles it with
//! [`VolarGarbleBackend`], evaluates it with [`VolarEvalBackend`], and
//! cross-checks the opened outputs against `cirrus_recompile_core::interpret`
//! -- mirroring `cirrus-recompile-tests/tests/multi_backend.rs`'s pattern,
//! but calling `cirrus_recompile_rt::execute` directly (no codegen step).

use cipher::consts::U16;
use cirrus_core::ContextWithCreate;
use cirrus_recompile_core::Recorder;
use cirrus_volar_boolar::execute as execute_boolar;
use cirrus_volar_garble::{VolarEvalBackend, VolarGarbleBackend};
use hybrid_array::Array;
use sha2::Sha256;
use volar_spec::garble::{Garble, GlobalSecret};
use volar_ir::circuit::BCircuit;
use volar_lir_test_corpus::make_biir_half_adder;

const CASES: [(bool, bool); 4] = [(false, false), (false, true), (true, false), (true, true)];

struct VecPusher<T>(std::vec::Vec<T>);
impl<T> Default for VecPusher<T> {
    fn default() -> Self {
        Self(std::vec::Vec::new())
    }
}
impl<T> cirrus_core::Pusher<T> for VecPusher<T> {
    fn push(&mut self, x: T) {
        self.0.push(x);
    }
}

fn sample_program() -> cirrus_recompile_core::Program {
    let circuit = BCircuit::try_from_ir(&make_biir_half_adder()).expect("corpus fixture is fused");
    let mut recorder = Recorder::new();
    let a = recorder.create(false).unwrap();
    let b = recorder.create(false).unwrap();
    let outputs = execute_boolar(&mut recorder, &circuit, &[a, b], &mut []).unwrap();
    recorder.finish(vec![a, b], outputs)
}

#[test]
fn garble_then_evaluate_matches_the_interpret_oracle() {
    let program = sample_program();

    for &(av, bv) in &CASES {
        let raw_inputs = [av, bv];
        let expected = cirrus_recompile_core::interpret(&program, &raw_inputs);

        let mut tables = VecPusher::default();
        let secret = GlobalSecret::new(Array::<u8, U16>::from_fn(|i| (i as u8).wrapping_mul(37)));
        let mut garbler = VolarGarbleBackend::<Sha256, U16>::new(&mut tables, secret);

        // The garbler's own false-labels for the input wires are picked by
        // the caller (a garbler never learns the true bit, so any label is
        // a valid false-label reference) -- directly, rather than through
        // `Op::Create`, exactly like a real program's data inputs.
        let garbler_input_labels: [Garble<U16>; 2] = core::array::from_fn(|salt| Garble {
            base: Array::<u8, U16>::from_fn(|i| (i as u8) ^ (salt as u8 * 17)),
        });

        let garbled_outputs =
            cirrus_recompile_rt::execute(&mut garbler, &program, &garbler_input_labels).unwrap();

        let evaluator_input_labels: Vec<_> = garbler_input_labels
            .iter()
            .zip(raw_inputs.iter())
            .map(|(label, &bit)| garbler.secret.encode(label, bit))
            .collect();

        let mut evaluator = VolarEvalBackend::<Sha256, _, U16>::new(tables.0.into_iter());
        let evaluated_outputs =
            cirrus_recompile_rt::execute(&mut evaluator, &program, &evaluator_input_labels)
                .unwrap();

        for (i, (evaluated, garbled)) in evaluated_outputs
            .iter()
            .zip(garbled_outputs.iter())
            .enumerate()
        {
            let opened = evaluated.open(garbled);
            let bit = opened[0] & 1 != 0;
            assert_eq!(bit, expected[i], "output {i} mismatch for ({av}, {bv})");
        }
    }
}
