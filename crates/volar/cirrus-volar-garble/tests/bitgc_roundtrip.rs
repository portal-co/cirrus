//! BitGC adapter round trip: `(x0 ∧ x1) ⊕ x2` through the Cirrus
//! streaming/pull backend over `volar_spec::bitgc`.
//!
//! Correctness-only profile (small NTT-friendly parameters); no security
//! claim. Mirrors the reference-adapter status in `volar_spec::bitgc`.
#![cfg(test)]

use cirrus_core::Pusher;
use cirrus_volar_garble::bitgc::{BitGcEvalBackend, BitGcGarbleBackend};
use volar_spec::bitgc::common::{Circuit, Gate, GlobalDelta, ReferenceEquation, Stitch, WireState};
use volar_spec::bitgc::prg::ReferencePrg;
use volar_spec::bitgc::seed::{offline_setup, stream_expansion};
use volar_spec::bitgc::swhe::{Context, Parameters, RandomSource, ZeroNoise};

struct TestRandom(u64);
impl RandomSource for TestRandom {
    fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), volar_spec::bitgc::swhe::Error> {
        for byte in output.iter_mut() {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (self.0 >> 33) as u8;
        }
        Ok(())
    }
}

fn test_context() -> Context {
    Context::new(Parameters {
        degree: 8,
        plaintext_modulus: 97,
        chain: vec![
            1_099_511_645_201,
            1_099_511_673_137,
            1_099_511_677_793,
            1_099_511_704_177,
            1_099_511_705_729,
            1_099_511_755_393,
            1_099_511_798_849,
            1_099_511_801_953,
        ],
        noise_bound: 2,
    })
    .unwrap()
}

/// `(x0 ∧ x1) ⊕ x2`.
fn circuit() -> Circuit {
    Circuit {
        inputs: 3,
        gates: vec![Gate::And { a: 0, b: 1 }, Gate::Xor { a: 3, b: 2 }],
        outputs: vec![4],
    }
}

/// A `Vec`-backed record sink.
struct VecPusher(Vec<Stitch>);
impl Pusher<Stitch> for VecPusher {
    fn push(&mut self, x: Stitch) {
        self.0.push(x);
    }
}

fn run(equation: ReferenceEquation, inputs: [bool; 3]) -> bool {
    let ctx = test_context();
    let circuit = circuit();
    let p = ctx.parameters().plaintext_modulus;
    let delta = GlobalDelta::new(5, p).unwrap();
    let prg = ReferencePrg::new(2);
    let mut random = TestRandom(0x1234_5678_9abc_def0);

    // Offline setup + expansion stream.
    let garbler =
        offline_setup(&ctx, circuit.wires(), &prg, 1 << 10, 7, 2, &mut random, &mut ZeroNoise)
            .unwrap();
    let steps =
        stream_expansion(&ctx, circuit.wires(), &prg, &garbler, &mut random, &mut ZeroNoise)
            .unwrap();
    let states: Vec<WireState> = steps.iter().map(|s| s.wire_state).collect();
    let evaluator_wires: Vec<_> = steps.iter().map(|s| s.eval.clone()).collect();

    // Garbler side: stream records into a sink.
    let mut sink = VecPusher(Vec::new());
    {
        let mut backend =
            BitGcGarbleBackend::new(&mut sink, &garbler, delta, equation, p);
        backend.garble_circuit(&circuit, &states).unwrap();
    }
    let records = sink.0;

    // Input labels from the garbler's states.
    let mut input_labels = Vec::new();
    for (w, &bit) in inputs.iter().enumerate() {
        let (a0, pi) = (states[w].a0, states[w].pi);
        let masked = bit ^ pi;
        let label = (a0 + if masked { delta.value() } else { 0 }) % p;
        let mut slots = vec![0u64; 8];
        slots[evaluator_wires[w].slot_a0] = label;
        input_labels.push(
            ctx.encrypt(&garbler.secret, 6, &slots, &mut random, &mut ZeroNoise)
                .unwrap(),
        );
    }

    // Evaluator side: pull records and evaluate.
    let evaluator = volar_spec::bitgc::seed::OfflineEvaluator {
        plan: garbler.plan,
        wires: evaluator_wires,
    };
    let mut backend = BitGcEvalBackend {
        records: records.into_iter(),
        expanded: evaluator,
        ctx: test_context(),
        ksks: garbler.ksks.clone(),
        delta,
        equation,
        output_key: garbler.secret.clone(),
    };
    backend.evaluate_circuit(&circuit, &input_labels).unwrap()[0]
}

#[test]
fn small_variant_truth_table() {
    for bits in 0..8u8 {
        let inputs = [bits & 1 == 1, bits & 2 == 2, bits & 4 == 4];
        let expect = (inputs[0] & inputs[1]) ^ inputs[2];
        assert_eq!(run(ReferenceEquation::SMALL, inputs), expect, "inputs {inputs:?}");
    }
}

#[test]
fn fast_variant_truth_table() {
    for bits in 0..8u8 {
        let inputs = [bits & 1 == 1, bits & 2 == 2, bits & 4 == 4];
        let expect = (inputs[0] & inputs[1]) ^ inputs[2];
        assert_eq!(run(ReferenceEquation::FAST, inputs), expect, "inputs {inputs:?}");
    }
}
