//! Records one `Program` covering every `Op` variant, commits its inputs
//! via `vole_commit_bit` against a shared `IdealCot`, runs it against
//! [`VoleProverContext`] (collecting the `hat` transcript) and separately
//! against [`VoleVerifierContext`] (consuming that transcript), and checks
//! the verifier's derived output shares against what the prover's own
//! (same-process) `Vope`s evaluate to at `Delta` -- the fundamental
//! correctness property of the VOLE-committed representation, independent
//! of gate type. A negative case then corrupts one transcript entry and
//! confirms the corresponding `vole_and_verifier_check` now rejects.
//!
//! `derive_and_q`/`bitand` on the verifier side never themselves error on a
//! corrupted `hat` -- that is intentional, matching `volar-spec`'s own
//! `tampered_hat_rejected` test: detection happens at the explicit
//! verification step (`vole_and_verifier_check`), not at interpretation
//! time.

use cipher::consts::U1;
use cirrus_core::ContextWithCreate;
use cirrus_recompile_core::Recorder;
use cirrus_volar_boolar::{MuxTreeContext, execute as execute_boolar};
use cirrus_volar_vole::{VoleProverContext, VoleVerifierContext};
use hybrid_array::Array;
use volar_ir::circuit::BCircuit;
use volar_lir_test_corpus::make_biir_half_adder;
use volar_spec::{
    SpecRng,
    field::Galois128,
    ot::IdealCot,
    vole::{
        prove::vole_and_verifier_check,
        setup::{random_nonzero_delta, vole_commit_bit},
    },
};

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

/// SplitMix64-based deterministic test RNG, matching `volar-spec`'s own
/// `TestRng` convention in `vole/setup.rs`/`ot/ideal_cot.rs`.
struct TestRng(u64);
impl SpecRng for TestRng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
}

fn sample_g128(r: &mut TestRng) -> Galois128 {
    let hi = ((r.next_u32() as u128) << 96) | ((r.next_u32() as u128) << 64);
    let lo = ((r.next_u32() as u128) << 32) | (r.next_u32() as u128);
    Galois128(hi | lo)
}
fn is_zero_g128(g: &Galois128) -> bool {
    g.0 == 0
}
fn bit_to_t(b: bool) -> Galois128 {
    Galois128(b as u128)
}

fn sample_program() -> cirrus_recompile_core::Program {
    let circuit = BCircuit::try_from_ir(&make_biir_half_adder()).expect("corpus fixture is fused");
    let mut recorder = MuxTreeContext::new(Recorder::new());
    let a = recorder.create(false).unwrap();
    let b = recorder.create(false).unwrap();
    let outputs = execute_boolar(&mut recorder, &circuit, &[a, b], &mut []).unwrap();
    recorder.into_inner().finish(vec![a, b], outputs)
}

#[test]
fn prover_and_verifier_agree_and_reject_a_corrupted_transcript() {
    let program = sample_program();

    for &(av, bv) in &CASES {
        let raw_inputs = [av, bv];

        let mut rng = TestRng(0xC1_2C_55_00 ^ (av as u64) << 8 ^ (bv as u64));
        let delta = random_nonzero_delta::<U1, Galois128, _>(&mut rng, sample_g128, is_zero_g128);
        let cot = IdealCot::<U1, Galois128>::new(delta.clone());

        let mut prover_inputs = Vec::new();
        let mut verifier_inputs = Vec::new();
        for &bit in &raw_inputs {
            let (vope, q) = vole_commit_bit(&cot, &mut rng, sample_g128, bit_to_t, bit);
            prover_inputs.push(vope);
            verifier_inputs.push(q);
        }

        let mut hats = VecPusher::<Array<Galois128, U1>>::default();
        let mut prover = VoleProverContext {
            hats: &mut hats,
            bit_to_t,
        };
        let prover_outputs =
            cirrus_recompile_rt::execute(&mut prover, &program, &prover_inputs).unwrap();

        // Honest verifier run.
        let mut verifier = VoleVerifierContext {
            delta: delta.clone(),
            hats: hats.0.clone().into_iter(),
        };
        let verifier_outputs =
            cirrus_recompile_rt::execute(&mut verifier, &program, &verifier_inputs).unwrap();

        for (i, (p, v)) in prover_outputs
            .iter()
            .zip(verifier_outputs.iter())
            .enumerate()
        {
            assert!(
                p.clone() * delta.clone() == *v,
                "output {i} mismatch for ({av}, {bv})"
            );
        }

        // Explicit soundness check at the "and" output wire (outputs[1]),
        // whose two operands are the raw `a`/`b` inputs directly.
        let q_a = verifier_inputs[0].clone();
        let q_b = verifier_inputs[1].clone();
        let and_hat = hats.0[0].clone();
        let (_, ok) = vole_and_verifier_check(&delta, &q_a, &q_b, &verifier_outputs[1], &and_hat);
        assert!(ok, "honest and-gate check rejected for ({av}, {bv})");

        // Negative case: keep the honestly-derived `q_and` (verifier_outputs[1],
        // matching what the prover actually committed to), but check it
        // against a DIFFERENT hat than the one it was derived from -- this
        // is what a tampered transcript looks like from the verifier's
        // perspective. (Re-deriving `q_and` from the same corrupted hat via
        // `derive_and_q` and self-checking it would be tautologically
        // consistent -- `derive_and_q` and `vole_and_verifier_check` are
        // inverses of the same relation -- so that would NOT test anything;
        // this mirrors volar-spec's own `tampered_hat_rejected` test.)
        let mut bad_hat = and_hat;
        bad_hat[0] = bad_hat[0] + Galois128(1);
        let (_, bad_ok) =
            vole_and_verifier_check(&delta, &q_a, &q_b, &verifier_outputs[1], &bad_hat);
        assert!(
            !bad_ok,
            "verifier accepted a corrupted hat for ({av}, {bv})"
        );
    }
}
