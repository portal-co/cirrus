// @ai: assisted
//! Cut-and-choose harness over the volar garble/eval backends, using the generic
//! [`Broadcast`] wrapper to run the evaluator's N copies in parallel.
//!
//! Same-process protocol demo: the garbler garbles N copies of a program (each
//! with an independent `GlobalSecret` + label chain derived from a per-copy
//! seed), commits to each, the evaluator challenges with an open subset, the
//! garbler opens (reveals the seed) and the evaluator re-garbles to verify, and
//! the remaining copies are evaluated with agreement required.
//!
//! An honest garbler is accepted; a garbler that corrupts one copy is caught
//! (either its opened copy fails re-garbling, or the use set disagrees).

use cipher::consts::{U16, U32};
use cirrus_recompile_core::{Idx, Op, Program};
use cirrus_volar_garble::cut_and_choose::Broadcast;
use cirrus_volar_garble::{VolarEvalBackend, VolarGarbleBackend};
use digest::Digest;
use hybrid_array::Array;
use sha2::Sha256;
use volar_spec::garble::{Eval, Garble, GarbleTable, GlobalSecret};

/// One garbled copy of the program: the table stream plus the input/output
/// false-label bases the garbler used, plus its per-copy global secret.
struct GarbledCopy {
    tables: Vec<GarbleTable<U16>>,
    input_bases: Vec<Garble<U16>>,
    output_bases: Vec<Garble<U16>>,
    secret: GlobalSecret<U16>,
}

fn dg(parts: &[&[u8]]) -> Array<u8, U32> {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize()
}

/// digest^`k`(seed) — the garbler's `fresh_label` chain.
fn chain(seed: &Array<u8, U32>, k: usize) -> Array<u8, U32> {
    let mut s = seed.clone();
    for _ in 0..k {
        s = Sha256::digest(&s);
    }
    s
}

fn garble_of(chain_bytes: &Array<u8, U32>) -> Garble<U16> {
    Garble {
        base: Array::<u8, U16>::from_fn(|i| chain_bytes[i]),
    }
}

fn table_eq(a: &GarbleTable<U16>, b: &GarbleTable<U16>) -> bool {
    a.table.iter().zip(b.table.iter()).all(|(x, y)| x[..] == y[..])
}
fn garble_eq(a: &Garble<U16>, b: &Garble<U16>) -> bool {
    a.base[..] == b.base[..]
}

// A tiny `Pusher` over a `Vec` so the garbler can stream tables.
struct PusherVec<T>(Vec<T>);
impl<T> cirrus_core::Pusher<T> for PusherVec<T> {
    fn push(&mut self, x: T) {
        self.0.push(x);
    }
}

/// Garble one copy deterministically from `copy_seed`. The input bases are the
/// first `n_inputs` chain labels; the internal gate labels continue the chain
/// (the garbler's `seed` is set past the input labels so they never collide).
fn garble_copy(
    program: &Program,
    copy_seed: &Array<u8, U32>,
    secret: &GlobalSecret<U16>,
) -> GarbledCopy {
    let n_inputs = program.inputs.len();
    let input_bases: Vec<Garble<U16>> =
        (0..n_inputs).map(|k| garble_of(&chain(copy_seed, k + 1))).collect();
    let mut pusher = PusherVec(Vec::new());
    let mut garbler = VolarGarbleBackend::<Sha256, U16>::new(&mut pusher, secret.clone());
    garbler.seed = chain(copy_seed, n_inputs);
    let output_bases = cirrus_recompile_rt::execute(&mut garbler, program, &input_bases).unwrap();
    GarbledCopy {
        tables: pusher.0,
        input_bases,
        output_bases,
        secret: secret.clone(),
    }
}

/// Re-garble from a revealed seed and check the copy matches exactly.
fn open_and_verify(program: &Program, copy_seed: &Array<u8, U32>, copy: &GarbledCopy) -> bool {
    let fresh = garble_copy(program, copy_seed, &copy.secret);
    fresh.tables.len() == copy.tables.len()
        && fresh.tables.iter().zip(&copy.tables).all(|(a, b)| table_eq(a, b))
        && fresh.input_bases.len() == copy.input_bases.len()
        && fresh.input_bases.iter().zip(&copy.input_bases).all(|(a, b)| garble_eq(a, b))
        && fresh.output_bases.len() == copy.output_bases.len()
        && fresh.output_bases.iter().zip(&copy.output_bases).all(|(a, b)| garble_eq(a, b))
}

/// A binding commitment to a copy's table stream and label bases.
fn commitment(copy: &GarbledCopy) -> Array<u8, U32> {
    let mut h = Sha256::new();
    for t in &copy.tables {
        for row in &t.table {
            h.update(&row[..]);
        }
    }
    for g in copy.input_bases.iter().chain(copy.output_bases.iter()) {
        h.update(&g.base[..]);
    }
    h.finalize()
}

fn sample_program() -> Program {
    // sum = a^b, carry = a&b, out = mux(sum, carry, a).
    Program {
        ops: vec![
            Op::Create(false), // slot 0 = input a
            Op::Create(false), // slot 1 = input b
            Op::BitXor(Idx(0), Idx(1)), // slot 2 = sum
            Op::BitAnd(Idx(0), Idx(1)), // slot 3 = carry
            Op::Mux { cond: Idx(2), then: Idx(3), r#else: Idx(0) }, // slot 4
        ],
        inputs: vec![Idx(0), Idx(1)],
        outputs: vec![Idx(2), Idx(3), Idx(4)],
        externals: vec![],
    }
}

/// The full cut-and-choose run. `corrupt` sabotages one copy's tables (a
/// malicious garbler). Returns the agreed output bits, or `Err` on detection.
fn cut_and_choose(
    program: &Program,
    n: usize,
    garbler_input: bool,
    eval_input: bool,
    corrupt: Option<usize>,
) -> Result<Vec<bool>, String> {
    let master = dg(&[b"master"]);
    // 1. Garble N copies.
    let mut copies: Vec<GarbledCopy> = (0..n)
        .map(|i| {
            let copy_seed = dg(&[&master[..], b"copy", &[i as u8]]);
            let secret = GlobalSecret::new(Array::<u8, U16>::from_fn(|b| {
                dg(&[&master[..], b"delta", &[i as u8]])[b]
            }));
            garble_copy(program, &copy_seed, &secret)
        })
        .collect();
    // A malicious garbler corrupts one copy's first table.
    if let Some(i) = corrupt {
        if let Some(t) = copies[i].tables.get_mut(0) {
            for row in t.table.iter_mut() {
                row[0] ^= 1;
            }
        }
    }
    // 2. Commit (a real run sends these before the challenge).
    let _commitments: Vec<_> = copies.iter().map(commitment).collect();
    // 3. Challenge: open the even-indexed copies.
    let open: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    // 4. Open + verify.
    for &i in &open {
        let copy_seed = dg(&[&master[..], b"copy", &[i as u8]]);
        if !open_and_verify(program, &copy_seed, &copies[i]) {
            return Err(format!("copy {i} failed re-garbling (malicious garbler)"));
        }
    }
    // 5. Use the unopened copies; their outputs must agree. Encode inputs per
    // copy (garbler input 0 directly; evaluator input 1 via a simulated OT —
    // same-process here, a real run delivers it by OT).
    let use_set: Vec<usize> = (0..n).filter(|i| i % 2 == 1).collect();
    let mut evaluators = Vec::new();
    let mut input_labels_per = Vec::new();
    for &i in &use_set {
        let copy = &copies[i];
        let labels: Vec<Eval<U16>> = vec![
            copy.secret.encode(&copy.input_bases[0], garbler_input),
            copy.secret.encode(&copy.input_bases[1], eval_input),
        ];
        evaluators.push(VolarEvalBackend::<Sha256, _, U16>::new(
            copy.tables.clone().into_iter(),
        ));
        input_labels_per.push(labels);
    }
    // Run the use-set evaluators in parallel via the Broadcast wrapper: the
    // program executes once, each Boolean op broadcast to all use-set copies.
    let mut broadcast = Broadcast(evaluators);
    let n_inputs = program.inputs.len();
    let broadcast_inputs: Vec<Vec<Eval<U16>>> = (0..n_inputs)
        .map(|k| input_labels_per.iter().map(|l| l[k].clone()).collect())
        .collect();
    let outputs = cirrus_recompile_rt::execute(&mut broadcast, program, &broadcast_inputs).unwrap();
    let mut agreed: Option<Vec<bool>> = None;
    for (j, &i) in use_set.iter().enumerate() {
        let decoded: Vec<bool> = outputs
            .iter()
            .zip(copies[i].output_bases.iter())
            .map(|(outs, base)| outs[j].open(base)[0] & 1 != 0)
            .collect();
        match &agreed {
            None => agreed = Some(decoded),
            Some(prev) if *prev != decoded => {
                return Err(format!("use-set copy {i} disagrees (malicious garbler)"));
            }
            _ => {}
        }
    }
    agreed.ok_or_else(|| "empty use set".into())
}

#[test]
fn cut_and_choose_honest_garbler_accepted() {
    let program = sample_program();
    for g in [false, true] {
        for e in [false, true] {
            let out = cut_and_choose(&program, 4, g, e, None).expect("honest garbler accepted");
            let sum = g ^ e;
            let carry = g & e;
            let mux = if sum { carry } else { g };
            assert_eq!(out, vec![sum, carry, mux], "output matches for g={g} e={e}");
        }
    }
}

#[test]
fn cut_and_choose_malicious_garbler_caught() {
    let program = sample_program();
    // Corrupting an *opened* copy (index 0) fails re-garbling.
    let r = cut_and_choose(&program, 4, false, true, Some(0));
    assert!(r.is_err(), "corrupting an opened copy is caught: {r:?}");
    // Corrupting a *use* copy (index 1) makes the use set disagree.
    let r = cut_and_choose(&program, 4, false, true, Some(1));
    assert!(r.is_err(), "corrupting a use copy is caught: {r:?}");
}
