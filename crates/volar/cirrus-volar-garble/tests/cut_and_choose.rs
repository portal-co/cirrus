// @ai: assisted
//! Cut-and-choose harness over the volar garble/eval backends, using the generic
//! [`Broadcast`] wrapper to run the evaluator's N copies in parallel.
//!
//! M3 adds *authentication*: the garbler commits to every copy's table stream,
//! input/output label bases, and its own (garbler) input label *before* the
//! challenge, so it cannot adapt copies to the challenge, swap copies, or change
//! its input label afterwards. The evaluator verifies:
//!   - each *opened* copy re-garbles (from the revealed seed) to the commitment;
//!   - each *use* copy's received tables/bases/label hash to the commitment;
//!   - all use copies agree on the output.
//!
//! An honest garbler is accepted. A malicious garbler is caught whether it
//! garbles dishonestly, adapts a copy after committing, delivers a wrong input
//! label, or commits to inconsistent inputs across the use set (agreement).

use cipher::consts::{U16, U32};
use cirrus_recompile_core::{Idx, Op, Program};
use cirrus_volar_garble::cut_and_choose::Broadcast;
use cirrus_volar_garble::{VolarEvalBackend, VolarGarbleBackend};
use digest::Digest;
use hybrid_array::Array;
use sha2::Sha256;
use volar_spec::garble::{Eval, Garble, GarbleTable, GlobalSecret};

type Hash = Array<u8, U32>;

/// One garbled copy of the program (garbler-private; the `secret` is never
/// revealed for a use copy, only for an opened one).
struct GarbledCopy {
    tables: Vec<GarbleTable<U16>>,
    input_bases: Vec<Garble<U16>>,
    output_bases: Vec<Garble<U16>>,
    secret: GlobalSecret<U16>,
    /// The garbler's delivered input label (for input wire 0, the garbler's bit).
    garbler_input_label: Eval<U16>,
}

fn dg(parts: &[&[u8]]) -> Hash {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize()
}

fn chain(seed: &Hash, k: usize) -> Hash {
    let mut s = seed.clone();
    for _ in 0..k {
        s = Sha256::digest(&s);
    }
    s
}

fn garble_of(chain_bytes: &Hash) -> Garble<U16> {
    Garble {
        base: Array::<u8, U16>::from_fn(|i| chain_bytes[i]),
    }
}

struct PusherVec<T>(Vec<T>);
impl<T> cirrus_core::Pusher<T> for PusherVec<T> {
    fn push(&mut self, x: T) {
        self.0.push(x);
    }
}

/// Garble one copy deterministically from `copy_seed`, and prepare the garbler's
/// input label for its private input wire (input 0).
fn garble_copy(program: &Program, copy_seed: &Hash, secret: &GlobalSecret<U16>, garbler_bit: bool) -> GarbledCopy {
    let n_inputs = program.inputs.len();
    let input_bases: Vec<Garble<U16>> =
        (0..n_inputs).map(|k| garble_of(&chain(copy_seed, k + 1))).collect();
    let mut pusher = PusherVec(Vec::new());
    let mut garbler = VolarGarbleBackend::<Sha256, U16>::new(&mut pusher, secret.clone());
    garbler.seed = chain(copy_seed, n_inputs);
    let output_bases = cirrus_recompile_rt::execute(&mut garbler, program, &input_bases).unwrap();
    let garbler_input_label = secret.encode(&input_bases[0], garbler_bit);
    GarbledCopy {
        tables: pusher.0,
        input_bases,
        output_bases,
        secret: secret.clone(),
        garbler_input_label,
    }
}

/// The authentication commitment: binds the table stream, the label bases, and
/// the garbler's input label. This is everything the evaluator later relies on,
/// so committing before the challenge prevents adaptive attacks.
fn commitment(
    tables: &[GarbleTable<U16>],
    input_bases: &[Garble<U16>],
    output_bases: &[Garble<U16>],
    garbler_input_label: &Eval<U16>,
) -> Hash {
    let mut h = Sha256::new();
    for t in tables {
        for row in &t.table {
            h.update(&row[..]);
        }
    }
    for g in input_bases.iter().chain(output_bases.iter()) {
        h.update(&g.base[..]);
    }
    h.update(&garbler_input_label.target[..]);
    h.finalize()
}

fn commitment_of(copy: &GarbledCopy) -> Hash {
    commitment(&copy.tables, &copy.input_bases, &copy.output_bases, &copy.garbler_input_label)
}

fn sample_program() -> Program {
    // sum = a^b, carry = a&b, out = mux(sum, carry, a).
    Program {
        ops: vec![
            Op::Create(false), // slot 0 = input a (garbler)
            Op::Create(false), // slot 1 = input b (evaluator)
            Op::BitXor(Idx(0), Idx(1)), // slot 2 = sum
            Op::BitAnd(Idx(0), Idx(1)), // slot 3 = carry
            Op::Mux { cond: Idx(2), then: Idx(3), r#else: Idx(0) }, // slot 4
        ],
        inputs: vec![Idx(0), Idx(1)],
        outputs: vec![Idx(2), Idx(3), Idx(4)],
        externals: vec![],
    }
}

/// A malicious garbler's attack.
#[derive(Clone, Copy, PartialEq)]
enum Attack {
    None,
    /// Corrupt copy i's tables *before* committing (dishonest garble).
    DishonestGarble(usize),
    /// Corrupt copy i's tables *after* committing (adaptive to the challenge).
    AdaptiveCorrupt(usize),
    /// Deliver a wrong garbler-input label for use copy i.
    WrongInputLabel(usize),
    /// Commit to inconsistent garbler bits across the use copies.
    InconsistentInput,
}

fn derive_seed(master: &Hash, i: usize) -> Hash {
    dg(&[&master[..], b"copy", &[i as u8]])
}
fn derive_secret(master: &Hash, i: usize) -> GlobalSecret<U16> {
    GlobalSecret::new(Array::<u8, U16>::from_fn(|b| dg(&[&master[..], b"delta", &[i as u8]])[b]))
}

/// The full cut-and-choose run with authentication. Returns the agreed output
/// bits, or `Err` when the garbler's misbehavior is detected.
fn run_protocol(
    program: &Program,
    n: usize,
    garbler_bit: bool,
    eval_input: bool,
    attack: Attack,
) -> Result<Vec<bool>, String> {
    let master = dg(&[b"master"]);
    // === GARBLER: garble N copies (inconsistent-input commits to split bits) ===
    let mut copies: Vec<GarbledCopy> = (0..n)
        .map(|i| {
            // The garbler's bit per copy (inconsistent-input flips it for some).
            let bit = if attack == Attack::InconsistentInput && i % 4 == 3 {
                !garbler_bit
            } else {
                garbler_bit
            };
            garble_copy(program, &derive_seed(&master, i), &derive_secret(&master, i), bit)
        })
        .collect();
    // === ATTACK: dishonest garble (corrupt before commit) ===
    if let Attack::DishonestGarble(i) = attack {
        if let Some(t) = copies[i].tables.get_mut(0) {
            for row in t.table.iter_mut() {
                row[0] ^= 1;
            }
        }
    }
    // === GARBLER: commit (before the challenge) ===
    let commitments: Vec<Hash> = copies.iter().map(commitment_of).collect();
    // === EVALUATOR: challenge — open the even-indexed copies ===
    let open_set: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let use_set: Vec<usize> = (0..n).filter(|i| i % 2 == 1).collect();
    // === ATTACK: adaptive corrupt (after commit, before delivery) ===
    if let Attack::AdaptiveCorrupt(i) = attack {
        if let Some(t) = copies[i].tables.get_mut(0) {
            for row in t.table.iter_mut() {
                row[0] ^= 1;
            }
        }
    }
    // === OPEN: garbler reveals the seed; evaluator re-garbles and verifies ===
    for &i in &open_set {
        let derived = garble_copy(program, &derive_seed(&master, i), &derive_secret(&master, i), garbler_bit);
        if commitment_of(&derived) != commitments[i] {
            return Err(format!("opened copy {i} does not match its commitment"));
        }
    }
    // === USE: evaluator verifies each use copy against its commitment, then ===
    // === evaluates all use copies in parallel and requires agreement.      ===
    let mut evaluators = Vec::new();
    let mut input_labels_per = Vec::new();
    for &i in &use_set {
        let mut copy_labels = vec![
            copies[i].garbler_input_label.clone(),
            copies[i].secret.encode(&copies[i].input_bases[1], eval_input),
        ];
        // ATTACK: wrong input label (garbler delivers a label it didn't commit to).
        if attack == Attack::WrongInputLabel(i) {
            copy_labels[0] = copies[i].secret.encode(&copies[i].input_bases[0], !garbler_bit);
        }
        // Verify the use copy's tables/bases/input-label against its commitment.
        let delivered = commitment(&copies[i].tables, &copies[i].input_bases, &copies[i].output_bases, &copy_labels[0]);
        if delivered != commitments[i] {
            return Err(format!("use copy {i} does not match its commitment"));
        }
        evaluators.push(VolarEvalBackend::<Sha256, _, U16>::new(
            copies[i].tables.clone().into_iter(),
        ));
        input_labels_per.push(copy_labels);
    }
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
            let out = run_protocol(&program, 4, g, e, Attack::None).expect("honest garbler accepted");
            let sum = g ^ e;
            let carry = g & e;
            let mux = if sum { carry } else { g };
            assert_eq!(out, vec![sum, carry, mux], "output matches for g={g} e={e}");
        }
    }
}

#[test]
fn cut_and_choose_dishonest_garble_caught() {
    let program = sample_program();
    // Corrupt an *opened* copy (0) before commit → re-garble mismatch.
    assert!(run_protocol(&program, 4, false, true, Attack::DishonestGarble(0)).is_err());
    // Corrupt a *use* copy (1) before commit → the use set disagrees.
    assert!(run_protocol(&program, 4, false, true, Attack::DishonestGarble(1)).is_err());
}

#[test]
fn cut_and_choose_adaptive_corrupt_caught() {
    let program = sample_program();
    // Corrupting a *use* copy after committing: the delivered tables no longer
    // match the pre-challenge commitment, so the evaluator detects it.
    assert!(run_protocol(&program, 4, false, true, Attack::AdaptiveCorrupt(1)).is_err());
    // Corrupting an *open* copy after committing is harmless: the open reveals the
    // seed and the evaluator re-garbles from it, ignoring the in-memory copy, so
    // the protocol still accepts (the adaptive threat is only against use copies).
    assert!(run_protocol(&program, 4, false, true, Attack::AdaptiveCorrupt(0)).is_ok());
}

#[test]
fn cut_and_choose_wrong_input_label_caught() {
    let program = sample_program();
    // Delivering a garbler-input label that doesn't match the commitment fails.
    assert!(run_protocol(&program, 4, false, true, Attack::WrongInputLabel(1)).is_err());
}

#[test]
fn cut_and_choose_inconsistent_input_caught() {
    let program = sample_program();
    // Committing to inconsistent garbler bits across use copies: the copies are
    // honestly garbled and individually consistent, but they compute different
    // outputs, so the agreement check catches it.
    assert!(run_protocol(&program, 4, false, true, Attack::InconsistentInput).is_err());
}
