//! D3 enabler: the constant-carrying evaluator backend.
//!
//! `VolarEvalBackend` collapses `create(false)` and `create(true)` to the
//! same all-zero `Eval`, which is correct for a woven circuit (no internal
//! constants) but wrong for the ERT machine, whose ABI/ALU build constant
//! words from distinct `zero`/`one` wires. `VcEvalBackend` sources the two
//! constant-wire labels from a supply, so `create(false)` and `create(true)`
//! are distinguishable and behave correctly under the gate operations.

use cipher::consts::U16;
use cirrus_core::{ContextWithBitXor, ContextWithCreate, ContextWithValue};
use cirrus_volar_garble::VcEvalBackend;
use hybrid_array::Array;
use sha2::Sha256;
use volar_spec::garble::{Eval, Garble, GarbleTable, GlobalSecret, gram_decode_label};

fn det_secret() -> GlobalSecret<U16> {
    GlobalSecret::new(Array::<u8, U16>::from_fn(|i| {
        (i as u8).wrapping_mul(37) | 1
    }))
}

fn const_base(tag: u8) -> Garble<U16> {
    Garble {
        base: Array::<u8, U16>::from_fn(|i| {
            (i as u8).wrapping_mul(11).wrapping_add(tag) | 1
        }),
    }
}

#[test]
fn vc_eval_constants_are_distinct_and_decode() {
    let secret = det_secret();
    let zero_base = const_base(0x00);
    let one_base = const_base(0x01);
    let zero_label = secret.encode(&zero_base, false);
    let one_label = secret.encode(&one_base, true);

    let tables: Vec<GarbleTable<U16>> = Vec::new();
    let consts = vec![zero_label.clone(), one_label.clone()];
    let mut ctx = VcEvalBackend::<Sha256, _, U16, _>::new(tables.into_iter(), consts.into_iter());

    let z = ctx.create(false).unwrap();
    let o = ctx.create(true).unwrap();

    // Distinct labels for distinct constants.
    assert_ne!(z.target, o.target, "const-0 and const-1 must differ");

    // Each decodes against its own base.
    assert!(!gram_decode_label(&z, &zero_base));
    assert!(gram_decode_label(&o, &one_base));
}

#[test]
fn vc_eval_create_is_stable_across_calls() {
    let secret = det_secret();
    let zero_label = secret.encode(&const_base(0x00), false);
    let one_label = secret.encode(&const_base(0x01), true);
    let tables: Vec<GarbleTable<U16>> = Vec::new();
    let consts = vec![zero_label.clone(), one_label.clone()];
    let mut ctx = VcEvalBackend::<Sha256, _, U16, _>::new(tables.into_iter(), consts.into_iter());

    // Repeated create(false) returns the same const-0 wire (the machine has
    // exactly one), and create(true) the const-1 wire.
    assert_eq!(ctx.create(false).unwrap().target, zero_label.target);
    assert_eq!(ctx.create(false).unwrap().target, zero_label.target);
    assert_eq!(ctx.create(true).unwrap().target, one_label.target);
}

#[test]
fn vc_eval_xor_with_const_zero_is_identity() {
    let secret = det_secret();
    // In a real garbled circuit the constant-0 wire's false-label is the
    // all-zero label, so XOR with it is the identity. Model that here: the
    // const-0 label is the all-zero `Eval` (the false-label of a zero base).
    let zero_label = Eval::<U16>::zero();
    let one_label = secret.encode(&const_base(0x01), true);
    let tables: Vec<GarbleTable<U16>> = Vec::new();
    let consts = vec![zero_label.clone(), one_label];
    let mut ctx = VcEvalBackend::<Sha256, _, U16, _>::new(tables.into_iter(), consts.into_iter());

    // A distinct data wire: encode bit `true` against its own base.
    let data_base = const_base(0x77);
    let data = secret.encode(&data_base, true);

    // x XOR const-0 == x (free-XOR with the all-zero wire).
    let z = ctx.create(false).unwrap();
    let out = ctx.bitxor(data.clone(), z).unwrap();
    assert!(
        gram_decode_label(&out, &data_base),
        "x XOR 0 must still decode to x=true"
    );
}

type _AssertWrapped = <VcEvalBackend<
    Sha256,
    std::vec::IntoIter<GarbleTable<U16>>,
    U16,
    std::vec::IntoIter<Eval<U16>>,
> as ContextWithValue<bool>>::Wrapped;
