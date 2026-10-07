//! GRAM action host (interpreter-side garbled RAM access gadget): decode
//! action-arg labels to plaintext, deliver results per `GramOutput` mode,
//! using the shared `volar_spec::garble` vocabulary.

use cipher::consts::U16;
use cirrus_volar_garble::{GramActionHost, GramActionResult};
use hybrid_array::Array;
use volar_spec::garble::{Garble, GlobalSecret, gram_decode_label};

fn det_secret() -> GlobalSecret<U16> {
    GlobalSecret::new(Array::<u8, U16>::from_fn(|i| {
        (i as u8).wrapping_mul(37) | 1
    }))
}

fn det_garble(seed: u8) -> Garble<U16> {
    Garble {
        base: Array::<u8, U16>::from_fn(|i| seed.wrapping_add(i as u8)),
    }
}

#[test]
fn decode_args_recovers_arg_bits() {
    let secret = det_secret();
    // Three action arg wires with distinct false-labels, values [t, f, t].
    let bases: [_; 3] = [det_garble(0x10), det_garble(0x40), det_garble(0x70)];
    let bits = [true, false, true];
    let args: [_; 3] = [
        secret.encode(&bases[0], bits[0]),
        secret.encode(&bases[1], bits[1]),
        secret.encode(&bases[2], bits[2]),
    ];
    let decoded = GramActionHost::decode_args(&args, &bases);
    assert_eq!(decoded, bits, "host must recover each arg wire's value");
}

#[test]
fn deliver_cleartext_learns_bit() {
    let host = GramActionHost::new(det_secret());
    let base = det_garble(0x20);
    for bit in [false, true] {
        match host.deliver(volar_spec::garble::GramOutput::Cleartext, &base, bit) {
            GramActionResult::Cleartext(bits) => assert_eq!(bits, vec![bit]),
            GramActionResult::Regarble(_) => panic!("cleartext mode must return plaintext"),
        }
    }
}

#[test]
fn deliver_regarble_returns_label_decoding_to_bit() {
    let host = GramActionHost::new(det_secret());
    let base = det_garble(0x30);
    for bit in [false, true] {
        match host.deliver(volar_spec::garble::GramOutput::Regarble, &base, bit) {
            GramActionResult::Regarble(labels) => {
                assert_eq!(labels.len(), 1);
                // The evaluator gets a label it cannot read without `base`,
                // but the host can verify it decodes back to the bit.
                assert_eq!(gram_decode_label(&labels[0], &base), bit);
            }
            GramActionResult::Cleartext(_) => panic!("regarble mode must return a label"),
        }
    }
}
