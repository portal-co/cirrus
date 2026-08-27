//! Drives [`VolarGarbleBackend`]/[`VolarEvalBackend`] directly from
//! `cirrus_ert::ert_emit` -- a real RV32 interpreter step, not a
//! hand-recorded `Program` -- confirming a garbled `ADD` instruction opens
//! to the correct native sum. Mirrors
//! `cirrus-garbled-circuit`'s own
//! `evaluator_replays_an_ert_add_from_the_complete_table_iterator` test,
//! with `volar_spec::garble` doing the cryptography instead of this
//! workspace's baseline four-row-table construction.

use std::convert::Infallible;

use cipher::consts::U16;
use cirrus_ert::{DefaultHandler as RvHashHandler, RawMemory, RvDefaultHandler, ert_emit};
use cirrus_volar_garble::{VolarEvalBackend, VolarEvalError, VolarGarbleBackend};
use hybrid_array::Array;
use rv_asm::{Inst, Reg, Xlen};
use sha2::Sha256;
use volar_spec::garble::{Eval, Garble, GarbleTable, GlobalSecret};

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

fn program(instructions: impl IntoIterator<Item = Inst>) -> Vec<u8> {
    instructions
        .into_iter()
        .flat_map(|instruction| instruction.encode_normal(Xlen::Rv32).to_le_bytes())
        .collect()
}

fn no_hash_garble<C>(_: &mut C, _: &[[Garble<U16>; 32]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}
fn no_hash_eval<C>(_: &mut C, _: &[[Eval<U16>; 32]]) -> Result<[u8; 32], VolarEvalError> {
    Ok([0; 32])
}

#[test]
fn ert_add_garbles_and_evaluates_to_the_native_sum() {
    let instructions = program([
        Inst::Add {
            dest: Reg::T0,
            src1: Reg::A1,
            src2: Reg::A2,
        },
        Inst::Ecall,
    ]);

    let left_value: u32 = 0x1020_3040;
    let right_value: u32 = 0x0102_0304;

    let secret = GlobalSecret::new(Array::<u8, U16>::from_fn(|i| (i as u8).wrapping_mul(53)));
    let garbling_zero = Garble::<U16>::zero();
    let garbling_one = secret.not_garble(&garbling_zero);
    let right_base = Garble::<U16> {
        base: Array::<u8, U16>::from_fn(|i| (i as u8) ^ 0x77),
    };

    // Garbler side: every A1 bit reuses the same false-label (and every A2
    // bit the same distinct false-label) -- each bit is still an
    // independent wire through its own carry-chain gate, so this is
    // correctness-preserving for a validation test, matching the same
    // simplification cirrus-garbled-circuit's own ERT test makes.
    let mut garbled_registers: [[Garble<U16>; 32]; 32] =
        core::array::from_fn(|_| core::array::from_fn(|_| garbling_zero.clone()));
    garbled_registers[Reg::A0.0 as usize] = core::array::from_fn(|_| garbling_one.clone());
    garbled_registers[Reg::A1.0 as usize] = core::array::from_fn(|_| garbling_zero.clone());
    garbled_registers[Reg::A2.0 as usize] = core::array::from_fn(|_| right_base.clone());
    let mut garbled_constants = [None; 32];
    garbled_constants[Reg::A0.0 as usize] = Some(u32::MAX);
    let mut garbled_rstack = [0u32; 8];
    let mut garbled_vstack: Vec<Garble<U16>> = std::vec![garbling_zero.clone(); 64];

    let mut tables = VecPusher::<GarbleTable<U16>>::default();
    let garbler = VolarGarbleBackend::<Sha256, U16>::new(&mut tables, secret.clone());
    let mut handler = RvDefaultHandler {
        inner: RvHashHandler {
            context: garbler,
            hash: no_hash_garble::<VolarGarbleBackend<Sha256, U16>>,
        },
    };

    let garbled = ert_emit(
        &mut handler,
        &mut garbled_vstack,
        64,
        RawMemory::from(instructions.as_slice()),
        &mut garbled_rstack,
        0,
        &mut garbled_registers,
        &mut garbled_constants,
        garbling_zero.clone(),
        garbling_one.clone(),
    );
    assert!(garbled.is_ok(), "garbling the ert-interpreted add succeeds");
    let garbled_result = garbled_registers[Reg::T0.0 as usize].clone();
    drop(handler);

    // Evaluator side: encode each bit's own true value against the
    // matching garbler false-label via `GlobalSecret::encode`.
    let evaluator_zero = Eval::<U16>::zero();
    let evaluator_one = secret.one_wire_eval();
    let mut evaluated_registers: [[Eval<U16>; 32]; 32] =
        core::array::from_fn(|_| core::array::from_fn(|_| evaluator_zero.clone()));
    evaluated_registers[Reg::A0.0 as usize] = core::array::from_fn(|_| evaluator_one.clone());
    evaluated_registers[Reg::A1.0 as usize] =
        core::array::from_fn(|bit| secret.encode(&garbling_zero, (left_value >> bit) & 1 != 0));
    evaluated_registers[Reg::A2.0 as usize] =
        core::array::from_fn(|bit| secret.encode(&right_base, (right_value >> bit) & 1 != 0));
    let mut evaluated_constants = [None; 32];
    evaluated_constants[Reg::A0.0 as usize] = Some(u32::MAX);
    let mut evaluated_rstack = [0u32; 8];
    let mut evaluated_vstack: Vec<Eval<U16>> = std::vec![evaluator_zero.clone(); 64];

    let evaluator = VolarEvalBackend::<Sha256, _, U16>::new(tables.0.into_iter());
    let mut evaluator_handler = RvDefaultHandler {
        inner: RvHashHandler {
            context: evaluator,
            hash: no_hash_eval::<VolarEvalBackend<Sha256, std::vec::IntoIter<GarbleTable<U16>>, U16>>,
        },
    };

    let evaluated = ert_emit(
        &mut evaluator_handler,
        &mut evaluated_vstack,
        64,
        RawMemory::from(instructions.as_slice()),
        &mut evaluated_rstack,
        0,
        &mut evaluated_registers,
        &mut evaluated_constants,
        evaluator_zero,
        evaluator_one,
    );
    assert!(
        evaluated.is_ok(),
        "evaluation consumes every add table in order"
    );

    let expected = left_value.wrapping_add(right_value);
    for bit in 0..32 {
        let opened = evaluated_registers[Reg::T0.0 as usize][bit].open(&garbled_result[bit]);
        let actual_bit = opened[0] & 1 != 0;
        let expected_bit = (expected >> bit) & 1 != 0;
        assert_eq!(actual_bit, expected_bit, "result bit {bit}");
    }
}
