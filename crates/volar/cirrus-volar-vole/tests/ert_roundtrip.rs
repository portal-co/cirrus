//! Drives [`VoleProverContext`]/[`VoleVerifierContext`] directly from
//! `cirrus_ert::ert_emit` -- a real RV32 interpreter step, not a
//! hand-recorded `Program` -- confirming a VOLE-committed `ADD`
//! instruction's prover/verifier shares agree, and that the prover's own
//! committed value tracks the correct native sum bit-for-bit.
//!
//! Each `A1`/`A2` bit is committed independently via `vole_commit_bit`
//! against a shared `IdealCot`, mirroring `cirrus-volar-garble`'s
//! `ert_roundtrip.rs` test but for the VOLE-ZK backend. `A0` (the ECALL
//! exit sentinel, a known constant) is built directly via `create`, needing
//! no OT commitment.

use core::convert::Infallible;

use cipher::consts::U1;
use cirrus_core::ContextWithCreate;
use cirrus_ert::{DefaultHandler as RvHashHandler, RawMemory, RvDefaultHandler, ert_emit};
use cirrus_volar_vole::{VoleProverContext, VoleVerifierContext, VoleVerifyError};
use hybrid_array::Array;
use rv_asm::{Inst, Reg, Xlen};
use volar_spec::{
    SpecRng,
    field::Galois128,
    ot::IdealCot,
    vole::{Q, Vope, setup::random_nonzero_delta, setup::vole_commit_bit},
};

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

fn program(instructions: impl IntoIterator<Item = Inst>) -> std::vec::Vec<u8> {
    instructions
        .into_iter()
        .flat_map(|instruction| instruction.encode_normal(Xlen::Rv32).to_le_bytes())
        .collect()
}

fn no_hash_prover<C>(_: &mut C, _: &[[Vope<U1, Galois128, U1>; 32]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}
fn no_hash_verifier<C>(_: &mut C, _: &[[Q<U1, Galois128>; 32]]) -> Result<[u8; 32], VoleVerifyError> {
    Ok([0; 32])
}

#[test]
fn ert_add_prover_and_verifier_agree_with_the_native_sum() {
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

    let mut rng = TestRng(0x5EED_1234_ABCD_EF01);
    let delta = random_nonzero_delta::<U1, Galois128, _>(&mut rng, sample_g128, is_zero_g128);
    let cot = IdealCot::<U1, Galois128>::new(delta.clone());

    let mut hats = VecPusher::<Array<Galois128, U1>>::default();
    let mut prover_ctx = VoleProverContext {
        hats: &mut hats,
        bit_to_t,
    };

    let mut prover_registers: [[Vope<U1, Galois128, U1>; 32]; 32] =
        core::array::from_fn(|_| core::array::from_fn(|_| Vope::default()));
    let mut verifier_registers: [[Q<U1, Galois128>; 32]; 32] =
        core::array::from_fn(|_| core::array::from_fn(|_| Q::default()));

    let all_one_prover: [Vope<U1, Galois128, U1>; 32] =
        core::array::from_fn(|_| prover_ctx.create(true).unwrap());
    prover_registers[Reg::A0.0 as usize] = all_one_prover;
    for bit in 0..32 {
        let (vope, q) = vole_commit_bit(&cot, &mut rng, sample_g128, bit_to_t, (left_value >> bit) & 1 != 0);
        prover_registers[Reg::A1.0 as usize][bit] = vope;
        verifier_registers[Reg::A1.0 as usize][bit] = q;
    }
    for bit in 0..32 {
        let (vope, q) = vole_commit_bit(&cot, &mut rng, sample_g128, bit_to_t, (right_value >> bit) & 1 != 0);
        prover_registers[Reg::A2.0 as usize][bit] = vope;
        verifier_registers[Reg::A2.0 as usize][bit] = q;
    }

    let mut prover_constants = [None; 32];
    prover_constants[Reg::A0.0 as usize] = Some(u32::MAX);
    let mut prover_rstack = [0u32; 8];
    let mut prover_vstack: std::vec::Vec<Vope<U1, Galois128, U1>> =
        (0..64).map(|_| Vope::default()).collect();

    let one_prover = prover_ctx.create(true).unwrap();
    let zero_prover = prover_ctx.create(false).unwrap();
    let mut handler = RvDefaultHandler {
        inner: RvHashHandler {
            context: prover_ctx,
            hash: no_hash_prover::<VoleProverContext<U1, Galois128>>,
        },
    };

    let proved = ert_emit(
        &mut handler,
        RawMemory::from(instructions.as_slice()),
        &mut prover_rstack,
        &mut prover_vstack,
        0,
        &mut prover_registers,
        &mut prover_constants,
        zero_prover,
        one_prover,
    );
    assert!(proved.is_ok(), "proving the ert-interpreted add succeeds");
    let prover_result = prover_registers[Reg::T0.0 as usize].clone();
    drop(handler);

    // Verifier side: for A0, build the constant-true share the same way
    // `create` does; the "one"/"zero" ABI arguments `ert_emit` itself needs
    // must match too.
    let mut verifier_ctx = VoleVerifierContext {
        delta: delta.clone(),
        hats: hats.0.clone().into_iter(),
    };
    let all_one_verifier: [Q<U1, Galois128>; 32] =
        core::array::from_fn(|_| verifier_ctx.create(true).unwrap());
    verifier_registers[Reg::A0.0 as usize] = all_one_verifier;

    let mut verifier_constants = [None; 32];
    verifier_constants[Reg::A0.0 as usize] = Some(u32::MAX);
    let mut verifier_rstack = [0u32; 8];
    let mut verifier_vstack: std::vec::Vec<Q<U1, Galois128>> =
        (0..64).map(|_| Q::default()).collect();

    let one_verifier = verifier_ctx.create(true).unwrap();
    let zero_verifier = verifier_ctx.create(false).unwrap();
    let mut verifier_handler = RvDefaultHandler {
        inner: RvHashHandler {
            context: verifier_ctx,
            hash: no_hash_verifier::<VoleVerifierContext<U1, Galois128, std::vec::IntoIter<Array<Galois128, U1>>>>,
        },
    };

    let verified = ert_emit(
        &mut verifier_handler,
        RawMemory::from(instructions.as_slice()),
        &mut verifier_rstack,
        &mut verifier_vstack,
        0,
        &mut verifier_registers,
        &mut verifier_constants,
        zero_verifier,
        one_verifier,
    );
    assert!(verified.is_ok(), "verifying consumes every hat in order");

    let expected = left_value.wrapping_add(right_value);
    for bit in 0..32 {
        // The VOLE relation holds: the verifier's derived share matches
        // what the prover's own (same-process) committed value evaluates
        // to at Delta.
        assert!(
            prover_result[bit].clone() * delta.clone() == verifier_registers[Reg::T0.0 as usize][bit],
            "prover/verifier share mismatch at bit {bit}"
        );
        // The circuit computed the right answer: the prover's committed
        // value's own "value" component (`u[0]`) tracks the plaintext bit
        // through every AND/XOR/mux gate (0/1 field arithmetic mirrors
        // boolean AND/XOR exactly for Galois128), the VOLE analogue of
        // `evaluated.open(&garbled)` in the garbling backend's test.
        let expected_bit = (expected >> bit) & 1 != 0;
        assert_eq!(
            prover_result[bit].u[0][0],
            bit_to_t(expected_bit),
            "result bit {bit} did not match the native ADD"
        );
    }
}
