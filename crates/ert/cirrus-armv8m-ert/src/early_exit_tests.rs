extern crate std;

use core::convert::Infallible;

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithStorage, ContextWithValue,
    HasError, StorageAddressBit,
};
use cirrus_ert_core::{EarlyExitLoopOptions, EcallOutcome, Handler};
use std::vec::Vec;

use crate::{ArmHandler, ErtError, RawMemory, SecurityAttribute, SecurityState, ert_emit};

fn value(word: &[bool; 32]) -> u32 {
    word.iter()
        .enumerate()
        .fold(0, |value, (bit, set)| value | ((*set as u32) << bit))
}

fn image(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// Byte address of halfword index `i`.
fn addr(i: usize) -> u32 {
    (i * 2) as u32
}

fn movs(rd: u8, imm8: u8) -> u16 {
    0x2000 | ((rd as u16) << 8) | imm8 as u16
}

fn adds_imm3(rd: u8, rn: u8, imm3: u8) -> u16 {
    0x1c00 | ((imm3 as u16) << 6) | ((rn as u16) << 3) | rd as u16
}

fn subs_imm3(rd: u8, rn: u8, imm3: u8) -> u16 {
    0x1e00 | ((imm3 as u16) << 6) | ((rn as u16) << 3) | rd as u16
}

fn subs_reg(rd: u8, rn: u8, rm: u8) -> u16 {
    0x1a00 | ((rm as u16) << 6) | ((rn as u16) << 3) | rd as u16
}

fn ldrb_reg(rt: u8, rn: u8, rm: u8) -> u16 {
    0x5c00 | ((rm as u16) << 6) | ((rn as u16) << 3) | rt as u16
}

fn strb_imm(rt: u8, rn: u8, imm5: u8) -> u16 {
    0x7000 | ((imm5 as u16) << 6) | ((rn as u16) << 3) | rt as u16
}

/// `ADD Rd, SP, #(words*4)` (T1, 16-bit; `words` is the word count, so the
/// byte immediate must be a multiple of 4 -- real hardware's constraint,
/// not just this decoder's).
fn add_reg_sp_imm(rd: u8, words: u8) -> u16 {
    0xa800 | ((rd as u16) << 8) | words as u16
}

/// `SUB SP, SP, #(words*4)`.
fn sub_sp_imm(words: u8) -> u16 {
    0xb080 | words as u16
}

/// `ADD SP, SP, #(words*4)`.
fn add_sp_imm(words: u8) -> u16 {
    0xb000 | words as u16
}

/// `CBZ`/`CBNZ Rn, <target>`; `pc` and `target` are logical (image-relative)
/// byte offsets of this instruction and its target, matching the decoder's
/// own `pc + 4 + imm` formula (the entry address's Thumb bit-0 bias cancels
/// out of the subtraction).
fn cbz_cbnz(nonzero: bool, register: u8, pc: u32, target: u32) -> u16 {
    let imm = target.wrapping_sub(pc).wrapping_sub(4);
    assert!(
        imm % 2 == 0 && imm <= 126,
        "CBZ/CBNZ immediate out of range"
    );
    let bit6 = ((imm >> 6) & 1) as u16;
    let bits51 = ((imm >> 1) & 31) as u16;
    0xb100 | ((nonzero as u16) << 11) | (bit6 << 9) | (bits51 << 3) | register as u16
}

/// Unconditional `B <target>` (T2, 16-bit).
fn b_uncond(pc: u32, target: u32) -> u16 {
    let offset = target.wrapping_sub(pc).wrapping_sub(4) as i32;
    0xe000 | ((offset >> 1) as u16 & 0x7ff)
}

/// `BNE <target>` (T1, 16-bit conditional branch, condition = NE = 1).
/// Unlike `CBZ`/`CBNZ`, `Bcc` can encode a *backward* displacement, which
/// is what a real Thumb toolchain uses for a loop's own concrete-bounded
/// back edge (`CBZ`/`CBNZ` are forward-only on real hardware). Since the
/// loop counter here stays concrete throughout, this resolves through the
/// interpreter's existing concrete-flags branch handling, unmodified by
/// the recognizer under test.
fn bne(pc: u32, target: u32) -> u16 {
    let offset = target.wrapping_sub(pc).wrapping_sub(4) as i32;
    0xd100 | ((offset >> 1) as u16 & 0xff)
}

/// `MOVW Rd, #imm16` (T3, 32-bit, does not set flags).
fn movw(dest: u8, immediate: u16) -> [u16; 2] {
    let imm = immediate as u32;
    let imm11 = (imm >> 11) & 1;
    let imm15_12 = (imm >> 12) & 0xf;
    let imm10_8 = (imm >> 8) & 0x7;
    let imm7_0 = imm & 0xff;
    let first = 0xf240u16 | ((imm11 as u16) << 10) | (imm15_12 as u16);
    let second = ((dest as u16) << 8) | ((imm10_8 as u16) << 12) | (imm7_0 as u16);
    [first, second]
}

const SVC0: u16 = 0xdf00;

/// A minimal handler that exits on concrete `r0 == 1` and, when `enabled`
/// is set, opts into the early-exit-loop recognizer under test.
struct TestHandler {
    enabled: bool,
}

impl HasError for TestHandler {
    type Error = Infallible;
}

impl ContextWithValue<bool> for TestHandler {
    type Wrapped = bool;
}

impl ContextWithBitAnd<bool> for TestHandler {
    fn bitand(&mut self, a: bool, b: bool) -> Result<bool, Infallible> {
        Ok(a & b)
    }
    fn bitand_assign(&mut self, a: &mut bool, b: bool) -> Result<(), Infallible> {
        *a &= b;
        Ok(())
    }
}

impl ContextWithBitOr<bool> for TestHandler {
    fn bitor(&mut self, a: bool, b: bool) -> Result<bool, Infallible> {
        Ok(a | b)
    }
    fn bitor_assign(&mut self, a: &mut bool, b: bool) -> Result<(), Infallible> {
        *a |= b;
        Ok(())
    }
}

impl ContextWithBitXor<bool> for TestHandler {
    fn bitxor(&mut self, a: bool, b: bool) -> Result<bool, Infallible> {
        Ok(a ^ b)
    }
    fn bitxor_assign(&mut self, a: &mut bool, b: bool) -> Result<(), Infallible> {
        *a ^= b;
        Ok(())
    }
}

impl ContextWithStorage<bool> for TestHandler {
    type Storage = [bool];

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<bool>],
    ) -> Result<bool, Infallible> {
        let index = address
            .iter()
            .enumerate()
            .fold(0usize, |index, (bit, address)| {
                index | ((address.wire as usize) << bit)
            });
        Ok(storage[index])
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<bool>],
        value: bool,
    ) -> Result<(), Infallible> {
        let index = address
            .iter()
            .enumerate()
            .fold(0usize, |index, (bit, address)| {
                index | ((address.wire as usize) << bit)
            });
        storage[index] = value;
        Ok(())
    }
}

impl Handler<bool> for TestHandler {
    fn ecall(
        &mut self,
        _regs: &mut [[bool; 32]],
        reg_consts: &mut [Option<u64>],
        _offsets: &mut [Option<i64>],
        _zero: &bool,
        _one: &bool,
    ) -> Result<EcallOutcome, Infallible> {
        match reg_consts[0] {
            Some(1) => Ok(EcallOutcome::Exit),
            _ => Ok(EcallOutcome::Unexpected),
        }
    }

    fn early_exit_loop_options(&self) -> EarlyExitLoopOptions {
        EarlyExitLoopOptions {
            enabled: self.enabled,
            max_lookahead_instructions: 64,
        }
    }
}

impl ArmHandler<bool> for TestHandler {
    fn svc_permitted(&mut self, _state: SecurityState) -> bool {
        true
    }
    fn security_attribute(&mut self, _address: u32) -> SecurityAttribute {
        SecurityAttribute::Secure
    }
}

/// `r2 = 1; for i in 0..len { if a[i] != b[i] { r2 = 0; break; } }`, built
/// into the canonical countdown-loop shape a real Thumb toolchain emits for
/// this idiom: `BNE` (fed by a concrete `SUBS`) as the loop's own back edge,
/// and a `SUBS`-then-`CBZ` as the single secret-dependent early exit, both
/// landing on the same merge point.
///
/// `a`/`b` are stored on the *symbolic* stack (via `STRB`, SP-relative),
/// not embedded in the program image: a load from a concrete `RawMemory`
/// address resolves through the interpreter's existing concrete fast path
/// regardless of what the recognizer does. Register-offset `LDRB`/`STRB`
/// can't take `SP` directly as their base (a 3-bit low-register field), so
/// `r5`/`r6` are set up once as low-register copies of `SP`/`SP+stride`
/// (`ADD Rd, SP, #imm`, which -- like `SUB SP, SP, #imm` itself -- only
/// encodes word-aligned immediates on real hardware, hence rounding the
/// per-buffer stride up to a multiple of 4).
fn memcmp_loop_program(a: &[u8], b: &[u8]) -> Vec<u8> {
    let len = a.len();
    let stride = len.div_ceil(4) * 4;
    let mut w = Vec::new();

    w.push(sub_sp_imm((2 * stride / 4) as u8)); // reserve stack
    w.push(add_reg_sp_imm(5, 0)); // r5 = base_a = sp + 0
    w.push(add_reg_sp_imm(6, (stride / 4) as u8)); // r6 = base_b = sp + stride

    for (k, &byte) in a.iter().enumerate() {
        w.push(movs(4, byte));
        w.push(strb_imm(4, 5, k as u8));
    }
    for (k, &byte) in b.iter().enumerate() {
        w.push(movs(4, byte));
        w.push(strb_imm(4, 6, k as u8));
    }

    w.push(movs(0, 0)); // i = 0
    w.push(movs(7, len as u8)); // remaining = len
    w.push(movs(2, 1)); // result = 1

    let h = w.len();
    w.push(ldrb_reg(3, 5, 0)); // byte_a = *(base_a + i)
    w.push(ldrb_reg(4, 6, 0)); // byte_b = *(base_b + i)
    w.push(subs_reg(3, 3, 4)); // diff = byte_a - byte_b

    let branch = w.len();
    w.push(0); // cbz r3, continue -- patched below

    let exit_prelude = w.len();
    let [movw0, movw1] = movw(2, 0); // result = 0
    w.push(movw0);
    w.push(movw1);
    let jump = w.len();
    w.push(0); // b done -- patched below

    let cont = w.len();
    w.push(adds_imm3(0, 0, 1)); // i += 1
    w.push(subs_imm3(7, 7, 1)); // remaining -= 1

    let latch = w.len();
    w.push(0); // bne H -- patched below

    let done = w.len();
    w.push(add_sp_imm((2 * stride / 4) as u8)); // restore sp
    w.push(movs(0, 1)); // exit signal
    w.push(SVC0);

    w[branch] = cbz_cbnz(false, 3, addr(branch), addr(cont));
    w[jump] = b_uncond(addr(jump), addr(done));
    w[latch] = bne(addr(latch), addr(h));
    let _ = exit_prelude;

    image(&w)
}

fn run_memcmp(a: &[u8], b: &[u8], enabled: bool) -> Result<u32, ErtError<Infallible>> {
    assert_eq!(a.len(), b.len());
    let mem = memcmp_loop_program(a, b);

    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    let mut rstack = [0u32; 8];
    let mut vstack = [false; 4096];
    let mut handler = TestHandler { enabled };
    let storage_bits = vstack.len();

    ert_emit(
        &mut handler,
        &mut vstack,
        storage_bits,
        RawMemory::from(mem.as_slice()),
        &mut rstack,
        1,
        &mut regs,
        &mut constants,
        false,
        true,
    )?;

    // The recognizer's mux clears `constants[2]` once a mismatch could have
    // flipped it, so the actual boolean lives in `regs[2]`'s bit pattern.
    Ok(value(&regs[2]))
}

#[test]
fn disabled_by_default_hard_errors_on_the_secret_dependent_compare() {
    assert!(matches!(
        run_memcmp(b"abcd", b"abcd", false),
        Err(ErtError::Unexpected)
    ));
}

#[test]
fn enabled_recognizes_equal_buffers_across_every_length() {
    for len in 1..=8usize {
        let data = std::vec![1u8; len];
        let result = match run_memcmp(&data, &data, true) {
            Ok(result) => result,
            Err(_) => panic!("recognized loop should not error, len={len}"),
        };
        assert_eq!(result, 1, "len={len}");
    }
}

#[test]
fn enabled_recognizes_a_mismatch_at_every_position() {
    let len = 6usize;
    for mismatch_at in 0..len {
        let a: Vec<u8> = (0..len as u8).collect();
        let mut b = a.clone();
        b[mismatch_at] = b[mismatch_at].wrapping_add(1);
        let result = match run_memcmp(&a, &b, true) {
            Ok(result) => result,
            Err(_) => panic!("recognized loop should not error, mismatch_at={mismatch_at}"),
        };
        assert_eq!(result, 0, "mismatch_at={mismatch_at}");
    }
}
