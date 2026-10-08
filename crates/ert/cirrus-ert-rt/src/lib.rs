#![no_std]
pub use core::arch::asm;
use core::array;
use core::convert::Infallible;

use rand_core::{TryCryptoRng, TryRng};
use sha2::Digest;

#[cfg(target_arch = "arm")]
core::arch::global_asm!(
    r#"
    .syntax unified
    .thumb
    .global __cirrus_ert_hash_words
    .type __cirrus_ert_hash_words,%function
    .thumb_func
__cirrus_ert_hash_words:
    push.w {{r4, r5, r6, r7, r8, lr}}
    mov r12, r0
    ldr r1, [r12, #0]
    ldr r2, [r12, #4]
    ldr r3, [r12, #8]
    ldr r4, [r12, #12]
    ldr r5, [r12, #16]
    ldr r6, [r12, #20]
    ldr r7, [r12, #24]
    ldr r8, [r12, #28]
    movs r0, #0
    svc #0
    str r1, [r12, #0]
    str r2, [r12, #4]
    str r3, [r12, #8]
    str r4, [r12, #12]
    str r5, [r12, #16]
    str r6, [r12, #20]
    str r7, [r12, #24]
    str r8, [r12, #28]
    pop.w {{r4, r5, r6, r7, r8, pc}}
"#
);

#[cfg(target_arch = "arm")]
unsafe extern "C" {
    fn __cirrus_ert_hash_words(words: *mut u32);
}
/// Exit the program
#[inline(always)]
pub fn exit<T>() -> T {
    crate::exit_with!()
}
/// Zero natively; the interpreter can overlay this address with a nonzero
/// value via `RawMemory::with_ert_detect` so guest code can tell whether it
/// is running under the Cirrus ERT.
#[unsafe(no_mangle)]
pub static __CIRRUS_ERT_DETECT_FLAG: u32 = 0;
/// True when running under the interpreter, false natively.
#[inline(always)]
pub fn is_ert() -> bool {
    // SAFETY: reading a valid, always-initialized static's own address.
    unsafe { core::ptr::read_volatile(&raw const __CIRRUS_ERT_DETECT_FLAG) != 0 }
}
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[macro_export]
/// Exit the program, with extra register arguments
macro_rules! exit_with {
    ($($a:tt)*) => {
        loop{
            unsafe{
                $crate::asm!("ecall", in("a0") 0xffff_ffff, $($a)*)
            }
        }
    };
}
#[cfg(target_arch = "aarch64")]
#[macro_export]
/// Exit the program through the AArch64 bare-metal ERT `SVC #0` convention.
macro_rules! exit_with {
    ($($a:tt)*) => {
        loop {
            unsafe {
                $crate::asm!("svc #0", in("x0") u64::MAX, $($a)*)
            }
        }
    };
}

#[cfg(target_arch = "arm")]
#[macro_export]
/// Exit the program through the Armv8-M ERT `SVC #0` convention.
macro_rules! exit_with {
    ($($a:tt)*) => {
        loop {
            unsafe {
                $crate::asm!("svc 0", in("r0") u32::MAX, $($a)*)
            }
        }
    };
}
#[cfg(not(any(
    target_arch = "riscv32",
    target_arch = "riscv64",
    target_arch = "arm",
    target_arch = "aarch64"
)))]
#[macro_export]
/// Exit the program, with extra register arguments
macro_rules! exit_with {
    ($($a:tt)*) => {
        loop {}
    };
}
/// Hash a value, with host-provided salts.
///
/// On Armv8-M this sends the eight little-endian words through the ERT
/// `SVC #0` hash convention (`r0 = 0`, payload/result in `r1` through `r8`).
/// On RV32 the payload moves through eight 32-bit registers starting at `a1`;
/// on RV64 the same 32 bytes move through four 64-bit registers.
pub fn hash(mut v: [u8; 32]) -> [u8; 32] {
    v = sha2::Sha256::digest(&v).0;
    #[cfg(target_arch = "riscv32")]
    {
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] =
            array::from_fn(|i| u32::from_le_bytes(array::from_fn(|j| v[j + i * 4])));
        unsafe {
            asm!("ecall", in("a0") 0, a = inout("a1") a, b = inout("x12") b, c = inout("x13") c, d = input("x14") d, e = inout("x15") e, f = inout("x16") f, g = input("x17") g, h = input("x18") h);
        }
        for (i, b) in [a, b, c, d, e, f, g, h]
            .into_iter()
            .flat_map(|a| a.to_le_bytes())
            .enumerate()
        {
            v[i] = b
        }
        return v;
    }
    #[cfg(target_arch = "riscv64")]
    {
        // The RV64 ERT hash `ECALL` moves the same 32-byte payload through
        // four 64-bit registers starting at `a1`.
        let [mut a, mut b, mut c, mut d] =
            array::from_fn(|i| u64::from_le_bytes(array::from_fn(|j| v[j + i * 8])));
        unsafe {
            asm!("ecall", in("a0") 0, a = inout("a1") a, b = inout("a2") b, c = inout("a3") c, d = inout("a4") d);
        }
        for (i, b) in [a, b, c, d]
            .into_iter()
            .flat_map(|a| a.to_le_bytes())
            .enumerate()
        {
            v[i] = b
        }
        return v;
    }
    #[cfg(target_arch = "arm")]
    {
        let mut words: [u32; 8] =
            array::from_fn(|i| u32::from_le_bytes(array::from_fn(|j| v[j + i * 4])));
        unsafe {
            __cirrus_ert_hash_words(words.as_mut_ptr());
        }
        for (index, byte) in words
            .into_iter()
            .flat_map(|word| word.to_le_bytes())
            .enumerate()
        {
            v[index] = byte;
        }
        return v;
    }
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "arm")))]
    unreachable!()
}
/// Sponge construction/XOF of [`hash`]
pub fn hash_many(x: &[u8]) -> Xof {
    let mut state = [0xff; 32];
    for c in x.chunks(16) {
        state = hash(state);
        state[0..16].fill(0x00);
        state[0..(c.len())].copy_from_slice(c);
    }
    state = hash(state);
    return Xof {
        state: hash(state),
        s: array::from_fn(|i| state[i]),
        i: 0,
    };
}
#[derive(Clone)]
pub struct Xof {
    state: [u8; 32],
    s: [u8; 16],
    i: u8,
}
impl Xof {
    pub fn next_byte(&mut self) -> u8 {
        let r = self.s[self.i as usize];
        self.i += 1;
        if self.i == 16 {
            self.s = array::from_fn(|i| self.state[i]);
            self.state = hash(self.state);
            self.i = 0;
        };
        r
    }
}
impl Iterator for Xof {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.next_byte())
    }
}
impl TryRng for Xof {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        rand_core::utils::next_word_via_fill(self)
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        rand_core::utils::next_word_via_fill(self)
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        for d in dst.iter_mut() {
            *d = self.next_byte();
        }
        Ok(())
    }
}
impl TryCryptoRng for Xof {}
