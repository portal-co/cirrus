#![no_std]
pub use core::arch::asm;
use core::convert::Infallible;
use core::{array, iter};

use rand_core::{TryCryptoRng, TryRng};
use sha2::Digest;
/// Exit the program
pub fn exit<T>() -> T {
    crate::exit_with!()
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
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
#[macro_export]
/// Exit the program, with extra register arguments
macro_rules! exit_with {
    ($($a:tt)*) => {
        loop {}
    };
}
/// Hash a value, with host-provided salts
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
