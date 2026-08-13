#![no_std]
use core::arch::asm;
use core::{array, iter};

use sha2::Digest;
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
pub fn hash_many(x: &[u8]) -> impl Iterator<Item = u8> {
    let mut state = [0xff; 32];
    for c in x.chunks(16) {
        state = hash(state);
        state[0..16].fill(0x00);
        state[0..(c.len())].copy_from_slice(c);
    }
    state = hash(state);
    return iter::from_fn(move || {
        let ext: [u8; 16] = array::from_fn(|i| state[i]);
        state = hash(state);
        Some(ext)
    })
    .flatten();
}
