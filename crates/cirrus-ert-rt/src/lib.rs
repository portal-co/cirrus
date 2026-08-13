#![no_std]
use core::arch::asm;
use core::array;
pub fn hash(mut v: [u8; 32]) -> [u8; 32] {
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
