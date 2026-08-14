#![no_std]
#![no_main]

use core::{array, convert::Infallible, panic::PanicInfo, ptr};

use cirrus_ert::{RawMemory, ert_func};

core::arch::global_asm!(
    r#"
    .section .text.init, "ax"
    .globl _start
_start:
    la t0, __bss_start
    la t1, __bss_end
.Lclear_bss:
    bgeu t0, t1, .Lstart_rust
    sb zero, 0(t0)
    addi t0, t0, 1
    j .Lclear_bss
.Lstart_rust:
    la sp, _stack_top
    call rust_main
.Lhalt:
    wfi
    j .Lhalt

    .section .text.ert_workload, "ax"
    .globl __ert_workload_entry
__ert_workload_entry:
    call sha256_compress
    mv a1, a0
    li a0, -1
    ecall
"#
);

unsafe extern "C" {
    static __ert_workload_entry: u8;
}

const INPUT: [u32; 16] = [0x6162_6380, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24];

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    finish_failure(0xfffe)
}

#[unsafe(no_mangle)]
extern "C" fn rust_main() -> ! {
    let expected = sha256_compress(
        INPUT[0], INPUT[1], INPUT[2], INPUT[3], INPUT[4], INPUT[5], INPUT[6], INPUT[7], INPUT[8],
        INPUT[9], INPUT[10], INPUT[11], INPUT[12], INPUT[13], INPUT[14], INPUT[15],
    );
    let mut context = ();
    let mut hash = no_hash;
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    let mut rstack = [0; 256];
    let mut vstack = [false; 65_536];
    let args = INPUT.map(|input| (word(input), None));
    let entry = ptr::addr_of!(__ert_workload_entry) as usize as u32;
    // SAFETY: QEMU maps this image at its native RV32 addresses. The workload
    // only fetches code and concrete static data from that mapped image.
    let memory = unsafe { RawMemory::new(ptr::null(), None) };

    let results = match ert_func::<_, _, 16, 2>(
        &mut context,
        &mut hash,
        memory,
        &mut rstack,
        &mut vstack,
        entry,
        &mut regs,
        &mut constants,
        false,
        true,
        args,
    ) {
        Ok(results) => results,
        Err(_) => finish_failure(1),
    };

    if results[0].1 != Some(u32::MAX) {
        finish_failure(2);
    }
    if results[1].0 != word(expected) {
        finish_failure(3);
    }
    if results[1].1.is_some() {
        finish_failure(4);
    }
    finish_success()
}

fn word(value: u32) -> [bool; 32] {
    array::from_fn(|bit| value & (1 << bit) != 0)
}

fn no_hash(_: &[[bool; 32]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}

fn finish_success() -> ! {
    finish(0x5555)
}

fn finish_failure(code: u16) -> ! {
    finish((u32::from(code.max(1)) << 16) | 0x3333)
}

fn finish(status: u32) -> ! {
    // SAFETY: QEMU's `virt` machine maps the SiFive test finisher here.
    unsafe { (0x0010_0000 as *mut u32).write_volatile(status) };
    loop {
        // SAFETY: waiting is only reached if QEMU did not consume the finisher write.
        unsafe { core::arch::asm!("wfi", options(nomem, nostack)) };
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn sha256_compress(
    w0: u32,
    w1: u32,
    w2: u32,
    w3: u32,
    w4: u32,
    w5: u32,
    w6: u32,
    w7: u32,
    w8: u32,
    w9: u32,
    w10: u32,
    w11: u32,
    w12: u32,
    w13: u32,
    w14: u32,
    w15: u32,
) -> u32 {
    let mut schedule = [
        w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15,
    ];
    let mut a: u32 = 0x6a09_e667;
    let mut b: u32 = 0xbb67_ae85;
    let mut c: u32 = 0x3c6e_f372;
    let mut d: u32 = 0xa54f_f53a;
    let mut e: u32 = 0x510e_527f;
    let mut f: u32 = 0x9b05_688c;
    let mut g: u32 = 0x1f83_d9ab;
    let mut h: u32 = 0x5be0_cd19;

    let mut round: u32 = 0;
    while round < 64 {
        let slot = (round & 15) as usize;
        let word = if round < 16 {
            schedule[slot]
        } else {
            let value = small_sigma_1(schedule[(round.wrapping_sub(2) & 15) as usize])
                .wrapping_add(schedule[(round.wrapping_sub(7) & 15) as usize])
                .wrapping_add(small_sigma_0(
                    schedule[(round.wrapping_sub(15) & 15) as usize],
                ))
                .wrapping_add(schedule[slot]);
            schedule[slot] = value;
            value
        };
        let temp1 = h
            .wrapping_add(big_sigma_1(e))
            .wrapping_add(choice(e, f, g))
            .wrapping_add(K[round as usize])
            .wrapping_add(word);
        let temp2 = big_sigma_0(a).wrapping_add(majority(a, b, c));
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
        round += 1;
    }
    a.wrapping_add(0x6a09_e667)
}

#[inline(never)]
fn choice(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (!x & z)
}

#[inline(never)]
fn majority(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}

#[inline(never)]
fn big_sigma_0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}

#[inline(never)]
fn big_sigma_1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

#[inline(never)]
fn small_sigma_0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

#[inline(never)]
fn small_sigma_1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

static K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];
