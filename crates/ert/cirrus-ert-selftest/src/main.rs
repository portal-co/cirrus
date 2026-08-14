#![no_std]
#![no_main]

use core::{array, convert::Infallible, panic::PanicInfo, ptr};

use cirrus_ert::{DefaultHandler, RawMemory, ert_func};
use cirrus_ert_sha256_fixture::sha256_compress;

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
    let mut handler = DefaultHandler {
        context: &mut context,
        hash: &mut hash,
    };
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
        &mut handler,
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
