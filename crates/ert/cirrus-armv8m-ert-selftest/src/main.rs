#![no_std]
#![no_main]

use core::{array, convert::Infallible, panic::PanicInfo, ptr};

use cirrus_armv8m_ert::{DecodeError, DefaultHandler, ErtError, RawMemory, ert_func};
use cirrus_ert_sha256_fixture::sha256_compress;

core::arch::global_asm!(
    r#"
    .section .vector_table, "a"
    .globl __vector_table
__vector_table:
    .word _stack_top
    .word reset_handler
    .rept 14
    .word 0
    .endr

    .section .text.init, "ax"
    .thumb
    .thumb_func
    .globl reset_handler
reset_handler:
    ldr sp, =_stack_top
    ldr r0, =rust_main
    bx r0
1:  b 1b

    .section .text.ert_workload, "ax"
    .thumb
    .thumb_func
    .globl __ert_workload_entry
__ert_workload_entry:
    bl sha256_compress
    mov r1, r0
    movs r0, #0
    subs r0, #1
    svc #0
"#
);

unsafe extern "C" {
    static __ert_workload_entry: u8;
}

const INPUT: [u32; 16] = [0x6162_6380, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24];

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    finish(2)
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
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    let mut rstack = [0; 512];
    let mut vstack = [false; 131_072];
    let args = INPUT.map(|input| (word(input), None));
    let entry = ptr::addr_of!(__ert_workload_entry) as usize as u32 | 1;
    // SAFETY: QEMU maps all linked RAM image bytes at their guest addresses;
    // the interpreted workload fetches only code/static data in that image.
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
        Err(ErtError::Emitted(_)) => finish(0xe3),
        Err(ErtError::Decode(DecodeError::Unsupported(instruction))) => finish(instruction),
        Err(ErtError::Decode(DecodeError::Malformed(instruction))) => finish(instruction),
        Err(ErtError::Decode(DecodeError::Truncated)) => finish(0xe4),
        Err(ErtError::Unexpected) => finish(0xe5),
    };
    if results[0].1 != Some(u32::MAX) {
        finish(0xc4);
    }
    if results[1].0 != word(expected) {
        finish(0xc5);
    }
    if results[1].1.is_some() {
        finish(0xc6);
    }
    finish(0)
}

fn word(value: u32) -> [bool; 32] {
    array::from_fn(|bit| value & (1 << bit) != 0)
}
fn no_hash(_: &[[bool; 32]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}

fn finish(code: u32) -> ! {
    // QEMU's ARM semihosting SYS_EXIT_EXTENDED conveys a success/failure status
    // to the host test after the interpreter has exercised `SVC #0` internally.
    let mut report = *b"ert=00000000\n\0";
    for (shift, byte) in (0..8).rev().zip(&mut report[4..12]) {
        let digit = ((code >> (shift * 4)) & 15) as u8;
        *byte = if digit < 10 {
            b'0' + digit
        } else {
            b'a' + digit - 10
        };
    }
    unsafe {
        core::arch::asm!(
            "mov r0, #4",
            "mov r1, {message}",
            "bkpt 0xab",
            message = in(reg) report.as_ptr(),
            options(nostack),
        );
        if code == 0 {
            core::arch::asm!(
                "mov r0, #0x18",
                "mov r1, r2",
                "bkpt 0xab",
                in("r2") 0x20_026u32,
                options(noreturn)
            );
        }
        EXIT_ARGS[1] = code;
        core::arch::asm!(
            "mov r0, #0x20",
            "mov r1, {args}",
            "bkpt 0xab",
            args = in(reg) ptr::addr_of!(EXIT_ARGS),
            options(noreturn)
        );
    }
}

static mut EXIT_ARGS: [u32; 2] = [0x20_026, 0];
