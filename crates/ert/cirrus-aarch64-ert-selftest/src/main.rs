#![no_std]
#![no_main]

use core::panic::PanicInfo;

use cirrus_ert_rt::asm;

core::arch::global_asm!(
    r#"
    .section .text.init, "ax"
    .globl _start
_start:
    adrp x0, _stack_top
    add x0, x0, :lo12:_stack_top
    mov sp, x0
    bl rust_main
.Lhalt:
    wfi
    b .Lhalt
"#
);

#[unsafe(no_mangle)]
extern "C" fn rust_main() -> ! {
    let input = [0x42u8; 32];
    let digest = cirrus_ert_rt::hash(input);
    let _ = digest;
    unsafe {
        asm!("svc #0", in("x0") u64::MAX);
    }
    loop {}
}

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    loop {}
}
