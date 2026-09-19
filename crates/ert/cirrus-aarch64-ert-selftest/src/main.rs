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
    // Keep the image deliberately inside the current audited ERT subset. This
    // is an encoding/boot fixture; the hash ABI is covered by facade unit tests
    // until a dedicated narrow hash workload is added.
    let mut x = 0x42u64;
    for _ in 0..3 {
        x = x.rotate_left(3) ^ 0x9e37_79b9_7f4a_7c15;
    }
    if x == 0 {
        unsafe {
            asm!("svc #0", in("x0") 1u64);
        }
    }
    unsafe {
        asm!("svc #0", in("x0") u64::MAX);
    }
    loop {}
}

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    loop {}
}
