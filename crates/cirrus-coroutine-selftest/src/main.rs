#![no_std]
#![no_main]

use core::{
    future::Future,
    panic::PanicInfo,
    pin::Pin,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

use cirrus_coroutine::{Coroutine, Puller, Pusher};

struct QemuCriticalSection;

critical_section::set_impl!(QemuCriticalSection);

// The self-test is strictly single-core, does not enable interrupts, and never
// invokes either endpoint from an exception handler. Firmware instead obtains
// its real critical-section implementation from its BSP or Embassy.
unsafe impl critical_section::Impl for QemuCriticalSection {
    unsafe fn acquire() -> critical_section::RawRestoreState {
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }

    unsafe fn release(_: critical_section::RawRestoreState) {
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
core::arch::global_asm!(
    r#"
    .section .text.init, "ax"
    .globl _start
_start:
    # Bare-metal QEMU starts with floating-point state disabled. This is a
    # harmless no-op for RV32IM and enables the RV64GC floating self-test.
    li t2, 0x6000
    csrs mstatus, t2
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
"#
);

#[cfg(target_arch = "arm")]
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
"#
);

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    finish(0xfffe)
}

#[unsafe(no_mangle)]
extern "C" fn rust_main() -> ! {
    #[cfg(any(target_arch = "aarch64", target_arch = "riscv64"))]
    run_floating();
    #[cfg(not(any(target_arch = "aarch64", target_arch = "riscv64")))]
    run_integer();
}

fn raw_waker() -> RawWaker {
    fn clone(_: *const ()) -> RawWaker {
        raw_waker()
    }
    fn wake(_: *const ()) {}
    fn wake_by_ref(_: *const ()) {}
    fn drop(_: *const ()) {}

    RawWaker::new(
        core::ptr::null(),
        &RawWakerVTable::new(clone, wake, wake_by_ref, drop),
    )
}

fn next<T, const CAPACITY: usize, const STACK_SLOTS: usize>(
    puller: &mut Puller<'_, T, CAPACITY, STACK_SLOTS>,
) -> T {
    let waker = unsafe { Waker::from_raw(raw_waker()) };
    let mut context = Context::from_waker(&waker);
    let future = puller.next();
    let mut future = core::pin::pin!(future);
    match Future::poll(Pin::as_mut(&mut future), &mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => finish(0xfffd),
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "riscv64")))]
fn integer_producer(pusher: &mut Pusher<'_, u64, 2, 128>) {
    // Retain enough live scalar state across each push to exercise ordinary
    // callee-saved register allocation on all supported architectures.
    let mut value = 0u64;
    let mut mix = [
        0x9e37_79b9_7f4a_7c15u64,
        0xd1b5_4a32_d192_ed03,
        0x94d0_49bb_1331_11eb,
        0xbf58_476d_1ce4_e5b9,
    ];
    loop {
        pusher.push(value);
        value = value.wrapping_add(1);
        for state in &mut mix {
            *state = state.rotate_left(11).wrapping_add(value);
        }
        core::hint::black_box(mix);
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "riscv64")))]
fn run_integer() -> ! {
    let coroutine = Coroutine::new(integer_producer);
    let mut coroutine = core::pin::pin!(coroutine);
    let mut puller = coroutine.as_mut().puller();

    for expected in 0..64 {
        if next(&mut puller) != expected {
            finish(0x1000 + expected as u32);
        }
    }
    finish(0)
}

#[cfg(any(target_arch = "aarch64", target_arch = "riscv64"))]
fn floating_producer(pusher: &mut Pusher<'_, f64, 2, 128>) {
    // The scalar values intentionally remain live across the handoff. The
    // AArch64 and RV64 context switchers preserve their ABI callee-save banks.
    let mut a = 0.5f64;
    let mut b = 1.25f64;
    let mut c = 2.75f64;
    let mut d = 4.5f64;
    loop {
        pusher.push(a + b + c + d);
        a += 0.125;
        b *= 1.000_976_562_5;
        c += b * 0.03125;
        d += c * 0.0078125;
        core::hint::black_box((a, b, c, d));
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "riscv64"))]
fn run_floating() -> ! {
    let coroutine = Coroutine::new(floating_producer);
    let mut coroutine = core::pin::pin!(coroutine);
    let mut puller = coroutine.as_mut().puller();

    let mut a = 0.5f64;
    let mut b = 1.25f64;
    let mut c = 2.75f64;
    let mut d = 4.5f64;
    for index in 0..64 {
        let actual = next(&mut puller);
        let expected = a + b + c + d;
        if actual.to_bits() != expected.to_bits() {
            finish(0x2000 + index);
        }
        a += 0.125;
        b *= 1.000_976_562_5;
        c += b * 0.03125;
        d += c * 0.0078125;
    }
    finish(0)
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
fn finish(code: u32) -> ! {
    // QEMU's virt machine exposes the SiFive test finisher at this address.
    let status = if code == 0 {
        0x5555
    } else {
        (code.max(1) << 16) | 0x3333
    };
    unsafe { (0x0010_0000 as *mut u32).write_volatile(status) };
    loop {
        unsafe { core::arch::asm!("wfi", options(nomem, nostack)) };
    }
}

#[cfg(target_arch = "arm")]
fn finish(code: u32) -> ! {
    let mut report = *b"coroutine=00000000\n\0";
    for (shift, byte) in (0..8).rev().zip(&mut report[10..18]) {
        let digit = ((code >> (shift * 4)) & 15) as u8;
        *byte = if digit < 10 {
            b'0' + digit
        } else {
            b'a' + digit - 10
        };
    }
    let arguments = [0x20_026u32, code];
    unsafe {
        core::arch::asm!(
            "mov r0, #4",
            "mov r1, {message}",
            "bkpt 0xab",
            message = in(reg) report.as_ptr(),
            options(nostack),
        );
        core::arch::asm!(
            "mov r0, #0x20",
            "mov r1, {arguments}",
            "bkpt 0xab",
            arguments = in(reg) arguments.as_ptr(),
            options(noreturn),
        );
    }
}
