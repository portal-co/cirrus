#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

//! A bounded, stackful adapter from synchronous [`cirrus_core::Pusher`]s to
//! an async pull interface.
//!
//! [`Coroutine`] owns a fixed-size ring and a second, fixed-size stack for a
//! synchronous producer. A full [`Pusher`] transfers to its paired [`Puller`]
//! through a private transfer waker. When the puller later finds the ring
//! empty, it transfers directly back to the producer. The outer async runtime
//! never schedules this handoff: [`Puller::next`] completes in the poll that
//! starts or resumes the producer.
//!
//! # Safety model
//!
//! This is a single-producer/single-puller, single-thread coroutine. It is
//! deliberately `!Send` and `!Sync`, must remain pinned after [`Puller`] is
//! created, and is not suitable for interrupt handlers. The producer must not
//! return or unwind. Dropping a started coroutine panics; production users
//! should use `panic = "abort"`.
//!
//! The producer stack is measured in 16-byte [`STACK_SLOT_BYTES`] slots. Its
//! size must cover the producer's normal Rust stack frames as well as any
//! callees reached between `push` calls.
//!
//! # Embassy
//!
//! No executor adapter is required. An Embassy task can simply await a pull:
//! `let value = puller.next().await;`. The future resolves from the same poll
//! after the internal symmetric handoff, so it does not register the task's
//! ordinary [`core::task::Waker`].

#[cfg(test)]
extern crate std;

#[cfg(not(cirrus_supported_target))]
compile_error!(
    "cirrus-coroutine supports only aarch64 hosts/aarch64-unknown-none, \
     riscv32im-unknown-none-elf, thumbv8m.main-none-eabi, and \
     riscv64gc-unknown-none-elf"
);

use core::{
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    mem::MaybeUninit,
    pin::Pin,
    ptr::NonNull,
};

use spin::Mutex;

/// The byte size and alignment of one producer-stack slot.
pub const STACK_SLOT_BYTES: usize = 16;

/// A pinned bounded coroutine with one synchronous producer and one puller.
///
/// `F` is called once, on the coroutine's producer stack, after the first
/// empty pull. It must loop forever. Stable Rust cannot currently express a
/// generic `FnOnce(...) -> !` bound, so `F` is accepted as `FnOnce`; if it
/// returns, the coroutine panics instead. A producer written as a looping
/// closure or a function whose declared return type is `()` meets this bound.
/// Wrap a named `fn(...) -> !` in a closure, for example
/// `Coroutine::new(|pusher| producer(pusher))`.
pub struct Coroutine<T, const CAPACITY: usize, const STACK_SLOTS: usize, F> {
    core: Core<T, CAPACITY, STACK_SLOTS>,
    producer: UnsafeCell<Option<F>>,
    // Saved machine contexts are meaningful only on the originating thread.
    not_send_or_sync: PhantomData<*mut ()>,
}

/// The synchronous half of a [`Coroutine`].
///
/// It implements [`cirrus_core::Pusher`]. A full successful push transfers to
/// the paired puller and resumes only after that puller drains the ring.
pub struct Pusher<'a, T, const CAPACITY: usize, const STACK_SLOTS: usize> {
    core: NonNull<Core<T, CAPACITY, STACK_SLOTS>>,
    marker: PhantomData<&'a Core<T, CAPACITY, STACK_SLOTS>>,
    not_send_or_sync: PhantomData<*mut ()>,
}

/// The async half of a [`Coroutine`].
///
/// Only one puller can be created. [`Puller::next`] never returns `Pending`;
/// an empty ring directly resumes the synchronous producer until it fills the
/// ring and transfers back.
pub struct Puller<'a, T, const CAPACITY: usize, const STACK_SLOTS: usize> {
    core: NonNull<Core<T, CAPACITY, STACK_SLOTS>>,
    marker: PhantomData<&'a mut Core<T, CAPACITY, STACK_SLOTS>>,
    not_send_or_sync: PhantomData<*mut ()>,
}

#[repr(C, align(16))]
struct StackSlot([u8; STACK_SLOT_BYTES]);

struct Core<T, const CAPACITY: usize, const STACK_SLOTS: usize> {
    ring: Mutex<Ring<T, CAPACITY>>,
    producer_context: UnsafeCell<context::Context>,
    consumer_context: UnsafeCell<context::Context>,
    entry: UnsafeCell<Option<Entry>>,
    started: Cell<bool>,
    claimed: Cell<bool>,
    stack: [MaybeUninit<StackSlot>; STACK_SLOTS],
}

#[derive(Clone, Copy)]
struct Entry {
    data: *mut (),
    call: unsafe extern "C" fn(*mut ()) -> !,
}

struct Ring<T, const CAPACITY: usize> {
    storage: MaybeUninit<[MaybeUninit<T>; CAPACITY]>,
    head: usize,
    len: usize,
}

impl<T, const CAPACITY: usize> Ring<T, CAPACITY> {
    const fn new() -> Self {
        Self {
            storage: MaybeUninit::uninit(),
            head: 0,
            len: 0,
        }
    }

    fn push(&mut self, value: T) -> bool {
        debug_assert!(CAPACITY > 0);
        debug_assert!(self.len < CAPACITY);

        let index = (self.head + self.len) % CAPACITY;
        // SAFETY: `index` is within the backing array and points at an empty
        // slot because `len < CAPACITY`.
        unsafe {
            self.storage
                .as_mut_ptr()
                .cast::<MaybeUninit<T>>()
                .add(index)
                .write(MaybeUninit::new(value));
        }
        self.len += 1;
        self.len == CAPACITY
    }

    fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        let index = self.head;
        self.head = (self.head + 1) % CAPACITY;
        self.len -= 1;
        // SAFETY: `index` was in the initialized prefix of the ring before
        // decrementing `len`, and `read` leaves that slot uninitialized.
        Some(unsafe {
            self.storage
                .as_mut_ptr()
                .cast::<MaybeUninit<T>>()
                .add(index)
                .read()
                .assume_init()
        })
    }
}

impl<T, const CAPACITY: usize> Drop for Ring<T, CAPACITY> {
    fn drop(&mut self) {
        for offset in 0..self.len {
            let index = (self.head + offset) % CAPACITY;
            // SAFETY: the logical ring range is exactly its initialized range.
            unsafe {
                self.storage
                    .as_mut_ptr()
                    .cast::<MaybeUninit<T>>()
                    .add(index)
                    .read()
                    .assume_init_drop();
            }
        }
    }
}

impl<T, const CAPACITY: usize, const STACK_SLOTS: usize> Core<T, CAPACITY, STACK_SLOTS> {
    const fn new() -> Self {
        Self {
            ring: Mutex::new(Ring::new()),
            producer_context: UnsafeCell::new(context::Context::EMPTY),
            consumer_context: UnsafeCell::new(context::Context::EMPTY),
            entry: UnsafeCell::new(None),
            started: Cell::new(false),
            claimed: Cell::new(false),
            stack: [const { MaybeUninit::uninit() }; STACK_SLOTS],
        }
    }

    fn stack_top(&self) -> *mut u8 {
        // SAFETY: one-past-the-end is valid to form and is the aligned top of
        // the owned producer stack.
        unsafe { self.stack.as_ptr().add(STACK_SLOTS).cast::<u8>() as *mut u8 }
    }
}

impl<T, const CAPACITY: usize, const STACK_SLOTS: usize, F> Coroutine<T, CAPACITY, STACK_SLOTS, F> {
    /// Create an unstarted coroutine.
    ///
    /// The value may move until [`Coroutine::puller`] is called through a
    /// pinned mutable reference. Panics if the ring has zero capacity or the
    /// producer stack has zero 16-byte slots.
    pub const fn new(producer: F) -> Self {
        assert!(CAPACITY != 0, "a coroutine ring requires nonzero capacity");
        assert!(
            STACK_SLOTS != 0,
            "a coroutine requires at least one 16-byte producer-stack slot"
        );
        Self {
            core: Core::new(),
            producer: UnsafeCell::new(Some(producer)),
            not_send_or_sync: PhantomData,
        }
    }
}

impl<T, const CAPACITY: usize, const STACK_SLOTS: usize, F> Coroutine<T, CAPACITY, STACK_SLOTS, F>
where
    F: for<'p> FnOnce(&'p mut Pusher<'p, T, CAPACITY, STACK_SLOTS>),
{
    /// Claim the unique pull side of this pinned coroutine.
    ///
    /// Panics if a puller has already been created for this coroutine.
    pub fn puller(self: Pin<&mut Self>) -> Puller<'_, T, CAPACITY, STACK_SLOTS> {
        // SAFETY: the caller pins `self`; from here every raw pointer stored
        // in the entry/context refers to that stable allocation.
        let this = unsafe { self.get_unchecked_mut() };
        assert!(
            !this.core.claimed.replace(true),
            "a coroutine has exactly one puller"
        );

        let data = (this as *mut Self).cast::<()>();
        // SAFETY: `entry` is initialized exactly once before the producer can
        // be started, and `data` stays valid while Puller borrows `self`.
        unsafe {
            *this.core.entry.get() = Some(Entry {
                data,
                call: producer_entry::<T, CAPACITY, STACK_SLOTS, F>,
            });
        }

        Puller {
            core: NonNull::from(&mut this.core),
            marker: PhantomData,
            not_send_or_sync: PhantomData,
        }
    }
}

impl<T, const CAPACITY: usize, const STACK_SLOTS: usize, F> Drop
    for Coroutine<T, CAPACITY, STACK_SLOTS, F>
{
    fn drop(&mut self) {
        assert!(
            !self.core.started.get(),
            "dropping a started coroutine would abandon its producer stack"
        );
    }
}

impl<'a, T, const CAPACITY: usize, const STACK_SLOTS: usize> Pusher<'a, T, CAPACITY, STACK_SLOTS> {
    fn new(core: NonNull<Core<T, CAPACITY, STACK_SLOTS>>) -> Self {
        Self {
            core,
            marker: PhantomData,
            not_send_or_sync: PhantomData,
        }
    }

    /// Push one value, transferring to the puller if this fills the ring.
    #[inline]
    pub fn push(&mut self, value: T) {
        // SAFETY: a Pusher is made only by the producer entry point, and the
        // coroutine protocol ensures the core outlives the producer stack.
        let core = unsafe { self.core.as_ref() };
        let full = {
            let mut ring = core.ring.lock();
            assert!(
                ring.len < CAPACITY,
                "a producer resumed before its puller drained the ring"
            );
            ring.push(value)
        };

        if full {
            // The guard above has been dropped before the control transfer.
            unsafe { TransferWaker { core: self.core }.wake() };
        }
    }
}

impl<'a, T, const CAPACITY: usize, const STACK_SLOTS: usize> cirrus_core::Pusher<T>
    for Pusher<'a, T, CAPACITY, STACK_SLOTS>
{
    fn push(&mut self, value: T) {
        Self::push(self, value);
    }
}

impl<'a, T, const CAPACITY: usize, const STACK_SLOTS: usize> Puller<'a, T, CAPACITY, STACK_SLOTS> {
    /// Pull the next FIFO value, starting or refilling the producer as needed.
    pub async fn next(&mut self) -> T {
        self.take_or_refill()
    }

    fn take_or_refill(&mut self) -> T {
        // SAFETY: Puller is the unique public handle to this core and its
        // lifetime is tied to the pinned coroutine that owns the storage.
        let core = unsafe { self.core.as_ref() };
        if let Some(value) = core.ring.lock().pop() {
            return value;
        }

        if !core.started.replace(true) {
            // SAFETY: `puller` initializes entry exactly once before exposing
            // this handle; this is the first activation of producer_context.
            let entry =
                unsafe { (*core.entry.get()).expect("a puller always has a producer entry") };
            unsafe {
                context::initialize(
                    &mut *core.producer_context.get(),
                    core.stack_top(),
                    entry.call as *const () as usize,
                    entry.data as usize,
                );
            }
        }

        // `transfer` returns only after a full Pusher invokes its transfer
        // waker, restoring this saved pull continuation.
        unsafe {
            context::transfer(core.consumer_context.get(), core.producer_context.get());
        }

        core.ring
            .lock()
            .pop()
            .expect("a producer transfer must leave a full ring")
    }
}

struct TransferWaker<T, const CAPACITY: usize, const STACK_SLOTS: usize> {
    core: NonNull<Core<T, CAPACITY, STACK_SLOTS>>,
}

impl<T, const CAPACITY: usize, const STACK_SLOTS: usize> TransferWaker<T, CAPACITY, STACK_SLOTS> {
    /// Resume the saved pull continuation. This is intentionally distinct from
    /// `core::task::Waker`: it performs a non-scheduling symmetric transfer.
    unsafe fn wake(self) {
        // SAFETY: Pusher runs only after Puller has saved consumer_context, and
        // `switch` restores that context before this function can return.
        let core = unsafe { self.core.as_ref() };
        unsafe {
            context::transfer(core.producer_context.get(), core.consumer_context.get());
        }
    }
}

unsafe extern "C" fn producer_entry<T, const CAPACITY: usize, const STACK_SLOTS: usize, F>(
    data: *mut (),
) -> !
where
    F: for<'p> FnOnce(&'p mut Pusher<'p, T, CAPACITY, STACK_SLOTS>),
{
    // SAFETY: `data` is established from the pinned Coroutine in `puller` and
    // the producer stack cannot start after that Coroutine is dropped.
    let coroutine = unsafe { &mut *data.cast::<Coroutine<T, CAPACITY, STACK_SLOTS, F>>() };
    // SAFETY: the entry trampoline runs exactly once, so it exclusively takes
    // the producer closure from its slot.
    let producer = unsafe {
        (*coroutine.producer.get())
            .take()
            .expect("a coroutine producer starts exactly once")
    };
    let mut pusher = Pusher::new(NonNull::from(&mut coroutine.core));
    producer(&mut pusher);
    panic!("a coroutine producer must not return");
}

#[cfg(target_arch = "aarch64")]
mod context {
    use core::arch::global_asm;

    #[repr(C, align(16))]
    pub(super) struct Context {
        // x19 through x29.
        x: [usize; 11],
        sp: usize,
        lr: usize,
        // d8 through d15. The ABI only requires the scalar lower 64 bits;
        // NEON/SVE state is intentionally not part of this coroutine ABI.
        d: [u64; 8],
    }

    impl Context {
        pub(super) const EMPTY: Self = Self {
            x: [0; 11],
            sp: 0,
            lr: 0,
            d: [0; 8],
        };
    }

    unsafe extern "C" {
        fn __cirrus_coroutine_switch(current: *mut Context, next: *const Context);
        fn __cirrus_coroutine_start();
    }

    global_asm!(
        r#"
        .text
        .p2align 2
        .globl {switch}
{switch}:
        stp x19, x20, [x0, #0]
        stp x21, x22, [x0, #16]
        stp x23, x24, [x0, #32]
        stp x25, x26, [x0, #48]
        stp x27, x28, [x0, #64]
        str x29, [x0, #80]
        mov x2, sp
        str x2, [x0, #88]
        str x30, [x0, #96]
        str d8, [x0, #104]
        str d9, [x0, #112]
        str d10, [x0, #120]
        str d11, [x0, #128]
        str d12, [x0, #136]
        str d13, [x0, #144]
        str d14, [x0, #152]
        str d15, [x0, #160]

        ldp x19, x20, [x1, #0]
        ldp x21, x22, [x1, #16]
        ldp x23, x24, [x1, #32]
        ldp x25, x26, [x1, #48]
        ldp x27, x28, [x1, #64]
        ldr x29, [x1, #80]
        ldr x2, [x1, #88]
        mov sp, x2
        ldr x30, [x1, #96]
        ldr d8, [x1, #104]
        ldr d9, [x1, #112]
        ldr d10, [x1, #120]
        ldr d11, [x1, #128]
        ldr d12, [x1, #136]
        ldr d13, [x1, #144]
        ldr d14, [x1, #152]
        ldr d15, [x1, #160]
        ret

        .p2align 2
        .globl {start}
{start}:
        mov x0, x19
        br x20
"#,
        switch = sym __cirrus_coroutine_switch,
        start = sym __cirrus_coroutine_start,
    );

    pub(super) unsafe fn initialize(
        context: &mut Context,
        stack_top: *mut u8,
        entry: usize,
        data: usize,
    ) {
        *context = Context::EMPTY;
        context.x[0] = data;
        context.x[1] = entry;
        context.sp = stack_top as usize;
        context.lr = __cirrus_coroutine_start as *const () as usize;
    }

    pub(super) unsafe fn transfer(current: *mut Context, next: *const Context) {
        unsafe { __cirrus_coroutine_switch(current, next) };
    }
}

#[cfg(target_arch = "riscv32")]
mod context {
    use core::arch::global_asm;

    #[repr(C, align(16))]
    pub(super) struct Context {
        // s0 through s11.
        s: [usize; 12],
        sp: usize,
        ra: usize,
    }

    impl Context {
        pub(super) const EMPTY: Self = Self {
            s: [0; 12],
            sp: 0,
            ra: 0,
        };
    }

    unsafe extern "C" {
        fn __cirrus_coroutine_switch(current: *mut Context, next: *const Context);
        fn __cirrus_coroutine_start();
    }

    global_asm!(
        r#"
        .text
        .p2align 2
        .globl {switch}
{switch}:
        sw s0, 0(a0)
        sw s1, 4(a0)
        sw s2, 8(a0)
        sw s3, 12(a0)
        sw s4, 16(a0)
        sw s5, 20(a0)
        sw s6, 24(a0)
        sw s7, 28(a0)
        sw s8, 32(a0)
        sw s9, 36(a0)
        sw s10, 40(a0)
        sw s11, 44(a0)
        sw sp, 48(a0)
        sw ra, 52(a0)

        lw s0, 0(a1)
        lw s1, 4(a1)
        lw s2, 8(a1)
        lw s3, 12(a1)
        lw s4, 16(a1)
        lw s5, 20(a1)
        lw s6, 24(a1)
        lw s7, 28(a1)
        lw s8, 32(a1)
        lw s9, 36(a1)
        lw s10, 40(a1)
        lw s11, 44(a1)
        lw t0, 48(a1)
        mv sp, t0
        lw ra, 52(a1)
        ret

        .p2align 2
        .globl {start}
{start}:
        mv a0, s0
        jr s1
"#,
        switch = sym __cirrus_coroutine_switch,
        start = sym __cirrus_coroutine_start,
    );

    pub(super) unsafe fn initialize(
        context: &mut Context,
        stack_top: *mut u8,
        entry: usize,
        data: usize,
    ) {
        *context = Context::EMPTY;
        context.s[0] = data;
        context.s[1] = entry;
        context.sp = stack_top as usize;
        context.ra = __cirrus_coroutine_start as *const () as usize;
    }

    pub(super) unsafe fn transfer(current: *mut Context, next: *const Context) {
        unsafe { __cirrus_coroutine_switch(current, next) };
    }
}

#[cfg(target_arch = "riscv64")]
mod context {
    use core::arch::global_asm;

    #[repr(C, align(16))]
    pub(super) struct Context {
        // s0 through s11.
        s: [usize; 12],
        sp: usize,
        ra: usize,
        // fs0 through fs11: the scalar callee-saved floating-point state.
        fs: [u64; 12],
    }

    impl Context {
        pub(super) const EMPTY: Self = Self {
            s: [0; 12],
            sp: 0,
            ra: 0,
            fs: [0; 12],
        };
    }

    unsafe extern "C" {
        fn __cirrus_coroutine_switch(current: *mut Context, next: *const Context);
        fn __cirrus_coroutine_start();
    }

    global_asm!(
        r#"
        .text
        .p2align 2
        .option push
        .option arch, rv64imafdc
        .globl {switch}
{switch}:
        sd s0, 0(a0)
        sd s1, 8(a0)
        sd s2, 16(a0)
        sd s3, 24(a0)
        sd s4, 32(a0)
        sd s5, 40(a0)
        sd s6, 48(a0)
        sd s7, 56(a0)
        sd s8, 64(a0)
        sd s9, 72(a0)
        sd s10, 80(a0)
        sd s11, 88(a0)
        sd sp, 96(a0)
        sd ra, 104(a0)
        fsd fs0, 112(a0)
        fsd fs1, 120(a0)
        fsd fs2, 128(a0)
        fsd fs3, 136(a0)
        fsd fs4, 144(a0)
        fsd fs5, 152(a0)
        fsd fs6, 160(a0)
        fsd fs7, 168(a0)
        fsd fs8, 176(a0)
        fsd fs9, 184(a0)
        fsd fs10, 192(a0)
        fsd fs11, 200(a0)

        ld s0, 0(a1)
        ld s1, 8(a1)
        ld s2, 16(a1)
        ld s3, 24(a1)
        ld s4, 32(a1)
        ld s5, 40(a1)
        ld s6, 48(a1)
        ld s7, 56(a1)
        ld s8, 64(a1)
        ld s9, 72(a1)
        ld s10, 80(a1)
        ld s11, 88(a1)
        ld t0, 96(a1)
        mv sp, t0
        ld ra, 104(a1)
        fld fs0, 112(a1)
        fld fs1, 120(a1)
        fld fs2, 128(a1)
        fld fs3, 136(a1)
        fld fs4, 144(a1)
        fld fs5, 152(a1)
        fld fs6, 160(a1)
        fld fs7, 168(a1)
        fld fs8, 176(a1)
        fld fs9, 184(a1)
        fld fs10, 192(a1)
        fld fs11, 200(a1)
        ret

        .p2align 2
        .globl {start}
{start}:
        mv a0, s0
        jr s1
        .option pop
"#,
        switch = sym __cirrus_coroutine_switch,
        start = sym __cirrus_coroutine_start,
    );

    pub(super) unsafe fn initialize(
        context: &mut Context,
        stack_top: *mut u8,
        entry: usize,
        data: usize,
    ) {
        *context = Context::EMPTY;
        context.s[0] = data;
        context.s[1] = entry;
        context.sp = stack_top as usize;
        context.ra = __cirrus_coroutine_start as *const () as usize;
    }

    pub(super) unsafe fn transfer(current: *mut Context, next: *const Context) {
        unsafe { __cirrus_coroutine_switch(current, next) };
    }
}

#[cfg(target_arch = "arm")]
mod context {
    use core::arch::global_asm;

    #[repr(C, align(16))]
    pub(super) struct Context {
        // r4 through r11; r9 is retained even on platforms that reserve it.
        r: [u32; 8],
        sp: u32,
        lr: u32,
    }

    impl Context {
        pub(super) const EMPTY: Self = Self {
            r: [0; 8],
            sp: 0,
            lr: 0,
        };
    }

    unsafe extern "C" {
        fn __cirrus_coroutine_switch(current: *mut Context, next: *const Context);
        fn __cirrus_coroutine_start();
    }

    global_asm!(
        r#"
        .syntax unified
        .thumb
        .text
        .p2align 2
        .global {switch}
        .thumb_func
{switch}:
        str r4, [r0, #0]
        str r5, [r0, #4]
        str r6, [r0, #8]
        str r7, [r0, #12]
        str r8, [r0, #16]
        str r9, [r0, #20]
        str r10, [r0, #24]
        str r11, [r0, #28]
        mov r2, sp
        str r2, [r0, #32]
        str lr, [r0, #36]

        ldr r4, [r1, #0]
        ldr r5, [r1, #4]
        ldr r6, [r1, #8]
        ldr r7, [r1, #12]
        ldr r8, [r1, #16]
        ldr r9, [r1, #20]
        ldr r10, [r1, #24]
        ldr r11, [r1, #28]
        ldr r2, [r1, #32]
        mov sp, r2
        ldr lr, [r1, #36]
        bx lr

        .p2align 2
        .global {start}
        .thumb_func
{start}:
        mov r0, r4
        bx r5
"#,
        switch = sym __cirrus_coroutine_switch,
        start = sym __cirrus_coroutine_start,
    );

    pub(super) unsafe fn initialize(
        context: &mut Context,
        stack_top: *mut u8,
        entry: usize,
        data: usize,
    ) {
        *context = Context::EMPTY;
        context.r[0] = data as u32;
        context.r[1] = entry as u32;
        context.sp = stack_top as usize as u32;
        context.lr = (__cirrus_coroutine_start as *const () as usize | 1) as u32;
    }

    pub(super) unsafe fn transfer(current: *mut Context, next: *const Context) {
        unsafe { __cirrus_coroutine_switch(current, next) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::{
        future::Future,
        mem::ManuallyDrop,
        pin::Pin,
        task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    fn block_on_ready<F: Future>(future: F) -> F::Output {
        let waker = unsafe { Waker::from_raw(raw_waker()) };
        let mut context = Context::from_waker(&waker);
        let mut future = core::pin::pin!(future);
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("Puller::next must complete in its first poll"),
        }
    }

    fn numbers(pusher: &mut Pusher<'_, u32, 2, 64>) {
        let mut next = 0;
        loop {
            cirrus_core::Pusher::push(pusher, next);
            next += 1;
        }
    }

    #[test]
    fn fifo_order_survives_repeated_full_empty_handoffs() {
        let mut coroutine = ManuallyDrop::new(Coroutine::new(numbers));
        // SAFETY: ManuallyDrop prevents the intentional started-coroutine drop
        // panic at the end of this test; the stack remains valid throughout.
        let pinned = unsafe { Pin::new_unchecked(&mut *coroutine) };
        let mut puller = pinned.puller();

        for expected in 0..32 {
            assert_eq!(block_on_ready(puller.next()), expected);
        }
    }

    static DROPPED: AtomicUsize = AtomicUsize::new(0);

    struct DropProbe(usize);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            DROPPED.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn non_copy_values(pusher: &mut Pusher<'_, DropProbe, 3, 64>) {
        let mut next = 0;
        loop {
            pusher.push(DropProbe(next));
            next += 1;
        }
    }

    #[test]
    fn supports_non_copy_values() {
        DROPPED.store(0, Ordering::SeqCst);
        let mut coroutine = ManuallyDrop::new(Coroutine::new(non_copy_values));
        // SAFETY: see fifo_order_survives_repeated_full_empty_handoffs.
        let pinned = unsafe { Pin::new_unchecked(&mut *coroutine) };
        let mut puller = pinned.puller();

        let first = block_on_ready(puller.next());
        let second = block_on_ready(puller.next());
        assert_eq!((first.0, second.0), (0, 1));
        drop(first);
        drop(second);
        assert_eq!(DROPPED.load(Ordering::SeqCst), 2);
    }

    fn one_at_a_time(pusher: &mut Pusher<'_, u8, 1, 64>) {
        let mut next = 0;
        loop {
            pusher.push(next);
            next = next.wrapping_add(1);
        }
    }

    #[test]
    fn capacity_one_handoffs_every_value() {
        let mut coroutine = ManuallyDrop::new(Coroutine::new(one_at_a_time));
        // SAFETY: see fifo_order_survives_repeated_full_empty_handoffs.
        let pinned = unsafe { Pin::new_unchecked(&mut *coroutine) };
        let mut puller = pinned.puller();

        for expected in 0..32 {
            assert_eq!(block_on_ready(puller.next()), expected);
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "riscv64"))]
    fn floating_values(pusher: &mut Pusher<'_, f64, 2, 64>) {
        let mut value = 0.25f64;
        let mut scale = 1.5f64;
        loop {
            pusher.push(value * scale);
            value += 0.125;
            scale += 0.03125;
            core::hint::black_box((value, scale));
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "riscv64"))]
    #[test]
    fn scalar_floats_survive_repeated_handoffs() {
        let mut coroutine = ManuallyDrop::new(Coroutine::new(floating_values));
        // SAFETY: see fifo_order_survives_repeated_full_empty_handoffs.
        let pinned = unsafe { Pin::new_unchecked(&mut *coroutine) };
        let mut puller = pinned.puller();
        let mut value = 0.25f64;
        let mut scale = 1.5f64;

        for _ in 0..32 {
            assert_eq!(
                block_on_ready(puller.next()).to_bits(),
                (value * scale).to_bits()
            );
            value += 0.125;
            scale += 0.03125;
        }
    }

    #[test]
    fn rejects_zero_capacity_before_starting() {
        let result = std::panic::catch_unwind(|| {
            let mut coroutine = Coroutine::new(|_: &mut Pusher<'_, u8, 0, 1>| {});
            // SAFETY: this reaches only the constructor's parameter check.
            let pinned = unsafe { Pin::new_unchecked(&mut coroutine) };
            let _ = pinned.puller();
        });
        assert!(result.is_err());
    }
}
