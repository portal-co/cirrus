#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

//! A bounded, stackful adapter from synchronous [`cirrus_core::Pusher`]s to
//! synchronous and async pull interfaces.
//!
//! [`Coroutine`] owns a fixed-size ring and a second, fixed-size stack for a
//! synchronous producer. A full [`Pusher`] transfers to its paired [`Puller`]
//! through a private transfer waker. When the puller later finds the ring
//! empty, it transfers directly back to the producer. [`Puller`] exposes that
//! mechanism synchronously; [`AsyncPuller`] adds task-waker notification for
//! async executors through its pinned [`AsyncPullerWrapper`].
//!
//! # Safety model
//!
//! This is a single-producer/single-puller, single-thread coroutine. It is
//! deliberately `!Send` and `!Sync`, must remain pinned after [`Puller`] is
//! created, and is not suitable for interrupt handlers. The producer and
//! [`AsyncPullerWrapper`] must not move threads or unwind across a transfer.
//! Dropping a started coroutine panics; production users should use
//! `panic = "abort"`.
//!
//! The producer stack is measured in 16-byte [`STACK_SLOT_BYTES`] slots. Its
//! size must cover the producer's normal Rust stack frames as well as any
//! callees reached between `push` calls.
//!
//! # Embassy
//!
//! Obtain a [`AsyncPuller`] from a pinned [`Puller`], then create and pin one
//! [`AsyncPullerWrapper`] on the task's thread. Its `next` future uses the
//! executor's ordinary [`core::task::Waker`], allowing an Embassy task to use
//! `wrapper.as_mut().next().await` without an Embassy dependency:
//!
//! ```ignore
//! let puller = coroutine.as_mut().puller();
//! let mut puller = core::pin::pin!(puller);
//! let handle = puller.as_mut().async_puller(); // Send + Sync
//! let wrapper = handle.wrapper(puller.as_mut());
//! let mut wrapper = core::pin::pin!(wrapper); // stays on this task/thread
//! let value = wrapper.as_mut().next().await;
//! ```
//!
//! A full push wakes the task and returns `Pending`; its scheduled re-poll
//! observes the buffered value. The synchronous [`Puller::take_or_refill`]
//! path does not register an ordinary task waker.

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
    future::Future,
    marker::{PhantomData, PhantomPinned},
    mem::MaybeUninit,
    pin::Pin,
    ptr::NonNull,
    task::{Context, Poll, Waker},
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

/// The synchronous pull half of a [`Coroutine`].
///
/// Only one puller can be created. [`Puller::take_or_refill`] directly resumes
/// the synchronous producer when the ring is empty, until it fills the ring
/// and transfers back.
pub struct Puller<'a, T, const CAPACITY: usize, const STACK_SLOTS: usize> {
    core: NonNull<Core<T, CAPACITY, STACK_SLOTS>>,
    marker: PhantomData<&'a mut Core<T, CAPACITY, STACK_SLOTS>>,
    not_send_or_sync: PhantomData<*mut ()>,
}

/// A `Send` and `Sync` handle for the async adapter of a [`Puller`].
///
/// This handle only owns task-waker registration; it never accesses values or
/// machine contexts. The actual transfer remains confined to an
/// [`AsyncPullerWrapper`], constructed from the original pinned puller on its
/// owning thread.
pub struct AsyncPuller<'a, T, const CAPACITY: usize, const STACK_SLOTS: usize> {
    core_id: *const (),
    state: NonNull<AsyncState>,
    marker: PhantomData<&'a AsyncState>,
    value: PhantomData<fn() -> T>,
}

/// The thread-pinned async adapter that performs intra-thread transfers.
///
/// Obtain this from [`AsyncPuller::wrapper`] and pin it for the lifetime of
/// the task that polls its [`AsyncPullerWrapper::next`] futures. It is
/// deliberately `!Send` and `!Sync`.
pub struct AsyncPullerWrapper<'p, 'a, T, const CAPACITY: usize, const STACK_SLOTS: usize> {
    core: NonNull<Core<T, CAPACITY, STACK_SLOTS>>,
    state: NonNull<AsyncState>,
    puller: PhantomData<Pin<&'p mut Puller<'a, T, CAPACITY, STACK_SLOTS>>>,
    not_send_or_sync: PhantomData<*mut ()>,
    _pinned: PhantomPinned,
}

/// The proper async future returned by [`AsyncPullerWrapper::next`].
pub struct AsyncNext<'w, 'p, 'a, T, const CAPACITY: usize, const STACK_SLOTS: usize> {
    wrapper: Pin<&'w mut AsyncPullerWrapper<'p, 'a, T, CAPACITY, STACK_SLOTS>>,
    not_send_or_sync: PhantomData<*mut ()>,
    _pinned: PhantomPinned,
}

#[repr(C, align(16))]
struct StackSlot([u8; STACK_SLOT_BYTES]);

struct Core<T, const CAPACITY: usize, const STACK_SLOTS: usize> {
    ring: Mutex<Ring<T, CAPACITY>>,
    async_state: AsyncState,
    producer_context: UnsafeCell<context::Context>,
    consumer_context: UnsafeCell<context::Context>,
    entry: UnsafeCell<Option<Entry>>,
    started: Cell<bool>,
    claimed: Cell<bool>,
    async_wrapper_claimed: Cell<bool>,
    stack: [MaybeUninit<StackSlot>; STACK_SLOTS],
}

struct AsyncState {
    task_waker: Mutex<Option<Waker>>,
}

impl AsyncState {
    const fn new() -> Self {
        Self {
            task_waker: Mutex::new(None),
        }
    }

    fn register(&self, waker: &Waker) {
        *self.task_waker.lock() = Some(waker.clone());
    }

    fn take_waker(&self) -> Option<Waker> {
        self.task_waker.lock().take()
    }
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
            async_state: AsyncState::new(),
            producer_context: UnsafeCell::new(context::Context::EMPTY),
            consumer_context: UnsafeCell::new(context::Context::EMPTY),
            entry: UnsafeCell::new(None),
            started: Cell::new(false),
            claimed: Cell::new(false),
            async_wrapper_claimed: Cell::new(false),
            stack: [const { MaybeUninit::uninit() }; STACK_SLOTS],
        }
    }

    fn stack_top(&self) -> *mut u8 {
        // SAFETY: one-past-the-end is valid to form and is the aligned top of
        // the owned producer stack.
        unsafe { self.stack.as_ptr().add(STACK_SLOTS).cast::<u8>() as *mut u8 }
    }

    fn start_if_needed(&self) {
        if !self.started.replace(true) {
            // SAFETY: `puller` initializes entry exactly once before exposing
            // either the direct puller or its async adapter.
            let entry =
                unsafe { (*self.entry.get()).expect("a puller always has a producer entry") };
            // SAFETY: the producer context is initialized only for its first
            // activation, before any transfer can restore it.
            unsafe {
                context::initialize(
                    &mut *self.producer_context.get(),
                    self.stack_top(),
                    entry.call as *const () as usize,
                    entry.data as usize,
                );
            }
        }
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
    /// Return a `Send`/`Sync` handle for this puller's async adapter.
    ///
    /// Call [`AsyncPuller::wrapper`] with this same pinned puller to create
    /// the thread-pinned adapter that owns the actual stack transfer.
    pub fn async_puller(self: Pin<&mut Self>) -> AsyncPuller<'a, T, CAPACITY, STACK_SLOTS> {
        // SAFETY: this merely reads the stable core pointer. Pinning is
        // required because the handle can later create a context-switching
        // wrapper from this puller.
        let this = unsafe { self.get_unchecked_mut() };
        // SAFETY: a Puller always points at its live owning Coroutine.
        let core = unsafe { this.core.as_ref() };
        AsyncPuller {
            core_id: this.core.as_ptr().cast(),
            state: NonNull::from(&core.async_state),
            marker: PhantomData,
            value: PhantomData,
        }
    }

    /// Pull the next FIFO value, starting or refilling the producer as needed.
    pub fn take_or_refill(&mut self) -> T {
        // SAFETY: Puller is the unique public handle to this core and its
        // lifetime is tied to the pinned coroutine that owns the storage.
        let core = unsafe { self.core.as_ref() };
        if let Some(value) = core.ring.lock().pop() {
            return value;
        }

        core.start_if_needed();

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

impl<'a, T, const CAPACITY: usize, const STACK_SLOTS: usize> Copy
    for AsyncPuller<'a, T, CAPACITY, STACK_SLOTS>
{
}

impl<'a, T, const CAPACITY: usize, const STACK_SLOTS: usize> Clone
    for AsyncPuller<'a, T, CAPACITY, STACK_SLOTS>
{
    fn clone(&self) -> Self {
        *self
    }
}

// AsyncPuller only manipulates its isolated AsyncState, which is protected by
// a spin mutex. It never dereferences Core or accesses T; wrapper creation
// additionally requires the original thread-local Puller.
unsafe impl<'a, T, const CAPACITY: usize, const STACK_SLOTS: usize> Send
    for AsyncPuller<'a, T, CAPACITY, STACK_SLOTS>
{
}

unsafe impl<'a, T, const CAPACITY: usize, const STACK_SLOTS: usize> Sync
    for AsyncPuller<'a, T, CAPACITY, STACK_SLOTS>
{
}

impl<'a, T, const CAPACITY: usize, const STACK_SLOTS: usize>
    AsyncPuller<'a, T, CAPACITY, STACK_SLOTS>
{
    /// Bind this handle to its original pinned puller on the owning thread.
    ///
    /// Only one wrapper can exist at a time. The wrapper, rather than this
    /// handle, is `!Send` and `!Sync` because it can save a live return
    /// context during an intra-thread transfer.
    pub fn wrapper<'p>(
        &self,
        puller: Pin<&'p mut Puller<'a, T, CAPACITY, STACK_SLOTS>>,
    ) -> AsyncPullerWrapper<'p, 'a, T, CAPACITY, STACK_SLOTS> {
        // SAFETY: the wrapper borrows the pinned puller for 'p, preventing a
        // concurrent direct pull, and reads only its stable core pointer.
        let puller = unsafe { puller.get_unchecked_mut() };
        assert_eq!(
            puller.core.as_ptr().cast::<()>().cast_const(),
            self.core_id,
            "an async handle must be wrapped by its originating puller"
        );
        // SAFETY: the pointer belongs to the same live core checked above.
        let core = unsafe { puller.core.as_ref() };
        assert!(
            !core.async_wrapper_claimed.replace(true),
            "an async puller has exactly one active wrapper"
        );

        AsyncPullerWrapper {
            core: puller.core,
            state: self.state,
            puller: PhantomData,
            not_send_or_sync: PhantomData,
            _pinned: PhantomPinned,
        }
    }
}

impl<'p, 'a, T, const CAPACITY: usize, const STACK_SLOTS: usize>
    AsyncPullerWrapper<'p, 'a, T, CAPACITY, STACK_SLOTS>
{
    /// Create a future that awaits the next FIFO value.
    pub fn next(self: Pin<&mut Self>) -> AsyncNext<'_, 'p, 'a, T, CAPACITY, STACK_SLOTS> {
        AsyncNext {
            wrapper: self,
            not_send_or_sync: PhantomData,
            _pinned: PhantomPinned,
        }
    }

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        // SAFETY: this wrapper is created only from the unique Puller and is
        // pinned for every poll that could save its return context.
        let this = unsafe { self.get_unchecked_mut() };
        // SAFETY: the unique wrapper's borrow prevents direct concurrent use
        // of Puller, and its core outlives the wrapper.
        let core = unsafe { this.core.as_ref() };

        if let Some(value) = core.ring.lock().pop() {
            return Poll::Ready(value);
        }

        // Register before resuming the producer. A full push takes and wakes
        // this Waker before restoring the wrapper's saved continuation.
        // SAFETY: state belongs to the same live Core as `core`.
        unsafe { this.state.as_ref() }.register(cx.waker());
        core.start_if_needed();

        // A full Pusher wakes the task then restores this continuation. The
        // wrapper deliberately returns Pending instead of consuming the value
        // in this poll; the executor's next poll observes the buffered value.
        unsafe {
            context::transfer(core.consumer_context.get(), core.producer_context.get());
        }
        Poll::Pending
    }
}

impl<'p, 'a, T, const CAPACITY: usize, const STACK_SLOTS: usize> Drop
    for AsyncPullerWrapper<'p, 'a, T, CAPACITY, STACK_SLOTS>
{
    fn drop(&mut self) {
        // SAFETY: the wrapper's borrow proves that this core is still live.
        let core = unsafe { self.core.as_ref() };
        core.async_wrapper_claimed.set(false);
    }
}

impl<'w, 'p, 'a, T, const CAPACITY: usize, const STACK_SLOTS: usize> Future
    for AsyncNext<'w, 'p, 'a, T, CAPACITY, STACK_SLOTS>
{
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: AsyncNext is !Unpin and delegates its pinned wrapper to the
        // only code path that saves/restores the consumer continuation.
        let this = unsafe { self.get_unchecked_mut() };
        this.wrapper.as_mut().poll_next(cx)
    }
}

struct TransferWaker<T, const CAPACITY: usize, const STACK_SLOTS: usize> {
    core: NonNull<Core<T, CAPACITY, STACK_SLOTS>>,
}

impl<T, const CAPACITY: usize, const STACK_SLOTS: usize> TransferWaker<T, CAPACITY, STACK_SLOTS> {
    /// Notify an async task when registered, then resume the saved pull
    /// continuation through a non-scheduling symmetric transfer.
    unsafe fn wake(self) {
        // SAFETY: Pusher runs only after Puller has saved consumer_context, and
        // `switch` restores that context before this function can return.
        let core = unsafe { self.core.as_ref() };
        if core.async_wrapper_claimed.get() {
            if let Some(waker) = core.async_state.take_waker() {
                // `take_waker` releases its mutex before calling arbitrary
                // task wake logic, which may immediately schedule another
                // poll.
                waker.wake();
            }
        }
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
        task::{Context, Poll, Waker},
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

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
            assert_eq!(puller.take_or_refill(), expected);
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

        let first = puller.take_or_refill();
        let second = puller.take_or_refill();
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
            assert_eq!(puller.take_or_refill(), expected);
        }
    }

    struct CountWaker(AtomicUsize);

    impl std::task::Wake for CountWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn poll_once<F: Future>(mut future: Pin<&mut F>, waker: &Waker) -> Poll<F::Output> {
        let mut context = Context::from_waker(waker);
        future.as_mut().poll(&mut context)
    }

    fn requires_send_and_sync<T: Send + Sync>() {}

    #[test]
    fn async_wrapper_notifies_then_returns_buffered_values() {
        let mut coroutine = ManuallyDrop::new(Coroutine::new(numbers));
        // SAFETY: ManuallyDrop prevents the intentional started-coroutine drop
        // panic at the end of this test; the stack remains valid throughout.
        let pinned = unsafe { Pin::new_unchecked(&mut *coroutine) };
        let puller = pinned.puller();
        let mut puller = core::pin::pin!(puller);
        let handle = puller.as_mut().async_puller();
        requires_send_and_sync::<AsyncPuller<'static, u32, 2, 64>>();
        let wrapper = handle.wrapper(puller.as_mut());
        let mut wrapper = core::pin::pin!(wrapper);
        let count = Arc::new(CountWaker(AtomicUsize::new(0)));
        let waker = Waker::from(count.clone());

        {
            let future = wrapper.as_mut().next();
            let mut future = core::pin::pin!(future);
            assert!(matches!(poll_once(future.as_mut(), &waker), Poll::Pending));
            assert_eq!(count.0.load(Ordering::SeqCst), 1);
            assert!(matches!(poll_once(future.as_mut(), &waker), Poll::Ready(0)));
        }

        {
            let future = wrapper.as_mut().next();
            let mut future = core::pin::pin!(future);
            assert!(matches!(poll_once(future.as_mut(), &waker), Poll::Ready(1)));
        }

        {
            let future = wrapper.as_mut().next();
            let mut future = core::pin::pin!(future);
            assert!(matches!(poll_once(future.as_mut(), &waker), Poll::Pending));
            assert_eq!(count.0.load(Ordering::SeqCst), 2);
            assert!(matches!(poll_once(future.as_mut(), &waker), Poll::Ready(2)));
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
            assert_eq!(puller.take_or_refill().to_bits(), (value * scale).to_bits());
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
