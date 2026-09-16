#![no_std]
#![warn(missing_docs)]

//! Symbolically execute a deliberately small, well-behaved RV32 or RV64 program subset.
//!
//! `cirrus-ert` evaluates registers as 32 Boolean wires while keeping a parallel,
//! optional concrete value for each register. The concrete values let the
//! interpreter resolve control flow and addresses that must be known while the
//! symbolic values are emitted through [`ContextWithRvOps`].
//!
//! The interpreter reads its little-endian RV32 instruction image through
//! [`RawMemory`], caller-owned symbolic storage (one Boolean element per bit),
//! and a concrete return stack in `rstack`. The caller passes both the storage
//! value and its bit capacity. It is byte-addressed by the guest stack pointer:
//! [`ert_emit`] starts `sp` at its end, while [`ert_func`] reserves caller stack
//! slots for arguments or results beyond `a7`. `zero` and `one` are the
//! caller's symbolic Boolean constants. The hash callback implements the
//! supported hash environment call.
//!
//! Host and embedded use the same API. A desktop or server maps an ELF (or a
//! slice) and calls [`ert_func`] in-process; extra ABI words live in the
//! virtual stack, and results come back in `a0`–`a7` (then stacked words).
//! QEMU / on-device firmware is the same instruction subset, not a different
//! interpreter. WASM and LLVM ingest are sibling paths; this crate does not
//! rank them. See this workspace's `crates/ert/frontend-choice.md`.
//!
//! [`RawMemory::from_slice`] maps guest address zero to a borrowed host buffer
//! and safely bounds every access. The unsafe [`RawMemory::new`] constructor is
//! intended for bare-metal callers that deliberately address their whole mapped
//! address space, and for host tests that replay a QEMU-linked image at its
//! native base by adjusting the pointer. The caller must ensure every
//! instruction-fetch and concrete-load byte that the program reaches is readable.
//!
//! This is not a general RISC-V emulator. Programs must use instructions
//! whose compressed or normal encoding expands to the supported subset
//! (compressed forms are decoded and handled identically); branch only on
//! concrete values; use the
//! supported stack-address form for symbolic memory; provide sufficiently large
//! stacks; and follow the supported direct-call/return convention. Unsupported
//! instructions, dynamic control flow or addresses, and invalid environment
//! calls return [`ErtError::Unexpected`].
//!
//! The supported instructions are `LUI`, `AUIPC`; `ADDI`, `ADD`, `SUB`, `AND`,
//! `OR`, `XOR`, their supported immediate forms; immediate and register shifts;
//! and `MUL`, `MULH`, `MULHSU`, and `MULHU`; `LB`, `LBU`, `LH`, `LHU`, `LW`,
//! `SB`, `SH`, `SW`; `JAL`, concrete-target `JALR` calls, the conventional
//! `jalr x0, 0(ra)` return, and the six integer branches; plus the hash and exit
//! `ECALL`s. Call and return boundaries (`JAL`, `JALR`, and the conventional
//! return form) can additionally be observed and replaced through
//! [`RvHandler::call_hook`]: a handler may count calls, divert a call to a
//! host-chosen concrete target, resolve an otherwise-rejected indirect call,
//! or replace the callee entirely (`ReturnNow`) after writing result
//! registers. The default hook does nothing and preserves historical
//! behavior. The `call-hooks` feature adds an optional allocation-using
//! [`hooks::CallRegistry`] of canned per-target replacements.
//! `ECALL`s. RV64 (`ert64_emit`/`ert64_func`) additionally supports the `*W`
//! word forms (`ADDIW`, `ADDW`, `SUBW`, the `*W` shifts), `LW` sign-extension,
//! `LWU`, `LD`, and `SD`, with `LD`/`SD` and 64-bit shifts operating on the
//! full width.
//!
//! Symbolic register shifts use a five-stage barrel shifter over `rs2[4:0]`
//! (six stages over `rs2[5:0]` on RV64).
//! Symbolic multiplication uses fixed long-multiplication rounds. A concrete
//! shift amount or multiplicand selects a smaller fixed-shift or constant-product
//! path, so callers should retain concrete metadata whenever it is known.

#[cfg(feature = "call-hooks")]
extern crate alloc;

use core::{array, error::Error};

#[cfg(feature = "prepared-recording")]
use core::convert::Infallible;

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithStorage, ContextWithValue,
    HasError, StorageAddressBit,
};
pub use cirrus_ert_core::{EcallOutcome, Handler, RawMemory};
#[cfg(feature = "prepared-recording")]
use cirrus_recompile_core::{Idx, PreparedRecorder};
#[cfg(feature = "prepared-recording")]
use cirrus_volar_boolar::MuxTreeContext;
use rv_asm::{DecodeError, Reg};

#[cfg(feature = "early-exit-loops")]
mod early_exit;
mod handlers;
mod machine;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests64;

#[cfg(test)]
mod tests_hooks;

#[cfg(all(test, feature = "early-exit-loops"))]
mod early_exit_tests;

#[cfg(feature = "call-hooks")]
pub mod hooks;

#[cfg(feature = "early-exit-loops")]
pub use cirrus_ert_core::EarlyExitLoopOptions;

use machine::{Machine, RstackWord, Runtime, add_bits, read_abi_results, write_abi_args};

/// The Boolean operations required to execute the supported RISC-V subset.
pub trait ContextWithRvOps<Val>:
    cirrus_ert_core::ContextWithErtOps<Val> + ContextWithStorage<Val>
{
}

impl<Val, T: cirrus_ert_core::ContextWithErtOps<Val> + ContextWithStorage<Val>>
    ContextWithRvOps<Val> for T
{
}

/// An error while decoding or symbolically executing a program.
pub enum ErtError<E> {
    /// The caller-supplied Boolean context or hash callback returned this error.
    Emitted(E),
    /// The instruction image could not be decoded as RV32.
    Decode(DecodeError),
    /// The program is outside the supported, well-behaved instruction subset.
    Unexpected,
}

/// A [`Handler`] that reproduces the historical `ECALL` convention: concrete
/// `a0 = 0` calls the `hash` callback on the eight words following `a1`, and
/// concrete `a0 = 0xffff_ffff` exits.
pub struct DefaultHandler<C, F> {
    /// The Boolean context bit operations, and the hash callback's own
    /// concrete-type access, are both delegated to this context.
    pub context: C,
    /// The hash callback invoked for the hash `ECALL`. Its first argument is
    /// the same context passed via `context`, letting a caller's closure use
    /// inherent/concrete methods beyond the three bit-op trait methods (e.g.
    /// a `MeasuredGc`'s own gate counters) while computing a hash.
    pub hash: F,
}

impl<C: HasError, F> HasError for DefaultHandler<C, F> {
    type Error = C::Error;
}

impl<C: ContextWithValue<bool>, F> ContextWithValue<bool> for DefaultHandler<C, F> {
    type Wrapped = C::Wrapped;
}

impl<C: ContextWithBitAnd<bool>, F> ContextWithBitAnd<bool> for DefaultHandler<C, F> {
    fn bitand(&mut self, a: C::Wrapped, b: C::Wrapped) -> Result<C::Wrapped, C::Error> {
        self.context.bitand(a, b)
    }
    fn bitand_assign(&mut self, a: &mut C::Wrapped, b: C::Wrapped) -> Result<(), C::Error> {
        self.context.bitand_assign(a, b)
    }
}

impl<C: ContextWithBitOr<bool>, F> ContextWithBitOr<bool> for DefaultHandler<C, F> {
    fn bitor(&mut self, a: C::Wrapped, b: C::Wrapped) -> Result<C::Wrapped, C::Error> {
        self.context.bitor(a, b)
    }
    fn bitor_assign(&mut self, a: &mut C::Wrapped, b: C::Wrapped) -> Result<(), C::Error> {
        self.context.bitor_assign(a, b)
    }
}

impl<C: ContextWithBitXor<bool>, F> ContextWithBitXor<bool> for DefaultHandler<C, F> {
    fn bitxor(&mut self, a: C::Wrapped, b: C::Wrapped) -> Result<C::Wrapped, C::Error> {
        self.context.bitxor(a, b)
    }
    fn bitxor_assign(&mut self, a: &mut C::Wrapped, b: C::Wrapped) -> Result<(), C::Error> {
        self.context.bitxor_assign(a, b)
    }
}

impl<C, F> ContextWithStorage<bool> for DefaultHandler<C, F>
where
    C: ContextWithStorage<bool>,
{
    type Storage = C::Storage;

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<C::Wrapped>],
    ) -> Result<C::Wrapped, C::Error> {
        self.context.storage_read(storage, address)
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<C::Wrapped>],
        value: C::Wrapped,
    ) -> Result<(), C::Error> {
        self.context.storage_write(storage, address, value)
    }
}

impl<C, F, W: Clone, E: Error, const BITS: usize> Handler<bool, BITS> for DefaultHandler<C, F>
where
    C: ContextWithRvOps<bool, Wrapped = W, Error = E>,
    F: FnMut(&mut C, &[[W; BITS]]) -> Result<[u8; 32], E>,
{
    fn ecall(
        &mut self,
        regs: &mut [[W; BITS]],
        reg_consts: &mut [Option<u64>],
        offsets: &mut [Option<i64>],
        zero: &W,
        one: &W,
    ) -> Result<EcallOutcome, E> {
        match reg_consts[Reg::A0.0 as usize] {
            Some(0) => {
                // The hash payload is 32 bytes: eight 32-bit registers on
                // RV32, four 64-bit registers on RV64.
                let hash_registers = if BITS == 64 { 4 } else { 8 };
                let hash = (self.hash)(
                    &mut self.context,
                    &regs[Reg::A1.0 as usize..][..hash_registers],
                )?;
                for (index, chunk) in hash.chunks_exact(BITS / 8).enumerate() {
                    let register = Reg::A1.0 as usize + index;
                    let value = u64::from_le_bytes(array::from_fn(|i| {
                        if i < BITS / 8 { chunk[i] } else { 0 }
                    }));
                    reg_consts[register] = Some(value);
                    offsets[register] = None;
                    for bit in 0..BITS {
                        regs[register][bit] = if (value >> bit) & 1 == 0 {
                            zero.clone()
                        } else {
                            one.clone()
                        };
                    }
                }
                Ok(EcallOutcome::Continue)
            }
            // RV64 accepts either encoding of the historical selector:
            // zero-extended `0x0000_0000_ffff_ffff` (the RT macro's literal)
            // or sign-extended `-1` (`li a0, -1`).
            Some(value) if value == 0xffff_ffff || (BITS == 64 && value == u64::MAX) => {
                Ok(EcallOutcome::Exit)
            }
            _ => Ok(EcallOutcome::Unexpected),
        }
    }
}

/// A call/return boundary observed by [`RvHandler::call_hook`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallEvent {
    /// A `JAL` call: `link` receives the return address and control continues
    /// at `target`.
    Jal {
        /// Address of the call instruction.
        caller_pc: u64,
        /// The call target.
        target: u64,
        /// The link register (`x0` for a plain jump).
        link: Reg,
    },
    /// A `JALR` call whose target resolved concretely.
    Jalr {
        /// Address of the call instruction.
        caller_pc: u64,
        /// The resolved call target.
        target: u64,
        /// The base register the target was resolved from.
        base: Reg,
        /// The instruction's immediate offset.
        offset: i64,
        /// The link register.
        link: Reg,
    },
    /// The conventional `jalr x0, 0(ra)` return, which will continue at
    /// `target` (the private return stack's top).
    Return {
        /// Address of the return instruction.
        from_pc: u64,
        /// The return target.
        target: u64,
    },
    /// A `JALR` whose base register was not concretely known. The default
    /// [`CallAction::Proceed`] response fails closed with
    /// [`ErtError::Unexpected`], preserving historical behavior; a hook may
    /// resolve the target with [`CallAction::Divert`] or replace the call
    /// with [`CallAction::ReturnNow`].
    UnresolvedJalr {
        /// Address of the `JALR` instruction.
        caller_pc: u64,
        /// The unresolved base register.
        base: Reg,
        /// The instruction's immediate offset.
        offset: i64,
        /// The link register.
        link: Reg,
    },
}

/// The hook's decision at a call/return boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallAction {
    /// Execute the call or return normally.
    Proceed,
    /// Replace a call without executing the callee: the handler has already
    /// written any result registers; execution continues at the instruction
    /// following the call and the private return stack is not pushed.
    /// Meaningless on a return, where it fails closed with
    /// [`ErtError::Unexpected`].
    ReturnNow,
    /// Continue execution at a different concrete target.
    Divert(u64),
}

/// An RV32/RV64-specific [`Handler`] extension point: call/return
/// interception. The default implementation preserves historical behavior
/// exactly — no interception, and unresolved indirect calls keep failing
/// closed — so unaffected callers pay nothing.
pub trait RvHandler<Val, const BITS: usize = 32>: Handler<Val, BITS> + ContextWithStorage<Val> {
    /// Observe or replace a call/return boundary. The register-file views
    /// match [`Handler::ecall`]'s: a `ReturnNow` response is expected to have
    /// written any result registers already. Returning `Err` aborts execution
    /// with a caller-emitted error.
    fn call_hook(
        &mut self,
        event: CallEvent,
        regs: &mut [[<Self as ContextWithValue<bool>>::Wrapped; BITS]],
        reg_consts: &mut [Option<u64>],
        offsets: &mut [Option<i64>],
        zero: &<Self as ContextWithValue<bool>>::Wrapped,
        one: &<Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<CallAction, Self::Error> {
        let _ = (event, regs, reg_consts, offsets, zero, one);
        Ok(CallAction::Proceed)
    }
}

/// Tunnels any [`RvHandler`] through, delegating the environment call and
/// the call hook to the wrapped handler — the RV half of the extension
/// pattern Arm's `ArmDefaultHandler` establishes for real.
pub struct RvDefaultHandler<H> {
    /// The wrapped handler.
    pub inner: H,
}

impl<H: HasError> HasError for RvDefaultHandler<H> {
    type Error = H::Error;
}

impl<H: ContextWithValue<bool>> ContextWithValue<bool> for RvDefaultHandler<H> {
    type Wrapped = H::Wrapped;
}

impl<H: ContextWithBitAnd<bool>> ContextWithBitAnd<bool> for RvDefaultHandler<H> {
    fn bitand(&mut self, a: H::Wrapped, b: H::Wrapped) -> Result<H::Wrapped, H::Error> {
        self.inner.bitand(a, b)
    }
    fn bitand_assign(&mut self, a: &mut H::Wrapped, b: H::Wrapped) -> Result<(), H::Error> {
        self.inner.bitand_assign(a, b)
    }
}

impl<H: ContextWithBitOr<bool>> ContextWithBitOr<bool> for RvDefaultHandler<H> {
    fn bitor(&mut self, a: H::Wrapped, b: H::Wrapped) -> Result<H::Wrapped, H::Error> {
        self.inner.bitor(a, b)
    }
    fn bitor_assign(&mut self, a: &mut H::Wrapped, b: H::Wrapped) -> Result<(), H::Error> {
        self.inner.bitor_assign(a, b)
    }
}

impl<H: ContextWithBitXor<bool>> ContextWithBitXor<bool> for RvDefaultHandler<H> {
    fn bitxor(&mut self, a: H::Wrapped, b: H::Wrapped) -> Result<H::Wrapped, H::Error> {
        self.inner.bitxor(a, b)
    }
    fn bitxor_assign(&mut self, a: &mut H::Wrapped, b: H::Wrapped) -> Result<(), H::Error> {
        self.inner.bitxor_assign(a, b)
    }
}

impl<H> ContextWithStorage<bool> for RvDefaultHandler<H>
where
    H: ContextWithStorage<bool>,
{
    type Storage = H::Storage;

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<H::Wrapped>],
    ) -> Result<H::Wrapped, H::Error> {
        self.inner.storage_read(storage, address)
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<H::Wrapped>],
        value: H::Wrapped,
    ) -> Result<(), H::Error> {
        self.inner.storage_write(storage, address, value)
    }
}

impl<H: Handler<bool, BITS>, const BITS: usize> Handler<bool, BITS> for RvDefaultHandler<H> {
    fn ecall(
        &mut self,
        regs: &mut [[H::Wrapped; BITS]],
        reg_consts: &mut [Option<u64>],
        offsets: &mut [Option<i64>],
        zero: &H::Wrapped,
        one: &H::Wrapped,
    ) -> Result<EcallOutcome, H::Error> {
        self.inner.ecall(regs, reg_consts, offsets, zero, one)
    }

    fn early_exit_loop_options(&self) -> cirrus_ert_core::EarlyExitLoopOptions {
        self.inner.early_exit_loop_options()
    }
}

impl<C, F, W: Clone, E: Error, const BITS: usize> RvHandler<bool, BITS> for DefaultHandler<C, F>
where
    C: ContextWithRvOps<bool, Wrapped = W, Error = E>,
    F: FnMut(&mut C, &[[W; BITS]]) -> Result<[u8; 32], E>,
{
    // The default `call_hook` (no interception) fires; nothing to add.
}

impl<H, const BITS: usize> RvHandler<bool, BITS> for RvDefaultHandler<H>
where
    H: RvHandler<bool, BITS>,
{
    fn call_hook(
        &mut self,
        event: CallEvent,
        regs: &mut [[<Self as ContextWithValue<bool>>::Wrapped; BITS]],
        reg_consts: &mut [Option<u64>],
        offsets: &mut [Option<i64>],
        zero: &<Self as ContextWithValue<bool>>::Wrapped,
        one: &<Self as ContextWithValue<bool>>::Wrapped,
    ) -> Result<CallAction, Self::Error> {
        self.inner.call_hook(event, regs, reg_consts, offsets, zero, one)
    }
}

/// The handler shape used by [`ert_func_prepared`]/[`ert_emit_prepared`] and
/// their RV64 counterparts [`ert64_func_prepared`]/[`ert64_emit_prepared`].
/// The hash callback's register-slice width follows the executed width.
///
/// After execution, consume `handler.inner.context` with
/// [`MuxTreeContext::into_inner`] and then [`PreparedRecorder::finish`] to
/// obtain the prepared artifact. Storage remains external and is lowered to
/// the recorder's ordinary MUX/demux Boolean operations.
#[cfg(feature = "prepared-recording")]
pub type PreparedRvHandler<F> =
    RvDefaultHandler<DefaultHandler<MuxTreeContext<PreparedRecorder>, F>>;

/// Couples an ERT handler with its caller-owned storage for one invocation.
/// The machine sees this as an object-safe runtime, while every actual storage
/// access is delegated to the handler's `ContextWithStorage<bool>` impl.
struct StorageRuntime<'a, H: ContextWithStorage<bool> + ?Sized> {
    handler: &'a mut H,
    storage: &'a mut H::Storage,
    zero: H::Wrapped,
    one: H::Wrapped,
}

impl<H: ContextWithStorage<bool> + ?Sized> HasError for StorageRuntime<'_, H> {
    type Error = H::Error;
}

impl<H: ContextWithStorage<bool> + ?Sized> ContextWithValue<bool> for StorageRuntime<'_, H> {
    type Wrapped = H::Wrapped;
}

impl<H> ContextWithBitAnd<bool> for StorageRuntime<'_, H>
where
    H: ContextWithStorage<bool> + ContextWithBitAnd<bool> + ?Sized,
{
    fn bitand(&mut self, a: H::Wrapped, b: H::Wrapped) -> Result<H::Wrapped, H::Error> {
        self.handler.bitand(a, b)
    }

    fn bitand_assign(&mut self, a: &mut H::Wrapped, b: H::Wrapped) -> Result<(), H::Error> {
        self.handler.bitand_assign(a, b)
    }
}

impl<H> ContextWithBitOr<bool> for StorageRuntime<'_, H>
where
    H: ContextWithStorage<bool> + ContextWithBitOr<bool> + ?Sized,
{
    fn bitor(&mut self, a: H::Wrapped, b: H::Wrapped) -> Result<H::Wrapped, H::Error> {
        self.handler.bitor(a, b)
    }

    fn bitor_assign(&mut self, a: &mut H::Wrapped, b: H::Wrapped) -> Result<(), H::Error> {
        self.handler.bitor_assign(a, b)
    }
}

impl<H> ContextWithBitXor<bool> for StorageRuntime<'_, H>
where
    H: ContextWithStorage<bool> + ContextWithBitXor<bool> + ?Sized,
{
    fn bitxor(&mut self, a: H::Wrapped, b: H::Wrapped) -> Result<H::Wrapped, H::Error> {
        self.handler.bitxor(a, b)
    }

    fn bitxor_assign(&mut self, a: &mut H::Wrapped, b: H::Wrapped) -> Result<(), H::Error> {
        self.handler.bitxor_assign(a, b)
    }
}

impl<H, W: Clone, E: Error, const BITS: usize> Runtime<W, BITS> for StorageRuntime<'_, H>
where
    H: RvHandler<bool, BITS, Wrapped = W, Error = E> + ?Sized,
{
    fn ecall(
        &mut self,
        regs: &mut [[W; BITS]],
        reg_consts: &mut [Option<u64>],
        offsets: &mut [Option<i64>],
        zero: &W,
        one: &W,
    ) -> Result<EcallOutcome, E> {
        self.handler.ecall(regs, reg_consts, offsets, zero, one)
    }

    fn call_hook(
        &mut self,
        event: CallEvent,
        regs: &mut [[W; BITS]],
        reg_consts: &mut [Option<u64>],
        offsets: &mut [Option<i64>],
        zero: &W,
        one: &W,
    ) -> Result<CallAction, E> {
        self.handler.call_hook(event, regs, reg_consts, offsets, zero, one)
    }

    fn storage_read_bit(&mut self, bit: usize) -> Result<W, E> {
        let address = self.storage_address(bit);
        self.handler.storage_read(self.storage, &address)
    }

    fn storage_write_bit(&mut self, bit: usize, value: W) -> Result<(), E> {
        let address = self.storage_address(bit);
        self.handler.storage_write(self.storage, &address, value)
    }

    #[cfg(feature = "early-exit-loops")]
    fn early_exit_loop_options(&self) -> cirrus_ert_core::EarlyExitLoopOptions {
        self.handler.early_exit_loop_options()
    }
}

impl<H> StorageRuntime<'_, H>
where
    H: ContextWithStorage<bool> + ?Sized,
    H::Wrapped: Clone,
{
    fn storage_address(
        &self,
        value: usize,
    ) -> [StorageAddressBit<H::Wrapped>; usize::BITS as usize] {
        array::from_fn(|bit| {
            let known = (value >> bit) & 1 != 0;
            StorageAddressBit {
                wire: if known {
                    self.one.clone()
                } else {
                    self.zero.clone()
                },
                known: Some(known),
            }
        })
    }
}

/// Add two little-endian symbolic 32-bit words with an initial carry bit.
///
/// The `zero` and `one` parameters are retained for compatibility with existing
/// callers. The implementation only needs the context operations and `carry`.
pub fn simple_add<W: Clone, E: Error>(
    t: &mut (dyn cirrus_ert_core::ContextWithErtOps<bool, Wrapped = W, Error = E> + '_),
    v: &[W; 32],
    w: &[W; 32],
    carry: W,
    _zero: W,
    _one: W,
) -> Result<[W; 32], E> {
    add_bits(t, v, w, carry)
}

/// Invoke a symbolic RV32 function using the RISC-V argument and result ABI.
///
/// The first eight arguments and results use `a0` through `a7`; further values
/// are placed in or read from caller-owned symbolic storage. `storage_bits` is
/// the capacity of that storage in bits. `args` carries both the symbolic word
/// and, when known, its concrete value.
///
/// This is the RV32 spelling of the ABI; [`ert64_func`] is the RV64 one. The
/// two share the interpreter: this wrapper converts the caller's `u32`
/// concrete metadata and return stack, runs the width-generic machine, and
/// converts the results back.
pub fn ert_func<W: Clone, E: Error, const N: usize, const M: usize, H>(
    t: &mut H,
    storage: &mut H::Storage,
    storage_bits: usize,
    mem: RawMemory<'_>,
    rstack: &mut [u32],
    pc: u32,
    regs: &mut [[W; 32]; 32],
    reg_consts: &mut [Option<u32>; 32],
    zero: W,
    one: W,
    args: [([W; 32], Option<u32>); N],
) -> Result<[([W; 32], Option<u32>); M], ErtError<E>>
where
    H: RvHandler<bool, Wrapped = W, Error = E> + ?Sized,
{
    let mut consts64: [Option<u64>; 32] =
        array::from_fn(|index| reg_consts[index].map(u64::from));
    let args64: [([W; 32], Option<u64>); N] =
        args.map(|(word, constant)| (word, constant.map(u64::from)));
    let result = ert_func_impl(
        t,
        storage,
        storage_bits,
        mem,
        rstack,
        u64::from(pc),
        regs,
        &mut consts64,
        zero,
        one,
        args64,
    );
    for (slot, value) in reg_consts.iter_mut().zip(consts64) {
        *slot = value.map(|value| value as u32);
    }
    let results = result?;
    Ok(results.map(|(word, constant)| (word, constant.map(|value| value as u32))))
}

/// Invoke a symbolic RV64 function using the RISC-V argument and result ABI.
///
/// The RV64 counterpart of [`ert_func`]: registers are 64 Boolean wires,
/// concrete metadata and return-stack entries are `u64`, and ABI overflow
/// words occupy 64-bit stack slots. The hash `ECALL` passes its 32-byte
/// payload through four 64-bit registers starting at `a1`; the exit `ECALL`
/// keeps the historical concrete `a0 = 0xffff_ffff` convention.
#[allow(clippy::too_many_arguments)]
pub fn ert64_func<W: Clone, E: Error, const N: usize, const M: usize, H>(
    t: &mut H,
    storage: &mut H::Storage,
    storage_bits: usize,
    mem: RawMemory<'_>,
    rstack: &mut [u64],
    pc: u64,
    regs: &mut [[W; 64]; 32],
    reg_consts: &mut [Option<u64>; 32],
    zero: W,
    one: W,
    args: [([W; 64], Option<u64>); N],
) -> Result<[([W; 64], Option<u64>); M], ErtError<E>>
where
    H: RvHandler<bool, 64, Wrapped = W, Error = E> + ?Sized,
{
    ert_func_impl(
        t,
        storage,
        storage_bits,
        mem,
        rstack,
        pc,
        regs,
        reg_consts,
        zero,
        one,
        args,
    )
}

#[allow(clippy::too_many_arguments)]
fn ert_func_impl<W: Clone, E: Error, const N: usize, const M: usize, const BITS: usize, R: RstackWord, H>(
    t: &mut H,
    storage: &mut H::Storage,
    storage_bits: usize,
    mem: RawMemory<'_>,
    rstack: &mut [R],
    pc: u64,
    regs: &mut [[W; BITS]; 32],
    reg_consts: &mut [Option<u64>; 32],
    zero: W,
    one: W,
    args: [([W; BITS], Option<u64>); N],
) -> Result<[([W; BITS], Option<u64>); M], ErtError<E>>
where
    H: RvHandler<bool, BITS, Wrapped = W, Error = E> + ?Sized,
{
    let stack_pointer =
        abi_stack_pointer(storage_bits, N.max(M), BITS).ok_or(ErtError::Unexpected)?;
    let mut runtime = StorageRuntime {
        handler: t,
        storage,
        zero: zero.clone(),
        one: one.clone(),
    };
    write_abi_args(
        &mut runtime,
        regs,
        reg_consts,
        storage_bits,
        stack_pointer,
        args,
    )
    .map_err(ErtError::Emitted)?;
    Machine::new(
        &mut runtime,
        mem,
        rstack,
        storage_bits,
        pc,
        regs,
        reg_consts,
        zero.clone(),
        one.clone(),
        stack_pointer,
    )
    .run()?;
    read_abi_results(&mut runtime, regs, reg_consts, storage_bits, stack_pointer)
        .map_err(ErtError::Emitted)
}

/// Execute RV32 through an opt-in [`PreparedRecorder`].
///
/// This preserves [`ert_func`]'s machine ABI, concrete metadata, and result
/// layout. It is deliberately separate from `ert_func`, so normal and direct
/// execution do not instantiate prepared-recording state.
#[cfg(feature = "prepared-recording")]
#[allow(clippy::too_many_arguments)]
pub fn ert_func_prepared<F, const N: usize, const M: usize>(
    t: &mut PreparedRvHandler<F>,
    storage: &mut [Idx],
    storage_bits: usize,
    mem: RawMemory<'_>,
    rstack: &mut [u32],
    pc: u32,
    regs: &mut [[Idx; 32]; 32],
    reg_consts: &mut [Option<u32>; 32],
    zero: Idx,
    one: Idx,
    args: [([Idx; 32], Option<u32>); N],
) -> Result<[([Idx; 32], Option<u32>); M], ErtError<Infallible>>
where
    F: FnMut(&mut MuxTreeContext<PreparedRecorder>, &[[Idx; 32]]) -> Result<[u8; 32], Infallible>,
{
    ert_func(
        t,
        storage,
        storage_bits,
        mem,
        rstack,
        pc,
        regs,
        reg_consts,
        zero,
        one,
        args,
    )
}

/// Execute RV64 through an opt-in [`PreparedRecorder`].
///
/// The RV64 counterpart of [`ert_func_prepared`]; the hash callback receives
/// four 64-bit registers instead of eight 32-bit ones.
#[cfg(feature = "prepared-recording")]
#[allow(clippy::too_many_arguments)]
pub fn ert64_func_prepared<F, const N: usize, const M: usize>(
    t: &mut PreparedRvHandler<F>,
    storage: &mut [Idx],
    storage_bits: usize,
    mem: RawMemory<'_>,
    rstack: &mut [u64],
    pc: u64,
    regs: &mut [[Idx; 64]; 32],
    reg_consts: &mut [Option<u64>; 32],
    zero: Idx,
    one: Idx,
    args: [([Idx; 64], Option<u64>); N],
) -> Result<[([Idx; 64], Option<u64>); M], ErtError<Infallible>>
where
    F: FnMut(&mut MuxTreeContext<PreparedRecorder>, &[[Idx; 64]]) -> Result<[u8; 32], Infallible>,
{
    ert64_func(
        t,
        storage,
        storage_bits,
        mem,
        rstack,
        pc,
        regs,
        reg_consts,
        zero,
        one,
        args,
    )
}

/// Execute a symbolic RV32 instruction image until the supported exit `ECALL`.
///
/// The interpreter resets `x0` and initializes `sp` to the byte length of the
/// supplied storage capacity. It accepts only the subset described in the [crate
/// documentation](self); an exit is `ECALL` with concrete `a0 = 0xffff_ffff`,
/// and a hash call is `ECALL` with concrete `a0 = 0`.
///
/// This is the RV32 spelling; [`ert64_emit`] is the RV64 one.
#[allow(clippy::too_many_arguments)]
pub fn ert_emit<W: Clone, E: Error, H>(
    t: &mut H,
    storage: &mut H::Storage,
    storage_bits: usize,
    mem: RawMemory<'_>,
    rstack: &mut [u32],
    pc: u32,
    regs: &mut [[W; 32]; 32],
    reg_consts: &mut [Option<u32>; 32],
    zero: W,
    one: W,
) -> Result<(), ErtError<E>>
where
    H: RvHandler<bool, Wrapped = W, Error = E> + ?Sized,
{
    let mut consts64: [Option<u64>; 32] =
        array::from_fn(|index| reg_consts[index].map(u64::from));
    let result = ert_emit_impl(
        t,
        storage,
        storage_bits,
        mem,
        rstack,
        u64::from(pc),
        regs,
        &mut consts64,
        zero,
        one,
    );
    for (slot, value) in reg_consts.iter_mut().zip(consts64) {
        *slot = value.map(|value| value as u32);
    }
    result
}

/// Execute a symbolic RV64 instruction image until the supported exit `ECALL`.
///
/// The RV64 counterpart of [`ert_emit`]: the image decodes as RV64, registers
/// are 64 wires, and the exit/hash `ECALL` conventions match [`ert64_func`].
#[allow(clippy::too_many_arguments)]
pub fn ert64_emit<W: Clone, E: Error, H>(
    t: &mut H,
    storage: &mut H::Storage,
    storage_bits: usize,
    mem: RawMemory<'_>,
    rstack: &mut [u64],
    pc: u64,
    regs: &mut [[W; 64]; 32],
    reg_consts: &mut [Option<u64>; 32],
    zero: W,
    one: W,
) -> Result<(), ErtError<E>>
where
    H: RvHandler<bool, 64, Wrapped = W, Error = E> + ?Sized,
{
    ert_emit_impl(
        t,
        storage,
        storage_bits,
        mem,
        rstack,
        pc,
        regs,
        reg_consts,
        zero,
        one,
    )
}

#[allow(clippy::too_many_arguments)]
fn ert_emit_impl<W: Clone, E: Error, const BITS: usize, R: RstackWord, H>(
    t: &mut H,
    storage: &mut H::Storage,
    storage_bits: usize,
    mem: RawMemory<'_>,
    rstack: &mut [R],
    pc: u64,
    regs: &mut [[W; BITS]; 32],
    reg_consts: &mut [Option<u64>; 32],
    zero: W,
    one: W,
) -> Result<(), ErtError<E>>
where
    H: RvHandler<bool, BITS, Wrapped = W, Error = E> + ?Sized,
{
    if storage_bits % 8 != 0 {
        return Err(ErtError::Unexpected);
    }
    let stack_pointer = u64::try_from(storage_bits / 8).map_err(|_| ErtError::Unexpected)?;
    let mut runtime = StorageRuntime {
        handler: t,
        storage,
        zero: zero.clone(),
        one: one.clone(),
    };
    Machine::new(
        &mut runtime,
        mem,
        rstack,
        storage_bits,
        pc,
        regs,
        reg_consts,
        zero,
        one,
        stack_pointer,
    )
    .run()
}

/// Execute RV32 until `ECALL` through an opt-in [`PreparedRecorder`].
///
/// Consume `t.inner.context` afterwards and call [`PreparedRecorder::finish`]
/// with the caller's declared input/output slots.
#[cfg(feature = "prepared-recording")]
#[allow(clippy::too_many_arguments)]
pub fn ert_emit_prepared<F>(
    t: &mut PreparedRvHandler<F>,
    storage: &mut [Idx],
    storage_bits: usize,
    mem: RawMemory<'_>,
    rstack: &mut [u32],
    pc: u32,
    regs: &mut [[Idx; 32]; 32],
    reg_consts: &mut [Option<u32>; 32],
    zero: Idx,
    one: Idx,
) -> Result<(), ErtError<Infallible>>
where
    F: FnMut(&mut MuxTreeContext<PreparedRecorder>, &[[Idx; 32]]) -> Result<[u8; 32], Infallible>,
{
    ert_emit(
        t,
        storage,
        storage_bits,
        mem,
        rstack,
        pc,
        regs,
        reg_consts,
        zero,
        one,
    )
}

/// Execute RV64 until `ECALL` through an opt-in [`PreparedRecorder`].
///
/// The RV64 counterpart of [`ert_emit_prepared`].
#[cfg(feature = "prepared-recording")]
#[allow(clippy::too_many_arguments)]
pub fn ert64_emit_prepared<F>(
    t: &mut PreparedRvHandler<F>,
    storage: &mut [Idx],
    storage_bits: usize,
    mem: RawMemory<'_>,
    rstack: &mut [u64],
    pc: u64,
    regs: &mut [[Idx; 64]; 32],
    reg_consts: &mut [Option<u64>; 32],
    zero: Idx,
    one: Idx,
) -> Result<(), ErtError<Infallible>>
where
    F: FnMut(&mut MuxTreeContext<PreparedRecorder>, &[[Idx; 64]]) -> Result<[u8; 32], Infallible>,
{
    ert64_emit(
        t,
        storage,
        storage_bits,
        mem,
        rstack,
        pc,
        regs,
        reg_consts,
        zero,
        one,
    )
}

fn abi_stack_pointer(storage_bits: usize, values: usize, value_bits: usize) -> Option<u64> {
    (storage_bits % 8 == 0).then_some(())?;
    let extra_values = values.saturating_sub(machine::ABI_REGS.len());
    let stack_bytes = u64::try_from(storage_bits / 8).ok()?;
    stack_bytes.checked_sub(u64::try_from(extra_values.checked_mul(value_bits / 8)?).ok()?)
}
