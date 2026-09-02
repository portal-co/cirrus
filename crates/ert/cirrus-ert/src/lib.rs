#![no_std]
#![warn(missing_docs)]

//! Symbolically execute a deliberately small, well-behaved RV32 program subset.
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
//! This is not a general RISC-V emulator. Programs must use aligned,
//! non-compressed instructions; branch only on concrete values; use the
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
//! `ECALL`s.
//!
//! Symbolic register shifts use a five-stage barrel shifter over `rs2[4:0]`.
//! Symbolic multiplication uses fixed long-multiplication rounds. A concrete
//! shift amount or multiplicand selects a smaller fixed-shift or constant-product
//! path, so callers should retain concrete metadata whenever it is known.

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

#[cfg(all(test, feature = "early-exit-loops"))]
mod early_exit_tests;

#[cfg(feature = "early-exit-loops")]
pub use cirrus_ert_core::EarlyExitLoopOptions;

use machine::{Machine, Runtime, add_bits, read_abi_results, write_abi_args};

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

impl<C, F, W: Clone, E: Error> Handler<bool> for DefaultHandler<C, F>
where
    C: ContextWithRvOps<bool, Wrapped = W, Error = E>,
    F: FnMut(&mut C, &[[W; 32]]) -> Result<[u8; 32], E>,
{
    fn ecall(
        &mut self,
        regs: &mut [[W; 32]],
        reg_consts: &mut [Option<u32>],
        offsets: &mut [Option<i32>],
        zero: &W,
        one: &W,
    ) -> Result<EcallOutcome, E> {
        match reg_consts[Reg::A0.0 as usize] {
            Some(0) => {
                let hash = (self.hash)(&mut self.context, &regs[Reg::A1.0 as usize..][..8])?;
                for (index, chunk) in hash.chunks_exact(4).enumerate() {
                    let register = Reg::A1.0 as usize + index;
                    let value = u32::from_le_bytes(array::from_fn(|i| chunk[i]));
                    reg_consts[register] = Some(value);
                    offsets[register] = None;
                    for bit in 0..32 {
                        regs[register][bit] = if (value >> bit) & 1 == 0 {
                            zero.clone()
                        } else {
                            one.clone()
                        };
                    }
                }
                Ok(EcallOutcome::Continue)
            }
            Some(0xffff_ffff) => Ok(EcallOutcome::Exit),
            _ => Ok(EcallOutcome::Unexpected),
        }
    }
}

/// An RV32-specific [`Handler`] extension point, currently without
/// additional requirements beyond [`Handler`] itself. Reserved so a future
/// RV32 capability can be added here later without changing the shared
/// [`Handler`] trait.
pub trait RvHandler<Val>: Handler<Val> + ContextWithStorage<Val> {}

/// Tunnels any [`Handler`] through as an [`RvHandler`], with no added
/// behavior today — the RV32 half of the extension pattern Arm's
/// `ArmDefaultHandler` establishes for real.
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

impl<H: Handler<bool>> Handler<bool> for RvDefaultHandler<H> {
    fn ecall(
        &mut self,
        regs: &mut [[H::Wrapped; 32]],
        reg_consts: &mut [Option<u32>],
        offsets: &mut [Option<i32>],
        zero: &H::Wrapped,
        one: &H::Wrapped,
    ) -> Result<EcallOutcome, H::Error> {
        self.inner.ecall(regs, reg_consts, offsets, zero, one)
    }

    fn early_exit_loop_options(&self) -> cirrus_ert_core::EarlyExitLoopOptions {
        self.inner.early_exit_loop_options()
    }
}

impl<T, Val> RvHandler<Val> for T where T: Handler<Val> + ContextWithStorage<Val> {}

/// The RV32 handler shape used by [`ert_func_prepared`] and
/// [`ert_emit_prepared`].
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

impl<H, W: Clone, E: Error> Runtime<W> for StorageRuntime<'_, H>
where
    H: RvHandler<bool, Wrapped = W, Error = E> + ?Sized,
{
    fn ecall(
        &mut self,
        regs: &mut [[W; 32]],
        reg_consts: &mut [Option<u32>],
        offsets: &mut [Option<i32>],
        zero: &W,
        one: &W,
    ) -> Result<EcallOutcome, E> {
        self.handler.ecall(regs, reg_consts, offsets, zero, one)
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
    let stack_pointer = abi_stack_pointer(storage_bits, N.max(M)).ok_or(ErtError::Unexpected)?;
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

/// Execute a symbolic RV32 instruction image until the supported exit `ECALL`.
///
/// The interpreter resets `x0` and initializes `sp` to the byte length of the
/// supplied storage capacity. It accepts only the subset described in the [crate
/// documentation](self); an exit is `ECALL` with concrete `a0 = 0xffff_ffff`,
/// and a hash call is `ECALL` with concrete `a0 = 0`.
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
    if storage_bits % 8 != 0 {
        return Err(ErtError::Unexpected);
    }
    let stack_pointer = u32::try_from(storage_bits / 8).map_err(|_| ErtError::Unexpected)?;
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

fn abi_stack_pointer(storage_bits: usize, values: usize) -> Option<u32> {
    (storage_bits % 8 == 0).then_some(())?;
    let extra_values = values.saturating_sub(machine::ABI_REGS.len());
    let stack_bytes = u32::try_from(storage_bits / 8).ok()?;
    stack_bytes.checked_sub(u32::try_from(extra_values.checked_mul(4)?).ok()?)
}
