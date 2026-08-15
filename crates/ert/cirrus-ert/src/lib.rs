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
//! [`RawMemory`], a symbolic stack in `vstack` (one Boolean wire per bit), and a
//! concrete return stack in `rstack`. `vstack` is byte-addressed by the guest
//! stack pointer: [`ert_emit`] starts `sp` at its end, while [`ert_func`]
//! reserves caller stack slots for arguments or results beyond `a7`. `zero` and
//! `one` are the caller's symbolic Boolean constants. The hash callback
//! implements the supported hash environment call.
//!
//! [`RawMemory::from_slice`] maps guest address zero to a borrowed host buffer
//! and safely bounds every access. The unsafe [`RawMemory::new`] constructor is
//! intended for bare-metal callers that deliberately address their whole mapped
//! address space; its caller must ensure every instruction-fetch and
//! concrete-load byte that the program reaches is readable.
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

use cirrus_core::{ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithValue, HasError};
pub use cirrus_ert_core::{EcallOutcome, Handler, RawMemory};
#[cfg(feature = "prepared-recording")]
use cirrus_recompile_core::{Idx, PreparedRecorder};
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

use machine::{Machine, add_bits, read_abi_results, write_abi_args};

/// The Boolean operations required to execute the supported RISC-V subset.
pub trait ContextWithRvOps<Val>: cirrus_ert_core::ContextWithErtOps<Val> {}

impl<Val, T: cirrus_ert_core::ContextWithErtOps<Val>> ContextWithRvOps<Val> for T {}

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
pub trait RvHandler<Val>: Handler<Val> {}

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

impl<H: Handler<bool>> RvHandler<bool> for RvDefaultHandler<H> {}

/// The RV32 handler shape used by [`ert_func_prepared`] and
/// [`ert_emit_prepared`].
///
/// After execution, consume `handler.inner.context` with
/// [`PreparedRecorder::finish`] to obtain the prepared artifact.  The hash
/// closure retains the ordinary ERT backend tunnel and receives the concrete
/// recorder directly.
#[cfg(feature = "prepared-recording")]
pub type PreparedRvHandler<F> = RvDefaultHandler<DefaultHandler<PreparedRecorder, F>>;

/// Add two little-endian symbolic 32-bit words with an initial carry bit.
///
/// The `zero` and `one` parameters are retained for compatibility with existing
/// callers. The implementation only needs the context operations and `carry`.
pub fn simple_add<W: Clone, E: Error>(
    t: &mut (dyn ContextWithRvOps<bool, Wrapped = W, Error = E> + '_),
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
/// are placed in or read from the symbolic stack. `args` carries both the
/// symbolic word and, when known, its concrete value.
pub fn ert_func<W: Clone, E: Error, const N: usize, const M: usize>(
    t: &mut (dyn RvHandler<bool, Wrapped = W, Error = E> + '_),
    mem: RawMemory<'_>,
    rstack: &mut [u32],
    vstack: &mut [W],
    pc: u32,
    regs: &mut [[W; 32]; 32],
    reg_consts: &mut [Option<u32>; 32],
    zero: W,
    one: W,
    args: [([W; 32], Option<u32>); N],
) -> Result<[([W; 32], Option<u32>); M], ErtError<E>> {
    let stack_pointer = abi_stack_pointer(vstack, N.max(M)).ok_or(ErtError::Unexpected)?;
    write_abi_args(regs, reg_consts, vstack, stack_pointer, args);
    Machine::new(
        t,
        mem,
        rstack,
        vstack,
        pc,
        regs,
        reg_consts,
        zero.clone(),
        one.clone(),
        stack_pointer,
    )
    .run()?;
    Ok(read_abi_results(regs, reg_consts, vstack, stack_pointer))
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
    mem: RawMemory<'_>,
    rstack: &mut [u32],
    vstack: &mut [Idx],
    pc: u32,
    regs: &mut [[Idx; 32]; 32],
    reg_consts: &mut [Option<u32>; 32],
    zero: Idx,
    one: Idx,
    args: [([Idx; 32], Option<u32>); N],
) -> Result<[([Idx; 32], Option<u32>); M], ErtError<Infallible>>
where
    F: FnMut(&mut PreparedRecorder, &[[Idx; 32]]) -> Result<[u8; 32], Infallible>,
{
    ert_func(
        t, mem, rstack, vstack, pc, regs, reg_consts, zero, one, args,
    )
}

/// Execute a symbolic RV32 instruction image until the supported exit `ECALL`.
///
/// The interpreter resets `x0` and initializes `sp` to the byte length of
/// `vstack`. It accepts only the subset described in the [crate
/// documentation](self); an exit is `ECALL` with concrete `a0 = 0xffff_ffff`,
/// and a hash call is `ECALL` with concrete `a0 = 0`.
pub fn ert_emit<W: Clone, E: Error>(
    t: &mut (dyn RvHandler<bool, Wrapped = W, Error = E> + '_),
    mem: RawMemory<'_>,
    rstack: &mut [u32],
    vstack: &mut [W],
    pc: u32,
    regs: &mut [[W; 32]; 32],
    reg_consts: &mut [Option<u32>; 32],
    zero: W,
    one: W,
) -> Result<(), ErtError<E>> {
    let stack_pointer = u32::try_from(vstack.len() / 8).map_err(|_| ErtError::Unexpected)?;
    Machine::new(
        t,
        mem,
        rstack,
        vstack,
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
    mem: RawMemory<'_>,
    rstack: &mut [u32],
    vstack: &mut [Idx],
    pc: u32,
    regs: &mut [[Idx; 32]; 32],
    reg_consts: &mut [Option<u32>; 32],
    zero: Idx,
    one: Idx,
) -> Result<(), ErtError<Infallible>>
where
    F: FnMut(&mut PreparedRecorder, &[[Idx; 32]]) -> Result<[u8; 32], Infallible>,
{
    ert_emit(t, mem, rstack, vstack, pc, regs, reg_consts, zero, one)
}

fn abi_stack_pointer<W>(vstack: &[W], values: usize) -> Option<u32> {
    let extra_values = values.saturating_sub(machine::ABI_REGS.len());
    let stack_bytes = u32::try_from(vstack.len() / 8).ok()?;
    stack_bytes.checked_sub(u32::try_from(extra_values.checked_mul(4)?).ok()?)
}
