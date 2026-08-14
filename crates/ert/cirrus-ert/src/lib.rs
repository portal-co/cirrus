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

use cirrus_core::{ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithValue, HasError};
pub use cirrus_ert_core::RawMemory;
use rv_asm::{DecodeError, Reg};

mod handlers;
mod machine;

#[cfg(test)]
mod tests;

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

/// A handler for the hash and exit `ECALL`s, and any others a caller adds.
///
/// `Machine` reaches [`Handler::ecall`] for every `ECALL` it decodes. The
/// caller-balanced-stack requirement for a successful exit is enforced by the
/// interpreter itself, not by the handler.
pub trait Handler<Val>: ContextWithRvOps<Val> {
    /// Handle an `ECALL`. `regs` and `reg_consts` are the full register file
    /// at the call; `zero` and `one` are the caller's symbolic Boolean
    /// constants, useful for turning a concrete result into a symbolic word.
    fn ecall(
        &mut self,
        regs: &mut [[Self::Wrapped; 32]; 32],
        reg_consts: &mut [Option<u32>; 32],
        zero: &Self::Wrapped,
        one: &Self::Wrapped,
    ) -> Result<EcallOutcome, ErtError<Self::Error>>;
}

/// The effect of a handled `ECALL` on control flow.
pub enum EcallOutcome {
    /// Continue execution at the next instruction.
    Continue,
    /// Exit the program, once the interpreter confirms the stack is balanced.
    Exit,
}

/// A [`Handler`] that reproduces the historical `ECALL` convention: concrete
/// `a0 = 0` calls the `hash` callback on the eight words following `a1`, and
/// concrete `a0 = 0xffff_ffff` exits.
pub struct DefaultHandler<'a, W, E> {
    /// The Boolean context bit operations are delegated to.
    pub context: &'a mut (dyn ContextWithRvOps<bool, Wrapped = W, Error = E> + 'a),
    /// The hash callback invoked for the hash `ECALL`.
    pub hash: &'a mut (dyn FnMut(&[[W; 32]]) -> Result<[u8; 32], E> + 'a),
}

impl<W, E: Error> HasError for DefaultHandler<'_, W, E> {
    type Error = E;
}

impl<W, E: Error> ContextWithValue<bool> for DefaultHandler<'_, W, E> {
    type Wrapped = W;
}

impl<W, E: Error> ContextWithBitAnd<bool> for DefaultHandler<'_, W, E> {
    fn bitand(&mut self, a: W, b: W) -> Result<W, E> {
        self.context.bitand(a, b)
    }
    fn bitand_assign(&mut self, a: &mut W, b: W) -> Result<(), E> {
        self.context.bitand_assign(a, b)
    }
}

impl<W, E: Error> ContextWithBitOr<bool> for DefaultHandler<'_, W, E> {
    fn bitor(&mut self, a: W, b: W) -> Result<W, E> {
        self.context.bitor(a, b)
    }
    fn bitor_assign(&mut self, a: &mut W, b: W) -> Result<(), E> {
        self.context.bitor_assign(a, b)
    }
}

impl<W, E: Error> ContextWithBitXor<bool> for DefaultHandler<'_, W, E> {
    fn bitxor(&mut self, a: W, b: W) -> Result<W, E> {
        self.context.bitxor(a, b)
    }
    fn bitxor_assign(&mut self, a: &mut W, b: W) -> Result<(), E> {
        self.context.bitxor_assign(a, b)
    }
}

impl<W: Clone, E: Error> Handler<bool> for DefaultHandler<'_, W, E> {
    fn ecall(
        &mut self,
        regs: &mut [[W; 32]; 32],
        reg_consts: &mut [Option<u32>; 32],
        zero: &W,
        one: &W,
    ) -> Result<EcallOutcome, ErtError<E>> {
        match reg_consts[Reg::A0.0 as usize] {
            Some(0) => {
                let hash = (self.hash)(&regs[Reg::A1.0 as usize..][..(256 / 32)])
                    .map_err(ErtError::Emitted)?;
                for ((register, constant), chunk) in regs[Reg::A1.0 as usize..][..(256 / 32)]
                    .iter_mut()
                    .zip(reg_consts[Reg::A1.0 as usize..][..(256 / 32)].iter_mut())
                    .zip(hash.chunks_exact(4))
                {
                    let value = u32::from_le_bytes(array::from_fn(|i| chunk[i]));
                    *constant = Some(value);
                    for bit in 0..32 {
                        register[bit] = if (value >> bit) & 1 == 0 {
                            zero.clone()
                        } else {
                            one.clone()
                        };
                    }
                }
                Ok(EcallOutcome::Continue)
            }
            Some(0xffff_ffff) => Ok(EcallOutcome::Exit),
            _ => Err(ErtError::Unexpected),
        }
    }
}

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
    t: &mut (dyn Handler<bool, Wrapped = W, Error = E> + '_),
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

/// Execute a symbolic RV32 instruction image until the supported exit `ECALL`.
///
/// The interpreter resets `x0` and initializes `sp` to the byte length of
/// `vstack`. It accepts only the subset described in the [crate
/// documentation](self); an exit is `ECALL` with concrete `a0 = 0xffff_ffff`,
/// and a hash call is `ECALL` with concrete `a0 = 0`.
pub fn ert_emit<W: Clone, E: Error>(
    t: &mut (dyn Handler<bool, Wrapped = W, Error = E> + '_),
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

fn abi_stack_pointer<W>(vstack: &[W], values: usize) -> Option<u32> {
    let extra_values = values.saturating_sub(machine::ABI_REGS.len());
    let stack_bytes = u32::try_from(vstack.len() / 8).ok()?;
    stack_bytes.checked_sub(u32::try_from(extra_values.checked_mul(4)?).ok()?)
}
