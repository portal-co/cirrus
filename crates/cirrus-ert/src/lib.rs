#![no_std]
#![warn(missing_docs)]

//! Symbolically execute a deliberately small, well-behaved RV32 program subset.
//!
//! `cirrus-ert` evaluates registers as 32 Boolean wires while keeping a parallel,
//! optional concrete value for each register. The concrete values let the
//! interpreter resolve control flow and addresses that must be known while the
//! symbolic values are emitted through [`ContextWithRvOps`].
//!
//! The interpreter accepts a little-endian RV32 instruction image in `mem`, a
//! symbolic stack in `vstack` (one Boolean wire per bit), and a concrete return
//! stack in `rstack`. `zero` and `one` are the caller's symbolic Boolean
//! constants. The hash callback implements the supported hash environment call.
//!
//! This is not a general RISC-V emulator. Programs must use aligned,
//! non-compressed instructions; branch only on concrete values; use the
//! supported stack-address form for symbolic memory; provide sufficiently large
//! stacks; and follow the supported direct-call/return convention. Unsupported
//! instructions, dynamic control flow or addresses, and invalid environment
//! calls return [`ErtError::Unexpected`].
//!
//! The supported instructions are `LUI`, `AUIPC`; `ADDI`, `ADD`, `SUB`, `AND`,
//! `OR`, `XOR`, their supported immediate forms, and immediate shifts; `LB`,
//! `LBU`, `LH`, `LHU`, `LW`, `SB`, `SH`, `SW`; `JAL`, the interpreter's return
//! form of `JALR`, and the six integer branches; plus the hash and exit `ECALL`s.

use core::{error::Error, mem::MaybeUninit};

use cirrus_core::{ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor};
use rv_asm::DecodeError;

mod handlers;
mod machine;

#[cfg(test)]
mod tests;

use machine::{Machine, read_abi_results, write_abi_args};

/// The Boolean operations required to execute the supported RISC-V subset.
pub trait ContextWithRvOps<Val>:
    ContextWithBitAnd<Val> + ContextWithBitOr<Val> + ContextWithBitXor<Val>
{
}

impl<Val, T: ContextWithBitAnd<Val> + ContextWithBitOr<Val> + ContextWithBitXor<Val>>
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

/// Add two little-endian symbolic 32-bit words with an initial carry bit.
///
/// The `zero` and `one` parameters are retained for compatibility with existing
/// callers. The implementation only needs the context operations and `carry`.
pub fn simple_add<W: Clone, E: Error>(
    t: &mut (dyn ContextWithRvOps<bool, Wrapped = W, Error = E> + '_),
    v: &[W; 32],
    w: &[W; 32],
    mut carry: W,
    _zero: W,
    _one: W,
) -> Result<[W; 32], E> {
    let mut x: [MaybeUninit<W>; 32] = [const { MaybeUninit::uninit() }; 32];
    for i in 0..32 {
        let s = t.bitxor(v[i].clone(), w[i].clone())?;
        x[i] = MaybeUninit::new(t.bitxor(s, carry.clone())?);
        let v = [v[i].clone(), w[i].clone(), carry.clone()];
        let mut w: [MaybeUninit<W>; 3] = [const { MaybeUninit::uninit() }; 3];
        for i in 0..3 {
            w[i] = MaybeUninit::new(t.bitand(v[(i + 2) % 3].clone(), v[(i + 1) % 3].clone())?);
        }
        let [a, b, c] = w.map(|a| unsafe { a.assume_init() });
        let b = t.bitor(c, b)?;
        carry = t.bitor(a, b)?;
    }
    Ok(x.map(|a| unsafe { a.assume_init() }))
}

/// Invoke a symbolic RV32 function using the RISC-V argument and result ABI.
///
/// The first eight arguments and results use `a0` through `a7`; further values
/// are placed in or read from the symbolic stack. `args` carries both the
/// symbolic word and, when known, its concrete value.
pub fn ert_func<W: Clone, E: Error, const N: usize, const M: usize>(
    t: &mut (dyn ContextWithRvOps<bool, Wrapped = W, Error = E> + '_),
    hash: &mut (dyn FnMut(&[[W; 32]]) -> Result<[u8; 32], E> + '_),
    mem: &mut [u8],
    rstack: &mut [u32],
    vstack: &mut [W],
    pc: u32,
    regs: &mut [[W; 32]; 32],
    reg_consts: &mut [Option<u32>; 32],
    zero: W,
    one: W,
    args: [([W; 32], Option<u32>); N],
) -> Result<[([W; 32], Option<u32>); M], ErtError<E>> {
    write_abi_args(regs, reg_consts, vstack, args);
    ert_emit(
        t,
        hash,
        mem,
        rstack,
        vstack,
        pc,
        regs,
        reg_consts,
        zero.clone(),
        one.clone(),
    )?;
    Ok(read_abi_results(regs, reg_consts, vstack))
}

/// Execute a symbolic RV32 instruction image until the supported exit `ECALL`.
///
/// The interpreter resets `x0` and derives `sp` at each instruction. It accepts
/// only the subset described in the [crate documentation](self); an exit is
/// `ECALL` with concrete `a0 = 0xffff_ffff`, and a hash call is `ECALL` with
/// concrete `a0 = 0`.
pub fn ert_emit<W: Clone, E: Error>(
    t: &mut (dyn ContextWithRvOps<bool, Wrapped = W, Error = E> + '_),
    hash: &mut (dyn FnMut(&[[W; 32]]) -> Result<[u8; 32], E> + '_),
    mem: &mut [u8],
    rstack: &mut [u32],
    vstack: &mut [W],
    pc: u32,
    regs: &mut [[W; 32]; 32],
    reg_consts: &mut [Option<u32>; 32],
    zero: W,
    one: W,
) -> Result<(), ErtError<E>> {
    Machine::new(
        t, hash, mem, rstack, vstack, pc, regs, reg_consts, zero, one,
    )
    .run()
}
