#![no_std]

use core::{array, error::Error, mem::MaybeUninit};

use cirrus_core::{
    ContextWithAdd, ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithDiv,
    ContextWithMul, ContextWithSub,
};
use rv_asm::{DecodeError, Inst};
pub trait ContextWithRvOps<Val>:
    ContextWithBitAnd<Val> + ContextWithBitOr<Val> + ContextWithBitXor<Val>
{
}
impl<Val, T: ContextWithBitAnd<Val> + ContextWithBitOr<Val> + ContextWithBitXor<Val>>
    ContextWithRvOps<Val> for T
{
}
pub enum ErtError<E> {
    Emitted(E),
    Decode(DecodeError),
    Unexpected,
}
pub fn simple_add<W: Clone, E: Error>(
    t: &mut (dyn ContextWithRvOps<bool, Wrapped = W, Error = E> + '_),
    v: &[W; 32],
    w: &[W; 32],
    mut carry: W,
    zero: W,
    one: W,
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
pub fn ert_emit<W: Clone, E: Error>(
    t: &mut (dyn ContextWithRvOps<bool, Wrapped = W, Error = E> + '_),
    mem: &mut [u8],
    mut pc: u32,
    mut regs: [[W; 32]; 32],
    zero: W,
    one: W,
) -> Result<[[W; 32]; 32], ErtError<E>> {
    loop {
        pc &= !3;
        let bc = match rv_asm::Inst::decode(
            u32::from_le_bytes(array::from_fn(|i| mem[i + pc as usize])),
            rv_asm::Xlen::Rv32,
        ) {
            Ok((bc, _)) => bc,
            Err(e) => return Err(ErtError::Decode(e)),
        };
        pc = match bc {
            Inst::Addi { imm, dest, src1 } => {
                if dest.0 != 0 {
                    let w = array::from_fn(|i| {
                        if (imm.as_i32() as u32 >> i) & 1 == 1 {
                            one.clone()
                        } else {
                            zero.clone()
                        }
                    });
                    regs[dest.0 as usize] = simple_add(
                        t,
                        &regs[src1.0 as usize],
                        &w,
                        zero.clone(),
                        zero.clone(),
                        one.clone(),
                    )
                    .map_err(|e| ErtError::Emitted(e))?;
                }
                pc + 4
            }
            Inst::Add { dest, src1, src2 } => {
                if dest.0 != 0 {
                    regs[dest.0 as usize] = simple_add(
                        t,
                        &regs[src1.0 as usize],
                        &regs[src2.0 as usize],
                        zero.clone(),
                        zero.clone(),
                        one.clone(),
                    )
                    .map_err(|e| ErtError::Emitted(e))?;
                }
                pc + 4
            }
            Inst::Sub { dest, src1, src2 } => {
                if dest.0 != 0 {
                    let mut a = regs[src1.0 as usize].clone();
                    for a in &mut a[..] {
                        *a = t
                            .bitxor(a.clone(), one.clone())
                            .map_err(|e| ErtError::Emitted(e))?;
                    }
                    regs[dest.0 as usize] = simple_add(
                        t,
                        &a,
                        &regs[src2.0 as usize],
                        one.clone(),
                        zero.clone(),
                        one.clone(),
                    )
                    .map_err(|e| ErtError::Emitted(e))?;
                }
                pc + 4
            }
            Inst::And { dest, src1, src2 } => {
                if dest.0 != 0 {
                    for i in 0..32 {
                        regs[dest.0 as usize][i] = t
                            .bitand(
                                regs[src1.0 as usize][i].clone(),
                                regs[src2.0 as usize][i].clone(),
                            )
                            .map_err(|e| ErtError::Emitted(e))?;
                    }
                }
                pc + 4
            }
            Inst::Or { dest, src1, src2 } => {
                if dest.0 != 0 {
                    for i in 0..32 {
                        regs[dest.0 as usize][i] = t
                            .bitor(
                                regs[src1.0 as usize][i].clone(),
                                regs[src2.0 as usize][i].clone(),
                            )
                            .map_err(|e| ErtError::Emitted(e))?;
                    }
                }
                pc + 4
            }
            Inst::Xor { dest, src1, src2 } => {
                if dest.0 != 0 {
                    for i in 0..32 {
                        regs[dest.0 as usize][i] = t
                            .bitxor(
                                regs[src1.0 as usize][i].clone(),
                                regs[src2.0 as usize][i].clone(),
                            )
                            .map_err(|e| ErtError::Emitted(e))?;
                    }
                }
                pc + 4
            }
            Inst::Andi { imm, dest, src1 } => {
                if dest.0 != 0 {
                    for i in 0..32 {
                        if imm.as_i32() as u32 & (1 << i) == 0 {
                            regs[dest.0 as usize][i] = zero.clone()
                        } else {
                            regs[dest.0 as usize][i] = regs[src1.0 as usize][i].clone();
                        }
                    }
                }
                pc + 4
            }
            Inst::Ori { imm, dest, src1 } => {
                if dest.0 != 0 {
                    for i in 0..32 {
                        if imm.as_i32() as u32 & (1 << i) != 0 {
                            regs[dest.0 as usize][i] = one.clone()
                        } else {
                            regs[dest.0 as usize][i] = regs[src1.0 as usize][i].clone();
                        }
                    }
                }
                pc + 4
            }
            Inst::Xori { imm, dest, src1 } => {
                if dest.0 != 0 {
                    for i in 0..32 {
                        if imm.as_i32() as u32 & (1 << i) != 0 {
                            regs[dest.0 as usize][i] = t
                                .bitxor(regs[src1.0 as usize][i].clone(), one.clone())
                                .map_err(|e| ErtError::Emitted(e))?;
                        } else {
                            regs[dest.0 as usize][i] = regs[src1.0 as usize][i].clone();
                        }
                    }
                }
                pc + 4
            }
            _ => return Err(ErtError::Unexpected),
        }
    }
}
