#![no_std]

use core::{array, error::Error, mem::MaybeUninit};

use cirrus_core::{
    ContextWithAdd, ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithDiv,
    ContextWithMul, ContextWithSub,
};
use rv_asm::{DecodeError, Imm, Inst, Reg};
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
    hash: &mut (dyn FnMut(&[[W; 32]]) -> Result<[u8; 32], E> + '_),
    mem: &mut [u8],
    rstack: &mut [u32],
    vstack: &mut [W],
    mut pc: u32,
    mut regs: [([W; 32]); 32],
    mut reg_consts: &mut [Option<u32>; 32],
    zero: W,
    one: W,
) -> Result<(), ErtError<E>> {
    let mut sp: u32 = 0;
    let mut rsp: u32 = 0;
    let mut offs: [Option<i32>; 32] = [const { None }; 32];
    loop {
        pc &= !3;
        let bc = match rv_asm::Inst::decode(
            u32::from_le_bytes(array::from_fn(|i| mem[i + pc as usize])),
            rv_asm::Xlen::Rv32,
        ) {
            Ok((bc, _)) => bc,
            Err(e) => return Err(ErtError::Decode(e)),
        };
        for r in regs[0].iter_mut() {
            *r = zero.clone();
        }
        reg_consts[0] = Some(0);
        offs[0] = None;
        reg_consts[Reg::SP.0 as usize] = None;
        offs[Reg::SP.0 as usize] = Some(0);
        for (i, r) in regs[Reg::SP.0 as usize].iter_mut().enumerate() {
            *r = if sp >> i == 0 {
                zero.clone()
            } else {
                one.clone()
            }
        }
        pc = match bc {
            //consts
            Inst::Lui { uimm, dest } => {
                offs[dest.0 as usize] = None;
                let v = uimm.as_u32();
                reg_consts[dest.0 as usize] = Some(v);
                for i in 0..32 {
                    regs[dest.0 as usize][i] = if v >> i == 0 {
                        zero.clone()
                    } else {
                        one.clone()
                    };
                }
                pc + 4
            }
            Inst::Auipc { uimm, dest } => {
                offs[dest.0 as usize] = None;
                let v = uimm.as_u32().wrapping_add(pc);
                reg_consts[dest.0 as usize] = Some(v);
                for i in 0..32 {
                    regs[dest.0 as usize][i] = if v >> i == 0 {
                        zero.clone()
                    } else {
                        one.clone()
                    };
                }
                pc + 4
            }
            //arith
            Inst::Addi { imm, dest, src1 } => {
                if dest == Reg::SP {
                    sp = sp.wrapping_add_signed(imm.as_i32());
                    for o in offs.iter_mut().flatten() {
                        *o = o.wrapping_add(imm.as_i32());
                    }
                }
                if src1 != dest {
                    offs[dest.0 as usize] = None;
                }
                if src1 == Reg::SP {
                    offs[dest.0 as usize] = Some(imm.as_i32())
                } else if let Some(a) = offs[src1.0 as usize] {
                    offs[dest.0 as usize] = Some(a.wrapping_add(imm.as_i32()))
                }
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
                reg_consts[dest.0 as usize] = match (reg_consts[src1.0 as usize]) {
                    (Some(a)) => Some(a.wrapping_add_signed(imm.as_i32())),
                    _ => None,
                };

                pc + 4
            }
            Inst::Add { dest, src1, src2 } => {
                if src1 != dest && src2 != dest {
                    offs[dest.0 as usize] = None;
                }
                if let Some(k) = reg_consts[src2.0 as usize] {
                    if src1 == Reg::SP {
                        offs[dest.0 as usize] = Some(k as i32)
                    } else if let Some(a) = offs[src1.0 as usize] {
                        offs[dest.0 as usize] = Some(a.wrapping_add_unsigned(k))
                    }
                }
                if let Some(k) = reg_consts[src1.0 as usize] {
                    if src2 == Reg::SP {
                        offs[dest.0 as usize] = Some(k as i32)
                    } else if let Some(a) = offs[src2.0 as usize] {
                        offs[dest.0 as usize] = Some(a.wrapping_add_unsigned(k))
                    }
                }

                regs[dest.0 as usize] = simple_add(
                    t,
                    &regs[src1.0 as usize],
                    &regs[src2.0 as usize],
                    zero.clone(),
                    zero.clone(),
                    one.clone(),
                )
                .map_err(|e| ErtError::Emitted(e))?;
                reg_consts[dest.0 as usize] =
                    match (reg_consts[src1.0 as usize], reg_consts[src2.0 as usize]) {
                        (Some(a), Some(b)) => Some(a.wrapping_add(b)),
                        _ => None,
                    };

                pc + 4
            }
            Inst::Sub { dest, src1, src2 } => {
                if src1 != dest && src2 != dest {
                    offs[dest.0 as usize] = None;
                }
                if let Some(k) = reg_consts[src2.0 as usize] {
                    if src1 == Reg::SP {
                        offs[dest.0 as usize] = Some(-(k as i32))
                    } else if let Some(a) = offs[src1.0 as usize] {
                        offs[dest.0 as usize] = Some(a.wrapping_sub_unsigned(k))
                    }
                }
                if let Some(k) = reg_consts[src1.0 as usize] {
                    if src2 == Reg::SP {
                        offs[dest.0 as usize] = Some(-(k as i32))
                    } else if let Some(a) = offs[src2.0 as usize] {
                        offs[dest.0 as usize] = Some(a.wrapping_sub_unsigned(k))
                    }
                }
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
                reg_consts[dest.0 as usize] =
                    match (reg_consts[src1.0 as usize], reg_consts[src2.0 as usize]) {
                        (Some(a), Some(b)) => Some(a.wrapping_sub(b)),
                        _ => None,
                    };

                pc + 4
            }
            Inst::And { dest, src1, src2 } => {
                offs[dest.0 as usize] = None;
                reg_consts[dest.0 as usize] =
                    match (reg_consts[src1.0 as usize], reg_consts[src2.0 as usize]) {
                        (Some(a), Some(b)) => Some(a & b),
                        _ => None,
                    };

                for i in 0..32 {
                    regs[dest.0 as usize][i] = t
                        .bitand(
                            regs[src1.0 as usize][i].clone(),
                            regs[src2.0 as usize][i].clone(),
                        )
                        .map_err(|e| ErtError::Emitted(e))?;
                }
                pc + 4
            }
            Inst::Or { dest, src1, src2 } => {
                offs[dest.0 as usize] = None;
                reg_consts[dest.0 as usize] =
                    match (reg_consts[src1.0 as usize], reg_consts[src2.0 as usize]) {
                        (Some(a), Some(b)) => Some(a | b),
                        _ => None,
                    };
                for i in 0..32 {
                    regs[dest.0 as usize][i] = t
                        .bitor(
                            regs[src1.0 as usize][i].clone(),
                            regs[src2.0 as usize][i].clone(),
                        )
                        .map_err(|e| ErtError::Emitted(e))?;
                }
                pc + 4
            }
            Inst::Xor { dest, src1, src2 } => {
                offs[dest.0 as usize] = None;
                reg_consts[dest.0 as usize] =
                    match (reg_consts[src1.0 as usize], reg_consts[src2.0 as usize]) {
                        (Some(a), Some(b)) => Some(a ^ b),
                        _ => None,
                    };
                for i in 0..32 {
                    regs[dest.0 as usize][i] = t
                        .bitxor(
                            regs[src1.0 as usize][i].clone(),
                            regs[src2.0 as usize][i].clone(),
                        )
                        .map_err(|e| ErtError::Emitted(e))?;
                }
                pc + 4
            }
            Inst::Andi { imm, dest, src1 } => {
                offs[dest.0 as usize] = None;
                reg_consts[dest.0 as usize] = match (reg_consts[src1.0 as usize]) {
                    (Some(a)) => Some(a & (imm.as_i32() as u32)),
                    _ => None,
                };
                for i in 0..32 {
                    if imm.as_i32() as u32 & (1 << i) == 0 {
                        regs[dest.0 as usize][i] = zero.clone()
                    } else {
                        regs[dest.0 as usize][i] = regs[src1.0 as usize][i].clone();
                    }
                }
                pc + 4
            }
            Inst::Ori { imm, dest, src1 } => {
                offs[dest.0 as usize] = None;
                reg_consts[dest.0 as usize] = match (reg_consts[src1.0 as usize]) {
                    (Some(a)) => Some(a | (imm.as_i32() as u32)),
                    _ => None,
                };
                for i in 0..32 {
                    if imm.as_i32() as u32 & (1 << i) != 0 {
                        regs[dest.0 as usize][i] = one.clone()
                    } else {
                        regs[dest.0 as usize][i] = regs[src1.0 as usize][i].clone();
                    }
                }
                pc + 4
            }
            Inst::Xori { imm, dest, src1 } => {
                offs[dest.0 as usize] = None;
                reg_consts[dest.0 as usize] = match (reg_consts[src1.0 as usize]) {
                    (Some(a)) => Some(a ^ (imm.as_i32() as u32)),
                    _ => None,
                };
                for i in 0..32 {
                    if imm.as_i32() as u32 & (1 << i) != 0 {
                        regs[dest.0 as usize][i] = t
                            .bitxor(regs[src1.0 as usize][i].clone(), one.clone())
                            .map_err(|e| ErtError::Emitted(e))?;
                    } else {
                        regs[dest.0 as usize][i] = regs[src1.0 as usize][i].clone();
                    }
                }
                pc + 4
            }
            Inst::Slli { imm, dest, src1 } => {
                offs[dest.0 as usize] = None;
                reg_consts[dest.0 as usize] = match (reg_consts[src1.0 as usize]) {
                    (Some(a)) => Some(a << (imm.as_i32() as u32 & 31)),
                    _ => None,
                };
                for a in regs[dest.0 as usize].iter_mut() {
                    *a = zero.clone()
                }
                for i in 0..32 {
                    let l = imm.as_u32() & 31;
                    if l + i >= 32 {
                        continue;
                    };
                    regs[dest.0 as usize][(i + l) as usize] =
                        regs[src1.0 as usize][i as usize].clone();
                }
                pc + 4
            }
            Inst::Srli { imm, dest, src1 } => {
                offs[dest.0 as usize] = None;
                reg_consts[dest.0 as usize] = match (reg_consts[src1.0 as usize]) {
                    (Some(a)) => Some(a >> (imm.as_i32() as u32 & 31)),
                    _ => None,
                };
                for a in regs[dest.0 as usize].iter_mut() {
                    *a = zero.clone()
                }
                for i in 0..32 {
                    let l = imm.as_u32() & 31;
                    if l > i {
                        continue;
                    };
                    regs[dest.0 as usize][(i - l) as usize] =
                        regs[src1.0 as usize][i as usize].clone();
                }
                pc + 4
            }
            Inst::Srai { imm, dest, src1 } => {
                offs[dest.0 as usize] = None;
                reg_consts[dest.0 as usize] = match (reg_consts[src1.0 as usize]) {
                    (Some(a)) => Some((a as i32 >> (imm.as_i32() as u32 & 31)) as u32),
                    _ => None,
                };
                let sign = regs[dest.0 as usize][31].clone();
                for a in regs[dest.0 as usize].iter_mut() {
                    *a = sign.clone();
                }
                for i in 0..32 {
                    let l = imm.as_u32() & 31;
                    if l > i {
                        continue;
                    };
                    regs[dest.0 as usize][(i - l) as usize] =
                        regs[src1.0 as usize][i as usize].clone();
                }
                pc + 4
            }

            // memory
            Inst::Lb { offset, dest, base } => 'a: {
                let offset = match base {
                    x if x == Reg::SP => offset,
                    x if offs[x.0 as usize].is_some() => {
                        Imm::new_i32(offs[base.0 as usize].unwrap().wrapping_add(offset.as_i32()))
                    }
                    x if reg_consts[x.0 as usize].is_some() => {
                        offs[dest.0 as usize] = None;
                        let v = i8::from_le_bytes(array::from_fn(|i| {
                            mem[i + reg_consts[x.0 as usize].unwrap() as usize]
                        })) as i32 as u32;
                        reg_consts[dest.0 as usize] = Some(v);
                        for i in 0..32 {
                            regs[dest.0 as usize][i] = if v >> i == 0 {
                                zero.clone()
                            } else {
                                one.clone()
                            };
                        }
                        break 'a pc + 4;
                    }
                    _ => return Err(ErtError::Unexpected),
                };
                offs[dest.0 as usize] = None;
                reg_consts[dest.0 as usize] = None;
                for i in 0..32 {
                    let j = i.min(7);
                    regs[dest.0 as usize][i] = vstack[offset.as_u32() as usize * 8 + j].clone();
                }
                pc + 4
            }
            Inst::Lh { offset, dest, base } => 'a: {
                let offset = match base {
                    x if x == Reg::SP => offset,
                    x if offs[x.0 as usize].is_some() => {
                        Imm::new_i32(offs[base.0 as usize].unwrap().wrapping_add(offset.as_i32()))
                    }
                    x if reg_consts[x.0 as usize].is_some() => {
                        offs[dest.0 as usize] = None;
                        let v = i16::from_le_bytes(array::from_fn(|i| {
                            mem[i + reg_consts[x.0 as usize].unwrap() as usize]
                        })) as i32 as u32;
                        reg_consts[dest.0 as usize] = Some(v);
                        for i in 0..32 {
                            regs[dest.0 as usize][i] = if v >> i == 0 {
                                zero.clone()
                            } else {
                                one.clone()
                            };
                        }
                        break 'a pc + 4;
                    }
                    _ => return Err(ErtError::Unexpected),
                };
                offs[dest.0 as usize] = None;
                reg_consts[dest.0 as usize] = None;
                for i in 0..32 {
                    let j = i.min(15);
                    regs[dest.0 as usize][i] = vstack[offset.as_u32() as usize * 8 + j].clone();
                }
                pc + 4
            }
            Inst::Lbu { offset, dest, base } => 'a: {
                let offset = match base {
                    x if x == Reg::SP => offset,
                    x if offs[x.0 as usize].is_some() => {
                        Imm::new_i32(offs[base.0 as usize].unwrap().wrapping_add(offset.as_i32()))
                    }
                    x if reg_consts[x.0 as usize].is_some() => {
                        offs[dest.0 as usize] = None;
                        let v = u8::from_le_bytes(array::from_fn(|i| {
                            mem[i + reg_consts[x.0 as usize].unwrap() as usize]
                        })) as u32;
                        reg_consts[dest.0 as usize] = Some(v);
                        for i in 0..32 {
                            regs[dest.0 as usize][i] = if v >> i == 0 {
                                zero.clone()
                            } else {
                                one.clone()
                            };
                        }
                        break 'a pc + 4;
                    }
                    _ => return Err(ErtError::Unexpected),
                };
                offs[dest.0 as usize] = None;
                reg_consts[dest.0 as usize] = None;
                for i in 0..32 {
                    let j = i.min(7);
                    if j != i {
                        regs[dest.0 as usize][i] = zero.clone();
                        continue;
                    }
                    regs[dest.0 as usize][i] = vstack[offset.as_u32() as usize * 8 + j].clone();
                }
                pc + 4
            }
            Inst::Lhu { offset, dest, base } => 'a: {
                let offset = match base {
                    x if x == Reg::SP => offset,
                    x if offs[x.0 as usize].is_some() => {
                        Imm::new_i32(offs[base.0 as usize].unwrap().wrapping_add(offset.as_i32()))
                    }
                    x if reg_consts[x.0 as usize].is_some() => {
                        offs[dest.0 as usize] = None;
                        let v = u16::from_le_bytes(array::from_fn(|i| {
                            mem[i + reg_consts[x.0 as usize].unwrap() as usize]
                        })) as u32;
                        reg_consts[dest.0 as usize] = Some(v);
                        for i in 0..32 {
                            regs[dest.0 as usize][i] = if v >> i == 0 {
                                zero.clone()
                            } else {
                                one.clone()
                            };
                        }
                        break 'a pc + 4;
                    }
                    _ => return Err(ErtError::Unexpected),
                };
                offs[dest.0 as usize] = None;
                reg_consts[dest.0 as usize] = None;
                for i in 0..32 {
                    let j = i.min(15);
                    if j != i {
                        regs[dest.0 as usize][i] = zero.clone();
                        continue;
                    }
                    regs[dest.0 as usize][i] = vstack[offset.as_u32() as usize * 8 + j].clone();
                }
                pc + 4
            }
            Inst::Lw { offset, dest, base } => 'a: {
                let offset = match base {
                    x if x == Reg::SP => offset,
                    x if offs[x.0 as usize].is_some() => {
                        Imm::new_i32(offs[base.0 as usize].unwrap().wrapping_add(offset.as_i32()))
                    }
                    x if reg_consts[x.0 as usize].is_some() => {
                        offs[dest.0 as usize] = None;
                        let v = u32::from_le_bytes(array::from_fn(|i| {
                            mem[i + reg_consts[x.0 as usize].unwrap() as usize]
                        }));
                        reg_consts[dest.0 as usize] = Some(v);
                        for i in 0..32 {
                            regs[dest.0 as usize][i] = if v >> i == 0 {
                                zero.clone()
                            } else {
                                one.clone()
                            };
                        }
                        break 'a pc + 4;
                    }
                    _ => return Err(ErtError::Unexpected),
                };
                offs[dest.0 as usize] = None;
                reg_consts[dest.0 as usize] = None;
                for i in 0..32 {
                    regs[dest.0 as usize][i] = vstack[offset.as_u32() as usize * 8 + i].clone();
                }
                pc + 4
            }
            Inst::Sb { offset, src, base } => {
                let offset = match base {
                    x if x == Reg::SP => offset,
                    x if offs[x.0 as usize].is_some() => {
                        Imm::new_i32(offs[base.0 as usize].unwrap().wrapping_add(offset.as_i32()))
                    }
                    _ => return Err(ErtError::Unexpected),
                };
                for i in 0..8 {
                    vstack[offset.as_u32() as usize * 8 + i] = regs[src.0 as usize][i].clone();
                }
                pc + 4
            }
            Inst::Sh { offset, src, base } => {
                let offset = match base {
                    x if x == Reg::SP => offset,
                    x if offs[x.0 as usize].is_some() => {
                        Imm::new_i32(offs[base.0 as usize].unwrap().wrapping_add(offset.as_i32()))
                    }
                    _ => return Err(ErtError::Unexpected),
                };
                for i in 0..16 {
                    vstack[offset.as_u32() as usize * 8 + i] = regs[src.0 as usize][i].clone();
                }
                pc + 4
            }
            Inst::Sw { offset, src, base } => {
                let offset = match base {
                    x if x == Reg::SP => offset,
                    x if offs[x.0 as usize].is_some() => {
                        Imm::new_i32(offs[base.0 as usize].unwrap().wrapping_add(offset.as_i32()))
                    }
                    _ => return Err(ErtError::Unexpected),
                };
                for i in 0..32 {
                    vstack[offset.as_u32() as usize * 8 + i] = regs[src.0 as usize][i].clone();
                }
                pc + 4
            }
            //control flow
            Inst::Jal { offset, dest } => {
                if dest.0 != 0 {
                    //assume call
                    rstack[rsp as usize] = pc + 4;
                    rsp += 1;
                }
                pc.wrapping_add_signed(offset.as_i32())
            }
            Inst::Jalr { offset, base, dest } => {
                //assume return
                rsp -= 1;
                rstack[rsp as usize]
            }
            Inst::Beq { offset, src1, src2 } => {
                let Some(a) = reg_consts[src1.0 as usize] else {
                    return Err(ErtError::Unexpected);
                };
                let Some(b) = reg_consts[src2.0 as usize] else {
                    return Err(ErtError::Unexpected);
                };
                if a == b {
                    pc.wrapping_add_signed(offset.as_i32())
                } else {
                    pc + 4
                }
            }
            Inst::Bne { offset, src1, src2 } => {
                let Some(a) = reg_consts[src1.0 as usize] else {
                    return Err(ErtError::Unexpected);
                };
                let Some(b) = reg_consts[src2.0 as usize] else {
                    return Err(ErtError::Unexpected);
                };
                if a != b {
                    pc.wrapping_add_signed(offset.as_i32())
                } else {
                    pc + 4
                }
            }
            Inst::Bgeu { offset, src1, src2 } => {
                let Some(a) = reg_consts[src1.0 as usize] else {
                    return Err(ErtError::Unexpected);
                };
                let Some(b) = reg_consts[src2.0 as usize] else {
                    return Err(ErtError::Unexpected);
                };
                if a >= b {
                    pc.wrapping_add_signed(offset.as_i32())
                } else {
                    pc + 4
                }
            }
            Inst::Bltu { offset, src1, src2 } => {
                let Some(a) = reg_consts[src1.0 as usize] else {
                    return Err(ErtError::Unexpected);
                };
                let Some(b) = reg_consts[src2.0 as usize] else {
                    return Err(ErtError::Unexpected);
                };
                if a < b {
                    pc.wrapping_add_signed(offset.as_i32())
                } else {
                    pc + 4
                }
            }
            Inst::Bge { offset, src1, src2 } => {
                let Some(a) = reg_consts[src1.0 as usize] else {
                    return Err(ErtError::Unexpected);
                };
                let Some(b) = reg_consts[src2.0 as usize] else {
                    return Err(ErtError::Unexpected);
                };
                if (a as i32) >= (b as i32) {
                    pc.wrapping_add_signed(offset.as_i32())
                } else {
                    pc + 4
                }
            }
            Inst::Blt { offset, src1, src2 } => {
                let Some(a) = reg_consts[src1.0 as usize] else {
                    return Err(ErtError::Unexpected);
                };
                let Some(b) = reg_consts[src2.0 as usize] else {
                    return Err(ErtError::Unexpected);
                };
                if (a as i32) < (b as i32) {
                    pc.wrapping_add_signed(offset.as_i32())
                } else {
                    pc + 4
                }
            }
            //ecall
            Inst::Ecall => match reg_consts[Reg::A0.0 as usize] {
                Some(0) => {
                    let h = hash(&regs[Reg::A1.0 as usize..][..(256 / 32)])
                        .map_err(|e| ErtError::Emitted(e))?;
                    for ((r, a), h) in regs[Reg::A1.0 as usize..][..(256 / 32)]
                        .iter_mut()
                        .zip(reg_consts[Reg::A1.0 as usize..][..(256 / 32)].iter_mut())
                        .zip(h.chunks_exact(256 / 32))
                    {
                        let h = u32::from_le_bytes(array::from_fn(|i| h[i]));
                        *a = Some(h);
                        for i in 0..32 {
                            r[i] = if h >> i == 0 {
                                zero.clone()
                            } else {
                                one.clone()
                            };
                        }
                    }
                    pc + 4
                }
                _ => return Err(ErtError::Unexpected),
            },
            _ => return Err(ErtError::Unexpected),
        }
    }
}
