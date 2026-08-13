use core::array;

use rv_asm::{Imm, Inst, Reg};

use crate::machine::Machine;
use crate::{ErtError, machine::LoadAddress, simple_add};

pub(crate) enum Flow {
    Next(u32),
    Exit,
}

#[derive(Clone, Copy)]
enum BitOp {
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy)]
enum Shift {
    Left,
    LogicalRight,
    ArithmeticRight,
}

#[derive(Clone, Copy)]
enum LoadKind {
    ByteSigned,
    HalfSigned,
    ByteUnsigned,
    HalfUnsigned,
    Word,
}

#[derive(Clone, Copy)]
enum Branch {
    Equal,
    NotEqual,
    GreaterEqualUnsigned,
    LessThanUnsigned,
    GreaterEqualSigned,
    LessThanSigned,
}

pub(crate) fn execute<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    instruction: Inst,
) -> Result<Flow, ErtError<E>> {
    match instruction {
        Inst::Lui { uimm, dest } => constant(machine, dest, uimm.as_u32(), machine.pc),
        Inst::Auipc { uimm, dest } => constant(
            machine,
            dest,
            uimm.as_u32().wrapping_add(machine.pc),
            machine.pc,
        ),

        Inst::Addi { imm, dest, src1 } => add_immediate(machine, imm, dest, src1),
        Inst::Add { dest, src1, src2 } => add(machine, dest, src1, src2),
        Inst::Sub { dest, src1, src2 } => subtract(machine, dest, src1, src2),
        Inst::And { dest, src1, src2 } => bitwise(machine, dest, src1, src2, BitOp::And),
        Inst::Or { dest, src1, src2 } => bitwise(machine, dest, src1, src2, BitOp::Or),
        Inst::Xor { dest, src1, src2 } => bitwise(machine, dest, src1, src2, BitOp::Xor),
        Inst::Andi { imm, dest, src1 } => bitwise_immediate(machine, imm, dest, src1, BitOp::And),
        Inst::Ori { imm, dest, src1 } => bitwise_immediate(machine, imm, dest, src1, BitOp::Or),
        Inst::Xori { imm, dest, src1 } => bitwise_immediate(machine, imm, dest, src1, BitOp::Xor),
        Inst::Slli { imm, dest, src1 } => shift(machine, imm, dest, src1, Shift::Left),
        Inst::Srli { imm, dest, src1 } => shift(machine, imm, dest, src1, Shift::LogicalRight),
        Inst::Srai { imm, dest, src1 } => shift(machine, imm, dest, src1, Shift::ArithmeticRight),

        Inst::Lb { offset, dest, base } => load(machine, offset, dest, base, LoadKind::ByteSigned),
        Inst::Lh { offset, dest, base } => load(machine, offset, dest, base, LoadKind::HalfSigned),
        Inst::Lbu { offset, dest, base } => {
            load(machine, offset, dest, base, LoadKind::ByteUnsigned)
        }
        Inst::Lhu { offset, dest, base } => {
            load(machine, offset, dest, base, LoadKind::HalfUnsigned)
        }
        Inst::Lw { offset, dest, base } => load(machine, offset, dest, base, LoadKind::Word),
        Inst::Sb { offset, src, base } => store(machine, offset, src, base, 8),
        Inst::Sh { offset, src, base } => store(machine, offset, src, base, 16),
        Inst::Sw { offset, src, base } => store(machine, offset, src, base, 32),

        Inst::Jal { offset, dest } => jump_and_link(machine, offset, dest),
        Inst::Jalr { .. } => return_from_call(machine),
        Inst::Beq { offset, src1, src2 } => branch(machine, offset, src1, src2, Branch::Equal),
        Inst::Bne { offset, src1, src2 } => branch(machine, offset, src1, src2, Branch::NotEqual),
        Inst::Bgeu { offset, src1, src2 } => {
            branch(machine, offset, src1, src2, Branch::GreaterEqualUnsigned)
        }
        Inst::Bltu { offset, src1, src2 } => {
            branch(machine, offset, src1, src2, Branch::LessThanUnsigned)
        }
        Inst::Bge { offset, src1, src2 } => {
            branch(machine, offset, src1, src2, Branch::GreaterEqualSigned)
        }
        Inst::Blt { offset, src1, src2 } => {
            branch(machine, offset, src1, src2, Branch::LessThanSigned)
        }

        Inst::Ecall => ecall(machine),
        _ => Err(ErtError::Unexpected),
    }
}

fn next<W, E>(machine: &Machine<'_, W, E>) -> Result<Flow, ErtError<E>> {
    Ok(Flow::Next(machine.pc + 4))
}

fn constant<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    dest: Reg,
    value: u32,
    pc: u32,
) -> Result<Flow, ErtError<E>> {
    machine.write_constant(dest, value);
    Ok(Flow::Next(pc + 4))
}

fn add_immediate<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    imm: Imm,
    dest: Reg,
    src1: Reg,
) -> Result<Flow, ErtError<E>> {
    if dest == Reg::SP {
        machine.sp = machine.sp.wrapping_add_signed(imm.as_i32());
        for offset in machine.offs.iter_mut().flatten() {
            *offset = offset.wrapping_add(imm.as_i32());
        }
    }
    if src1 != dest {
        machine.offs[dest.0 as usize] = None;
    }
    if src1 == Reg::SP {
        machine.offs[dest.0 as usize] = Some(imm.as_i32());
    } else if let Some(offset) = machine.offs[src1.0 as usize] {
        machine.offs[dest.0 as usize] = Some(offset.wrapping_add(imm.as_i32()));
    }

    let word = array::from_fn(|i| {
        if (imm.as_i32() as u32 >> i) & 1 == 1 {
            machine.one.clone()
        } else {
            machine.zero.clone()
        }
    });
    machine.regs[dest.0 as usize] = simple_add(
        machine.t,
        &machine.regs[src1.0 as usize],
        &word,
        machine.zero.clone(),
        machine.zero.clone(),
        machine.one.clone(),
    )
    .map_err(ErtError::Emitted)?;
    machine.reg_consts[dest.0 as usize] = match machine.reg_consts[src1.0 as usize] {
        Some(value) => Some(value.wrapping_add_signed(imm.as_i32())),
        None => None,
    };
    next(machine)
}

fn add<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    dest: Reg,
    src1: Reg,
    src2: Reg,
) -> Result<Flow, ErtError<E>> {
    if src1 != dest && src2 != dest {
        machine.offs[dest.0 as usize] = None;
    }
    if let Some(constant) = machine.reg_consts[src2.0 as usize] {
        if src1 == Reg::SP {
            machine.offs[dest.0 as usize] = Some(constant as i32);
        } else if let Some(offset) = machine.offs[src1.0 as usize] {
            machine.offs[dest.0 as usize] = Some(offset.wrapping_add_unsigned(constant));
        }
    }
    if let Some(constant) = machine.reg_consts[src1.0 as usize] {
        if src2 == Reg::SP {
            machine.offs[dest.0 as usize] = Some(constant as i32);
        } else if let Some(offset) = machine.offs[src2.0 as usize] {
            machine.offs[dest.0 as usize] = Some(offset.wrapping_add_unsigned(constant));
        }
    }

    machine.regs[dest.0 as usize] = simple_add(
        machine.t,
        &machine.regs[src1.0 as usize],
        &machine.regs[src2.0 as usize],
        machine.zero.clone(),
        machine.zero.clone(),
        machine.one.clone(),
    )
    .map_err(ErtError::Emitted)?;
    machine.reg_consts[dest.0 as usize] = match (
        machine.reg_consts[src1.0 as usize],
        machine.reg_consts[src2.0 as usize],
    ) {
        (Some(a), Some(b)) => Some(a.wrapping_add(b)),
        _ => None,
    };
    next(machine)
}

fn subtract<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    dest: Reg,
    src1: Reg,
    src2: Reg,
) -> Result<Flow, ErtError<E>> {
    if src1 != dest && src2 != dest {
        machine.offs[dest.0 as usize] = None;
    }
    if let Some(constant) = machine.reg_consts[src2.0 as usize] {
        if src1 == Reg::SP {
            machine.offs[dest.0 as usize] = Some(-(constant as i32));
        } else if let Some(offset) = machine.offs[src1.0 as usize] {
            machine.offs[dest.0 as usize] = Some(offset.wrapping_sub_unsigned(constant));
        }
    }
    if let Some(constant) = machine.reg_consts[src1.0 as usize] {
        if src2 == Reg::SP {
            machine.offs[dest.0 as usize] = Some(-(constant as i32));
        } else if let Some(offset) = machine.offs[src2.0 as usize] {
            machine.offs[dest.0 as usize] = Some(offset.wrapping_sub_unsigned(constant));
        }
    }

    let mut left = machine.regs[src1.0 as usize].clone();
    for bit in &mut left {
        *bit = machine
            .t
            .bitxor(bit.clone(), machine.one.clone())
            .map_err(ErtError::Emitted)?;
    }
    machine.regs[dest.0 as usize] = simple_add(
        machine.t,
        &left,
        &machine.regs[src2.0 as usize],
        machine.one.clone(),
        machine.zero.clone(),
        machine.one.clone(),
    )
    .map_err(ErtError::Emitted)?;
    machine.reg_consts[dest.0 as usize] =
        match (machine.offs[src1.0 as usize], machine.offs[src2.0 as usize]) {
            (Some(a), Some(b)) => Some(a.wrapping_sub(b) as u32),
            _ => match (
                machine.reg_consts[src1.0 as usize],
                machine.reg_consts[src2.0 as usize],
            ) {
                (Some(a), Some(b)) => Some(a.wrapping_sub(b)),
                _ => None,
            },
        };
    next(machine)
}

fn bitwise<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    dest: Reg,
    src1: Reg,
    src2: Reg,
    operation: BitOp,
) -> Result<Flow, ErtError<E>> {
    machine.offs[dest.0 as usize] = None;
    machine.reg_consts[dest.0 as usize] = match (
        machine.reg_consts[src1.0 as usize],
        machine.reg_consts[src2.0 as usize],
    ) {
        (Some(a), Some(b)) => Some(match operation {
            BitOp::And => a & b,
            BitOp::Or => a | b,
            BitOp::Xor => a ^ b,
        }),
        _ => None,
    };
    for bit in 0..32 {
        machine.regs[dest.0 as usize][bit] = match operation {
            BitOp::And => machine.t.bitand(
                machine.regs[src1.0 as usize][bit].clone(),
                machine.regs[src2.0 as usize][bit].clone(),
            ),
            BitOp::Or => machine.t.bitor(
                machine.regs[src1.0 as usize][bit].clone(),
                machine.regs[src2.0 as usize][bit].clone(),
            ),
            BitOp::Xor => machine.t.bitxor(
                machine.regs[src1.0 as usize][bit].clone(),
                machine.regs[src2.0 as usize][bit].clone(),
            ),
        }
        .map_err(ErtError::Emitted)?;
    }
    next(machine)
}

fn bitwise_immediate<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    imm: Imm,
    dest: Reg,
    src1: Reg,
    operation: BitOp,
) -> Result<Flow, ErtError<E>> {
    let immediate = imm.as_i32() as u32;
    machine.offs[dest.0 as usize] = None;
    machine.reg_consts[dest.0 as usize] =
        machine.reg_consts[src1.0 as usize].map(|value| match operation {
            BitOp::And => value & immediate,
            BitOp::Or => value | immediate,
            BitOp::Xor => value ^ immediate,
        });
    for bit in 0..32 {
        machine.regs[dest.0 as usize][bit] = if immediate & (1 << bit) == 0 {
            match operation {
                BitOp::And => machine.zero.clone(),
                BitOp::Or | BitOp::Xor => machine.regs[src1.0 as usize][bit].clone(),
            }
        } else {
            match operation {
                BitOp::And | BitOp::Or => machine.one.clone(),
                BitOp::Xor => machine
                    .t
                    .bitxor(
                        machine.regs[src1.0 as usize][bit].clone(),
                        machine.one.clone(),
                    )
                    .map_err(ErtError::Emitted)?,
            }
        };
    }
    next(machine)
}

fn shift<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    imm: Imm,
    dest: Reg,
    src1: Reg,
    direction: Shift,
) -> Result<Flow, ErtError<E>> {
    let amount = imm.as_u32() & 31;
    machine.offs[dest.0 as usize] = None;
    machine.reg_consts[dest.0 as usize] =
        machine.reg_consts[src1.0 as usize].map(|value| match direction {
            Shift::Left => value << amount,
            Shift::LogicalRight => value >> amount,
            Shift::ArithmeticRight => (value as i32 >> amount) as u32,
        });

    match direction {
        Shift::Left | Shift::LogicalRight => {
            for bit in machine.regs[dest.0 as usize].iter_mut() {
                *bit = machine.zero.clone();
            }
        }
        Shift::ArithmeticRight => {
            let sign = machine.regs[dest.0 as usize][31].clone();
            for bit in machine.regs[dest.0 as usize].iter_mut() {
                *bit = sign.clone();
            }
        }
    }

    for bit in 0..32 {
        let target = match direction {
            Shift::Left => {
                if amount + bit >= 32 {
                    continue;
                }
                amount + bit
            }
            Shift::LogicalRight | Shift::ArithmeticRight => {
                if amount > bit {
                    continue;
                }
                bit - amount
            }
        };
        machine.regs[dest.0 as usize][target as usize] =
            machine.regs[src1.0 as usize][bit as usize].clone();
    }
    next(machine)
}

fn load<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    offset: Imm,
    dest: Reg,
    base: Reg,
    kind: LoadKind,
) -> Result<Flow, ErtError<E>> {
    match machine.load_address(base, offset)? {
        LoadAddress::Concrete(address) => {
            let value = match kind {
                LoadKind::ByteSigned => {
                    i8::from_le_bytes(array::from_fn(|i| machine.mem[i + address as usize])) as i32
                        as u32
                }
                LoadKind::HalfSigned => {
                    i16::from_le_bytes(array::from_fn(|i| machine.mem[i + address as usize])) as i32
                        as u32
                }
                LoadKind::ByteUnsigned => {
                    u8::from_le_bytes(array::from_fn(|i| machine.mem[i + address as usize])) as u32
                }
                LoadKind::HalfUnsigned => {
                    u16::from_le_bytes(array::from_fn(|i| machine.mem[i + address as usize])) as u32
                }
                LoadKind::Word => {
                    u32::from_le_bytes(array::from_fn(|i| machine.mem[i + address as usize]))
                }
            };
            machine.write_constant(dest, value);
        }
        LoadAddress::Stack(offset) => {
            machine.offs[dest.0 as usize] = None;
            machine.reg_consts[dest.0 as usize] = None;
            let (width, sign_extend) = match kind {
                LoadKind::ByteSigned => (8, true),
                LoadKind::HalfSigned => (16, true),
                LoadKind::ByteUnsigned => (8, false),
                LoadKind::HalfUnsigned => (16, false),
                LoadKind::Word => (32, false),
            };
            for bit in 0..32 {
                let source_bit = bit.min(width - 1);
                if !sign_extend && source_bit != bit {
                    machine.regs[dest.0 as usize][bit] = machine.zero.clone();
                    continue;
                }
                machine.regs[dest.0 as usize][bit] =
                    machine.vstack[offset.as_u32() as usize * 8 + source_bit].clone();
            }
        }
    }
    next(machine)
}

fn store<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    offset: Imm,
    src: Reg,
    base: Reg,
    width: usize,
) -> Result<Flow, ErtError<E>> {
    let offset = machine.stack_offset(base, offset)?;
    for bit in 0..width {
        machine.vstack[offset.as_u32() as usize * 8 + bit] =
            machine.regs[src.0 as usize][bit].clone();
    }
    next(machine)
}

fn jump_and_link<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    offset: Imm,
    dest: Reg,
) -> Result<Flow, ErtError<E>> {
    if dest.0 != 0 {
        machine.rstack[machine.rsp as usize] = machine.pc + 4;
        machine.rsp += 1;
    }
    Ok(Flow::Next(machine.pc.wrapping_add_signed(offset.as_i32())))
}

fn return_from_call<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
) -> Result<Flow, ErtError<E>> {
    machine.rsp -= 1;
    Ok(Flow::Next(machine.rstack[machine.rsp as usize]))
}

fn branch<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    offset: Imm,
    src1: Reg,
    src2: Reg,
    condition: Branch,
) -> Result<Flow, ErtError<E>> {
    let Some(left) = machine.reg_consts[src1.0 as usize] else {
        return Err(ErtError::Unexpected);
    };
    let Some(right) = machine.reg_consts[src2.0 as usize] else {
        return Err(ErtError::Unexpected);
    };
    let taken = match condition {
        Branch::Equal => left == right,
        Branch::NotEqual => left != right,
        Branch::GreaterEqualUnsigned => left >= right,
        Branch::LessThanUnsigned => left < right,
        Branch::GreaterEqualSigned => (left as i32) >= (right as i32),
        Branch::LessThanSigned => (left as i32) < (right as i32),
    };
    Ok(Flow::Next(if taken {
        machine.pc.wrapping_add_signed(offset.as_i32())
    } else {
        machine.pc + 4
    }))
}

fn ecall<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
) -> Result<Flow, ErtError<E>> {
    match machine.reg_consts[Reg::A0.0 as usize] {
        Some(0) => {
            let hash = (machine.hash)(&machine.regs[Reg::A1.0 as usize..][..(256 / 32)])
                .map_err(ErtError::Emitted)?;
            for ((register, constant), chunk) in machine.regs[Reg::A1.0 as usize..][..(256 / 32)]
                .iter_mut()
                .zip(machine.reg_consts[Reg::A1.0 as usize..][..(256 / 32)].iter_mut())
                .zip(hash.chunks_exact(256 / 32))
            {
                let value = u32::from_le_bytes(array::from_fn(|i| chunk[i]));
                *constant = Some(value);
                for bit in 0..32 {
                    register[bit] = if value >> bit == 0 {
                        machine.zero.clone()
                    } else {
                        machine.one.clone()
                    };
                }
            }
            next(machine)
        }
        Some(0xffff_ffff) => {
            if machine.sp + 1 != machine.vstack.len() as u32 {
                return Err(ErtError::Unexpected);
            }
            Ok(Flow::Exit)
        }
        _ => Err(ErtError::Unexpected),
    }
}
