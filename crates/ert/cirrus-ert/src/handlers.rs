use core::array;

use cirrus_ert_core::{compare_word, partial_bitwise_word, BitOp, ComparePredicate};
use rv_asm::{Imm, Inst, Reg};

use crate::machine::{add_bits, Machine};
use crate::{machine::LoadAddress, EcallOutcome, ErtError};

pub(crate) enum Flow {
    Next(u32),
    Exit,
}

#[derive(Clone, Copy)]
enum Shift {
    Left,
    LogicalRight,
    ArithmeticRight,
}

#[derive(Clone, Copy)]
enum Product {
    Low,
    HighSigned,
    HighSignedUnsigned,
    HighUnsigned,
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
        Inst::Slti { imm, dest, src1 } => {
            set_less_than_immediate(machine, dest, src1, imm, ComparePredicate::LtS)
        }
        Inst::Sltiu { imm, dest, src1 } => {
            set_less_than_immediate(machine, dest, src1, imm, ComparePredicate::LtU)
        }
        Inst::Add { dest, src1, src2 } => add(machine, dest, src1, src2),
        Inst::Sub { dest, src1, src2 } => subtract(machine, dest, src1, src2),
        Inst::Slt { dest, src1, src2 } => {
            set_less_than(machine, dest, src1, src2, ComparePredicate::LtS)
        }
        Inst::Sltu { dest, src1, src2 } => {
            set_less_than(machine, dest, src1, src2, ComparePredicate::LtU)
        }
        Inst::And { dest, src1, src2 } => bitwise(machine, dest, src1, src2, BitOp::And),
        Inst::Or { dest, src1, src2 } => bitwise(machine, dest, src1, src2, BitOp::Or),
        Inst::Xor { dest, src1, src2 } => bitwise(machine, dest, src1, src2, BitOp::Xor),
        Inst::Andi { imm, dest, src1 } => bitwise_immediate(machine, imm, dest, src1, BitOp::And),
        Inst::Ori { imm, dest, src1 } => bitwise_immediate(machine, imm, dest, src1, BitOp::Or),
        Inst::Xori { imm, dest, src1 } => bitwise_immediate(machine, imm, dest, src1, BitOp::Xor),
        Inst::Slli { imm, dest, src1 } => shift(machine, imm, dest, src1, Shift::Left),
        Inst::Srli { imm, dest, src1 } => shift(machine, imm, dest, src1, Shift::LogicalRight),
        Inst::Srai { imm, dest, src1 } => shift(machine, imm, dest, src1, Shift::ArithmeticRight),
        Inst::Sll { dest, src1, src2 } => runtime_shift(machine, dest, src1, src2, Shift::Left),
        Inst::Srl { dest, src1, src2 } => {
            runtime_shift(machine, dest, src1, src2, Shift::LogicalRight)
        }
        Inst::Sra { dest, src1, src2 } => {
            runtime_shift(machine, dest, src1, src2, Shift::ArithmeticRight)
        }
        Inst::Mul { dest, src1, src2 } => multiply(machine, dest, src1, src2, Product::Low),
        Inst::Mulh { dest, src1, src2 } => multiply(machine, dest, src1, src2, Product::HighSigned),
        Inst::Mulhsu { dest, src1, src2 } => {
            multiply(machine, dest, src1, src2, Product::HighSignedUnsigned)
        }
        Inst::Mulhu { dest, src1, src2 } => {
            multiply(machine, dest, src1, src2, Product::HighUnsigned)
        }

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
        Inst::Jalr { offset, base, dest } => jump_and_link_register(machine, offset, base, dest),
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

        Inst::Ecall => match machine.t.ecall(
            &mut machine.regs[..],
            &mut machine.reg_consts[..],
            &mut machine.offs[..],
            &machine.zero,
            &machine.one,
        ) {
            Ok(EcallOutcome::Continue) => next(machine),
            Ok(EcallOutcome::Exit) if machine.sp == machine.stack_top => Ok(Flow::Exit),
            Ok(EcallOutcome::Exit) | Ok(EcallOutcome::Unexpected) => Err(ErtError::Unexpected),
            Err(e) => Err(ErtError::Emitted(e)),
        },
        _ => Err(ErtError::Unexpected),
    }
}

fn set_less_than<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    dest: Reg,
    src1: Reg,
    src2: Reg,
    predicate: ComparePredicate,
) -> Result<Flow, ErtError<E>> {
    machine.offs[dest.0 as usize] = None;
    if let (Some(left), Some(right)) = (
        machine.reg_consts[src1.0 as usize],
        machine.reg_consts[src2.0 as usize],
    ) {
        let result = compare_concrete(left, right, predicate) as u32;
        machine.write_constant(dest, result);
        return next(machine);
    }
    let comparison = compare_word(
        machine.t,
        &machine.regs[src1.0 as usize],
        &machine.regs[src2.0 as usize],
        predicate,
        &machine.one,
    )
    .map_err(ErtError::Emitted)?;
    let mut word = machine.word_from_constant(0);
    word[0] = comparison;
    machine.regs[dest.0 as usize] = word;
    machine.reg_consts[dest.0 as usize] = None;
    next(machine)
}

fn set_less_than_immediate<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    dest: Reg,
    src1: Reg,
    imm: Imm,
    predicate: ComparePredicate,
) -> Result<Flow, ErtError<E>> {
    let immediate = imm.as_i32() as u32;
    machine.offs[dest.0 as usize] = None;
    if let Some(left) = machine.reg_consts[src1.0 as usize] {
        machine.write_constant(dest, compare_concrete(left, immediate, predicate) as u32);
        return next(machine);
    }
    let right = machine.word_from_constant(immediate);
    let comparison = compare_word(
        machine.t,
        &machine.regs[src1.0 as usize],
        &right,
        predicate,
        &machine.one,
    )
    .map_err(ErtError::Emitted)?;
    let mut word = machine.word_from_constant(0);
    word[0] = comparison;
    machine.regs[dest.0 as usize] = word;
    machine.reg_consts[dest.0 as usize] = None;
    next(machine)
}

fn compare_concrete(left: u32, right: u32, predicate: ComparePredicate) -> bool {
    match predicate {
        ComparePredicate::Eq => left == right,
        ComparePredicate::Ne => left != right,
        ComparePredicate::GeU => left >= right,
        ComparePredicate::LtU => left < right,
        ComparePredicate::GeS => (left as i32) >= (right as i32),
        ComparePredicate::LtS => (left as i32) < (right as i32),
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
            *offset = offset.wrapping_sub(imm.as_i32());
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

    if let Some(value) = machine.reg_consts[src1.0 as usize] {
        machine.write_constant(dest, value.wrapping_add_signed(imm.as_i32()));
        return next(machine);
    }

    let word = array::from_fn(|i| {
        if (imm.as_i32() as u32 >> i) & 1 == 1 {
            machine.one.clone()
        } else {
            machine.zero.clone()
        }
    });
    machine.regs[dest.0 as usize] = add_bits(
        machine.t,
        &machine.regs[src1.0 as usize],
        &word,
        machine.zero.clone(),
    )
    .map_err(ErtError::Emitted)?;
    machine.reg_consts[dest.0 as usize] = None;
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

    if let (Some(a), Some(b)) = (
        machine.reg_consts[src1.0 as usize],
        machine.reg_consts[src2.0 as usize],
    ) {
        machine.write_constant(dest, a.wrapping_add(b));
        return next(machine);
    }

    machine.regs[dest.0 as usize] = add_bits(
        machine.t,
        &machine.regs[src1.0 as usize],
        &machine.regs[src2.0 as usize],
        machine.zero.clone(),
    )
    .map_err(ErtError::Emitted)?;
    machine.reg_consts[dest.0 as usize] = None;
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

    let concrete = match (machine.offs[src1.0 as usize], machine.offs[src2.0 as usize]) {
        (Some(a), Some(b)) => Some(a.wrapping_sub(b) as u32),
        _ => match (
            machine.reg_consts[src1.0 as usize],
            machine.reg_consts[src2.0 as usize],
        ) {
            (Some(a), Some(b)) => Some(a.wrapping_sub(b)),
            _ => None,
        },
    };
    if let Some(value) = concrete {
        machine.write_constant(dest, value);
        return next(machine);
    }

    let mut left = machine.regs[src1.0 as usize].clone();
    for bit in &mut left {
        *bit = machine
            .t
            .bitxor(bit.clone(), machine.one.clone())
            .map_err(ErtError::Emitted)?;
    }
    machine.regs[dest.0 as usize] = add_bits(
        machine.t,
        &left,
        &machine.regs[src2.0 as usize],
        machine.one.clone(),
    )
    .map_err(ErtError::Emitted)?;
    machine.reg_consts[dest.0 as usize] = None;
    next(machine)
}

fn fold(operation: BitOp, a: u32, b: u32) -> u32 {
    match operation {
        BitOp::And => a & b,
        BitOp::Or => a | b,
        BitOp::Xor => a ^ b,
    }
}

fn degenerate(operation: BitOp, constant: u32) -> bool {
    match operation {
        BitOp::And => constant == 0,
        BitOp::Or => constant == u32::MAX,
        BitOp::Xor => false,
    }
}

fn degenerate_value(operation: BitOp) -> u32 {
    match operation {
        BitOp::And => 0,
        BitOp::Or => u32::MAX,
        BitOp::Xor => unreachable!("Xor is never degenerate"),
    }
}

fn bitwise<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    dest: Reg,
    src1: Reg,
    src2: Reg,
    operation: BitOp,
) -> Result<Flow, ErtError<E>> {
    machine.offs[dest.0 as usize] = None;
    let c1 = machine.reg_consts[src1.0 as usize];
    let c2 = machine.reg_consts[src2.0 as usize];
    if let (Some(a), Some(b)) = (c1, c2) {
        machine.write_constant(dest, fold(operation, a, b));
        return next(machine);
    }
    let partial = match (c1, c2) {
        (Some(constant), None) => Some((constant, src2)),
        (None, Some(constant)) => Some((constant, src1)),
        _ => None,
    };
    if let Some((constant, symbolic)) = partial {
        if degenerate(operation, constant) {
            machine.write_constant(dest, degenerate_value(operation));
            return next(machine);
        }
        machine.reg_consts[dest.0 as usize] = None;
        machine.regs[dest.0 as usize] = partial_bitwise_word(
            machine.t,
            constant,
            &machine.regs[symbolic.0 as usize],
            &machine.zero,
            &machine.one,
            operation,
        )
        .map_err(ErtError::Emitted)?;
        return next(machine);
    }
    machine.reg_consts[dest.0 as usize] = None;
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
    if let Some(value) = machine.reg_consts[src1.0 as usize] {
        machine.write_constant(dest, fold(operation, value, immediate));
        return next(machine);
    }
    if degenerate(operation, immediate) {
        machine.write_constant(dest, degenerate_value(operation));
        return next(machine);
    }
    machine.reg_consts[dest.0 as usize] = None;
    machine.regs[dest.0 as usize] = partial_bitwise_word(
        machine.t,
        immediate,
        &machine.regs[src1.0 as usize],
        &machine.zero,
        &machine.one,
        operation,
    )
    .map_err(ErtError::Emitted)?;
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
    let source = machine.regs[src1.0 as usize].clone();
    machine.offs[dest.0 as usize] = None;
    machine.reg_consts[dest.0 as usize] =
        machine.reg_consts[src1.0 as usize].map(|value| match direction {
            Shift::Left => value << amount,
            Shift::LogicalRight => value >> amount,
            Shift::ArithmeticRight => (value as i32 >> amount) as u32,
        });
    machine.regs[dest.0 as usize] = shifted_word(machine, &source, amount, direction);
    next(machine)
}

fn runtime_shift<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    dest: Reg,
    src1: Reg,
    src2: Reg,
    direction: Shift,
) -> Result<Flow, ErtError<E>> {
    let source = machine.regs[src1.0 as usize].clone();
    let amount_word = machine.regs[src2.0 as usize].clone();
    machine.offs[dest.0 as usize] = None;

    if let Some(amount) = machine.reg_consts[src2.0 as usize] {
        machine.reg_consts[dest.0 as usize] =
            machine.reg_consts[src1.0 as usize].map(|value| match direction {
                Shift::Left => value << (amount & 31),
                Shift::LogicalRight => value >> (amount & 31),
                Shift::ArithmeticRight => (value as i32 >> (amount & 31)) as u32,
            });
        machine.regs[dest.0 as usize] = shifted_word(machine, &source, amount & 31, direction);
        return next(machine);
    }

    let mut shifted = source;
    for stage in 0..5 {
        let candidate = shifted_word(machine, &shifted, 1 << stage, direction);
        shifted = select_word(machine, amount_word[stage].clone(), &candidate, &shifted)?;
    }
    machine.reg_consts[dest.0 as usize] = None;
    machine.regs[dest.0 as usize] = shifted;
    next(machine)
}

fn shifted_word<W: Clone, E: core::error::Error>(
    machine: &Machine<'_, W, E>,
    source: &[W; 32],
    amount: u32,
    direction: Shift,
) -> [W; 32] {
    let fill = match direction {
        Shift::ArithmeticRight => source[31].clone(),
        Shift::Left | Shift::LogicalRight => machine.zero.clone(),
    };
    array::from_fn(|destination_bit| match direction {
        Shift::Left if destination_bit >= amount as usize => {
            source[destination_bit - amount as usize].clone()
        }
        Shift::LogicalRight | Shift::ArithmeticRight
            if destination_bit + (amount as usize) < 32 =>
        {
            source[destination_bit + (amount as usize)].clone()
        }
        _ => fill.clone(),
    })
}

fn select_word<W: Clone, E: core::error::Error, const N: usize>(
    machine: &mut Machine<'_, W, E>,
    condition: W,
    then: &[W; N],
    r#else: &[W; N],
) -> Result<[W; N], ErtError<E>> {
    let mut selected = r#else.clone();
    for bit in 0..N {
        let difference = machine
            .t
            .bitxor(then[bit].clone(), r#else[bit].clone())
            .map_err(ErtError::Emitted)?;
        let difference = machine
            .t
            .bitand(condition.clone(), difference)
            .map_err(ErtError::Emitted)?;
        selected[bit] = machine
            .t
            .bitxor(r#else[bit].clone(), difference)
            .map_err(ErtError::Emitted)?;
    }
    Ok(selected)
}

fn multiply<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    dest: Reg,
    src1: Reg,
    src2: Reg,
    product: Product,
) -> Result<Flow, ErtError<E>> {
    let left = machine.regs[src1.0 as usize].clone();
    let right = machine.regs[src2.0 as usize].clone();
    let left_constant = machine.reg_consts[src1.0 as usize];
    let right_constant = machine.reg_consts[src2.0 as usize];
    machine.offs[dest.0 as usize] = None;

    if let (Some(left), Some(right)) = (left_constant, right_constant) {
        machine.write_constant(dest, concrete_product(product, left, right));
        return next(machine);
    }
    if left_constant == Some(0) || right_constant == Some(0) {
        machine.write_constant(dest, 0);
        return next(machine);
    }

    let result = match product {
        Product::Low => {
            if let Some(constant) = left_constant {
                multiply_by_constant(machine, &right, constant)?
            } else if let Some(constant) = right_constant {
                multiply_by_constant(machine, &left, constant)?
            } else {
                multiply_selected(machine, &left, &right)?
            }
        }
        Product::HighSigned | Product::HighSignedUnsigned | Product::HighUnsigned => {
            let full_product = if let Some(constant) = left_constant {
                let multiplicand = zero_extend(&right, &machine.zero);
                multiply_by_constant(machine, &multiplicand, constant)?
            } else if let Some(constant) = right_constant {
                let multiplicand = zero_extend(&left, &machine.zero);
                multiply_by_constant(machine, &multiplicand, constant)?
            } else {
                let multiplicand = zero_extend(&left, &machine.zero);
                multiply_selected(machine, &multiplicand, &right)?
            };
            let high = array::from_fn(|bit| full_product[32 + bit].clone());
            correct_high_product(
                machine,
                high,
                &left,
                &right,
                left_constant,
                right_constant,
                product,
            )?
        }
    };
    machine.reg_consts[dest.0 as usize] = None;
    machine.regs[dest.0 as usize] = result;
    next(machine)
}

fn concrete_product(product: Product, left: u32, right: u32) -> u32 {
    match product {
        Product::Low => left.wrapping_mul(right),
        Product::HighSigned => (((left as i32 as i64) * (right as i32 as i64)) >> 32) as u32,
        Product::HighSignedUnsigned => {
            (((left as i32 as i64) * (right as u64 as i64)) >> 32) as u32
        }
        Product::HighUnsigned => ((left as u64 * right as u64) >> 32) as u32,
    }
}

fn multiply_selected<W: Clone, E: core::error::Error, const N: usize>(
    machine: &mut Machine<'_, W, E>,
    multiplicand: &[W; N],
    multiplier: &[W; 32],
) -> Result<[W; N], ErtError<E>> {
    let mut accumulator = array::from_fn(|_| machine.zero.clone());
    let mut addend = multiplicand.clone();
    for multiplier_bit in multiplier {
        let sum = add_bits(machine.t, &accumulator, &addend, machine.zero.clone())
            .map_err(ErtError::Emitted)?;
        accumulator = select_word(machine, multiplier_bit.clone(), &sum, &accumulator)?;
        addend = shift_left_one(&addend, &machine.zero);
    }
    Ok(accumulator)
}

fn multiply_by_constant<W: Clone, E: core::error::Error, const N: usize>(
    machine: &mut Machine<'_, W, E>,
    multiplicand: &[W; N],
    multiplier: u32,
) -> Result<[W; N], ErtError<E>> {
    let mut accumulator = array::from_fn(|_| machine.zero.clone());
    let mut addend = multiplicand.clone();
    for bit in 0..32 {
        if multiplier & (1 << bit) != 0 {
            accumulator = add_bits(machine.t, &accumulator, &addend, machine.zero.clone())
                .map_err(ErtError::Emitted)?;
        }
        addend = shift_left_one(&addend, &machine.zero);
    }
    Ok(accumulator)
}

fn zero_extend<W: Clone>(word: &[W; 32], zero: &W) -> [W; 64] {
    array::from_fn(|bit| {
        if bit < 32 {
            word[bit].clone()
        } else {
            zero.clone()
        }
    })
}

fn shift_left_one<W: Clone, const N: usize>(word: &[W; N], zero: &W) -> [W; N] {
    array::from_fn(|bit| {
        if bit == 0 {
            zero.clone()
        } else {
            word[bit - 1].clone()
        }
    })
}

fn correct_high_product<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    mut high: [W; 32],
    left: &[W; 32],
    right: &[W; 32],
    left_constant: Option<u32>,
    right_constant: Option<u32>,
    product: Product,
) -> Result<[W; 32], ErtError<E>> {
    if matches!(product, Product::HighSigned | Product::HighSignedUnsigned) {
        high = subtract_if_negative(machine, high, left, right, left_constant)?;
    }
    if matches!(product, Product::HighSigned) {
        high = subtract_if_negative(machine, high, right, left, right_constant)?;
    }
    Ok(high)
}

fn subtract_if_negative<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    value: [W; 32],
    signed_operand: &[W; 32],
    subtrahend: &[W; 32],
    signed_constant: Option<u32>,
) -> Result<[W; 32], ErtError<E>> {
    match signed_constant {
        Some(constant) if constant >> 31 == 0 => Ok(value),
        Some(_) => subtract_word(machine, &value, subtrahend),
        None => {
            let difference = subtract_word(machine, &value, subtrahend)?;
            select_word(machine, signed_operand[31].clone(), &difference, &value)
        }
    }
}

fn subtract_word<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    minuend: &[W; 32],
    subtrahend: &[W; 32],
) -> Result<[W; 32], ErtError<E>> {
    let mut inverted = subtrahend.clone();
    for bit in &mut inverted {
        *bit = machine
            .t
            .bitxor(bit.clone(), machine.one.clone())
            .map_err(ErtError::Emitted)?;
    }
    add_bits(machine.t, minuend, &inverted, machine.one.clone()).map_err(ErtError::Emitted)
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
                    i8::from_le_bytes(machine.mem.read(address).ok_or(ErtError::Unexpected)?) as i32
                        as u32
                }
                LoadKind::HalfSigned => {
                    i16::from_le_bytes(machine.mem.read(address).ok_or(ErtError::Unexpected)?)
                        as i32 as u32
                }
                LoadKind::ByteUnsigned => {
                    u8::from_le_bytes(machine.mem.read(address).ok_or(ErtError::Unexpected)?) as u32
                }
                LoadKind::HalfUnsigned => {
                    u16::from_le_bytes(machine.mem.read(address).ok_or(ErtError::Unexpected)?)
                        as u32
                }
                LoadKind::Word => {
                    u32::from_le_bytes(machine.mem.read(address).ok_or(ErtError::Unexpected)?)
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
            let stack_bits = machine.stack_bits(offset, width)?;
            for bit in 0..32 {
                let source_bit = bit.min(width - 1);
                if !sign_extend && source_bit != bit {
                    machine.regs[dest.0 as usize][bit] = machine.zero.clone();
                    continue;
                }
                machine.regs[dest.0 as usize][bit] =
                    machine.read_stack_bit(stack_bits.start + source_bit)?;
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
    let stack_bits = machine.stack_bits(offset, width)?;
    for bit in 0..width {
        machine.write_stack_bit(
            stack_bits.start + bit,
            machine.regs[src.0 as usize][bit].clone(),
        )?;
    }
    next(machine)
}

fn jump_and_link<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    offset: Imm,
    dest: Reg,
) -> Result<Flow, ErtError<E>> {
    push_return(machine, dest)?;
    Ok(Flow::Next(machine.pc.wrapping_add_signed(offset.as_i32())))
}

fn jump_and_link_register<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    offset: Imm,
    base: Reg,
    dest: Reg,
) -> Result<Flow, ErtError<E>> {
    if dest == Reg::ZERO && base == Reg::RA && offset == Imm::ZERO {
        return return_from_call(machine);
    }
    let base = machine.reg_consts[base.0 as usize].ok_or(ErtError::Unexpected)?;
    let target = base.wrapping_add_signed(offset.as_i32()) & !1;
    push_return(machine, dest)?;
    if dest != Reg::ZERO {
        machine.write_constant(dest, machine.pc + 4);
    }
    Ok(Flow::Next(target))
}

fn push_return<W, E>(machine: &mut Machine<'_, W, E>, dest: Reg) -> Result<(), ErtError<E>> {
    if dest != Reg::ZERO {
        *machine
            .rstack
            .get_mut(machine.rsp as usize)
            .ok_or(ErtError::Unexpected)? = machine.pc + 4;
        machine.rsp += 1;
    }
    Ok(())
}

fn return_from_call<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
) -> Result<Flow, ErtError<E>> {
    machine.rsp = machine.rsp.checked_sub(1).ok_or(ErtError::Unexpected)?;
    Ok(Flow::Next(machine.rstack[machine.rsp as usize]))
}

fn branch<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    offset: Imm,
    src1: Reg,
    src2: Reg,
    condition: Branch,
) -> Result<Flow, ErtError<E>> {
    if let (Some(left), Some(right)) = (
        machine.reg_consts[src1.0 as usize],
        machine.reg_consts[src2.0 as usize],
    ) {
        let taken = match condition {
            Branch::Equal => left == right,
            Branch::NotEqual => left != right,
            Branch::GreaterEqualUnsigned => left >= right,
            Branch::LessThanUnsigned => left < right,
            Branch::GreaterEqualSigned => (left as i32) >= (right as i32),
            Branch::LessThanSigned => (left as i32) < (right as i32),
        };
        return Ok(Flow::Next(if taken {
            machine.pc.wrapping_add_signed(offset.as_i32())
        } else {
            machine.pc + 4
        }));
    }

    #[cfg(feature = "early-exit-loops")]
    if let Some(flow) = early_exit_loop_branch(machine, offset, src1, src2, condition)? {
        return Ok(flow);
    }

    Err(ErtError::Unexpected)
}

/// Attempt the opt-in "deoptimize secret-dependent early-exit loops"
/// recognizer (see `crate::early_exit`) before falling through to today's
/// hard error. Returns `Ok(None)` whenever the recognizer isn't enabled or
/// this branch doesn't match the narrow, provably-safe idiom it looks for.
#[cfg(feature = "early-exit-loops")]
fn early_exit_loop_branch<W: Clone, E: core::error::Error>(
    machine: &mut Machine<'_, W, E>,
    offset: Imm,
    src1: Reg,
    src2: Reg,
    condition: Branch,
) -> Result<Option<Flow>, ErtError<E>> {
    let options = machine.t.early_exit_loop_options();
    if !options.enabled {
        return Ok(None);
    }

    let branch_pc = machine.pc;
    let cached = machine
        .loop_sites
        .iter()
        .flatten()
        .find(|site| site.branch_pc == branch_pc)
        .copied();
    let site = match cached {
        Some(site) => site,
        None => {
            let predicate = match condition {
                Branch::Equal => cirrus_ert_core::ComparePredicate::Eq,
                Branch::NotEqual => cirrus_ert_core::ComparePredicate::Ne,
                Branch::GreaterEqualUnsigned => cirrus_ert_core::ComparePredicate::GeU,
                Branch::LessThanUnsigned => cirrus_ert_core::ComparePredicate::LtU,
                Branch::GreaterEqualSigned => cirrus_ert_core::ComparePredicate::GeS,
                Branch::LessThanSigned => cirrus_ert_core::ComparePredicate::LtS,
            };
            let Some(site) = crate::early_exit::recognize(
                &machine.mem,
                branch_pc,
                offset,
                predicate,
                options.max_lookahead_instructions,
            ) else {
                return Ok(None);
            };
            if let Some(slot) = machine.loop_sites.iter_mut().find(|slot| slot.is_none()) {
                *slot = Some(site);
            }
            site
        }
    };

    let should_take = cirrus_ert_core::compare_word(
        machine.t,
        &machine.regs[src1.0 as usize],
        &machine.regs[src2.0 as usize],
        site.predicate,
        &machine.one,
    )
    .map_err(ErtError::Emitted)?;
    let should_exit = if site.exit_when_taken {
        should_take
    } else {
        machine
            .t
            .bitxor(should_take, machine.one.clone())
            .map_err(ErtError::Emitted)?
    };
    for slot in site.exit_writes.iter().take(site.exit_write_count) {
        let (dest, value) = slot.expect("exit_write_count bounds the initialized prefix");
        let candidate = machine.word_from_constant(value);
        let current = machine.regs[dest.0 as usize].clone();
        machine.regs[dest.0 as usize] =
            select_word(machine, should_exit.clone(), &candidate, &current)?;
        machine.reg_consts[dest.0 as usize] = None;
        machine.offs[dest.0 as usize] = None;
    }
    Ok(Some(Flow::Next(site.continue_target)))
}
