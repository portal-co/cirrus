use core::array;

use cirrus_ert_core::{
    compare_word, fixed_shift, partial_bitwise_word, runtime_shift, BitOp, ComparePredicate, Shift,
};
use rv_asm::{Imm, Inst, Reg};

use crate::machine::{LoadAddress, Machine, RstackWord, add_bits, sext, word_mask};
use crate::{EcallOutcome, ErtError};

#[doc(hidden)]
pub enum Flow {
    Next(u64),
    Exit,
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
    /// `LW` on RV64: a 32-bit lane, sign-extended.
    WordSigned,
    ByteUnsigned,
    HalfUnsigned,
    /// `LWU` on RV64: a 32-bit lane, zero-extended.
    WordUnsigned,
    /// `LD` (RV64 only).
    Double,
}

impl LoadKind {
    fn width(self) -> usize {
        match self {
            LoadKind::ByteSigned | LoadKind::ByteUnsigned => 8,
            LoadKind::HalfSigned | LoadKind::HalfUnsigned => 16,
            LoadKind::WordSigned | LoadKind::WordUnsigned => 32,
            LoadKind::Double => 64,
        }
    }

    fn sign_extend(self) -> bool {
        matches!(
            self,
            LoadKind::ByteSigned | LoadKind::HalfSigned | LoadKind::WordSigned
        )
    }
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

/// Sign-extend the low 32 bits of `value` to 64 bits (the RV64 `*W`
/// instruction semantics; identity on RV32 results, which never reach this).
fn sext32(value: u64) -> u64 {
    value as u32 as i32 as i64 as u64
}

/// The RV64 `LUI`/`AUIPC` immediate handling: the 32-bit U-immediate is
/// sign-extended to 64 bits. RV32 keeps the plain zero-extended word.
fn uimmediate<const BITS: usize>(uimm: u32) -> u64 {
    if BITS == 64 {
        sext32(u64::from(uimm))
    } else {
        u64::from(uimm)
    }
}

/// The decoder accepts `*W` forms only on RV64; every W-form handler
/// debug-asserts that invariant.
fn w_low<W: Clone, const BITS: usize>(word: &[W; BITS]) -> [W; 32] {
    debug_assert_eq!(BITS, 64);
    array::from_fn(|bit| word[bit].clone())
}

/// Sign-extend a 32-bit result word into the full register width.
fn w_extend<W: Clone, const BITS: usize>(low: &[W; 32]) -> [W; BITS] {
    debug_assert_eq!(BITS, 64);
    array::from_fn(|bit| {
        if bit < 32 {
            low[bit].clone()
        } else {
            low[31].clone()
        }
    })
}

#[doc(hidden)]
pub fn execute<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    instruction: Inst,
) -> Result<Flow, ErtError<E>> {
    match instruction {
        Inst::Lui { uimm, dest } => {
            constant(machine, dest, uimmediate::<BITS>(uimm.as_u32()), machine.pc)
        }
        Inst::Auipc { uimm, dest } => constant(
            machine,
            dest,
            uimmediate::<BITS>(uimm.as_u32()).wrapping_add(machine.pc),
            machine.pc,
        ),

        Inst::Addi { imm, dest, src1 } => add_immediate(machine, imm, dest, src1),
        Inst::AddiW { imm, dest, src1 } => add_immediate_w(machine, imm, dest, src1),
        Inst::Slti { imm, dest, src1 } => {
            set_less_than_immediate(machine, dest, src1, imm, ComparePredicate::LtS)
        }
        Inst::Sltiu { imm, dest, src1 } => {
            set_less_than_immediate(machine, dest, src1, imm, ComparePredicate::LtU)
        }
        Inst::Add { dest, src1, src2 } => add(machine, dest, src1, src2),
        Inst::AddW { dest, src1, src2 } => add_w(machine, dest, src1, src2, false),
        Inst::Sub { dest, src1, src2 } => subtract(machine, dest, src1, src2),
        Inst::SubW { dest, src1, src2 } => add_w(machine, dest, src1, src2, true),
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
        Inst::SlliW { imm, dest, src1 } => shift_w(machine, imm, dest, src1, Shift::Left),
        Inst::Srli { imm, dest, src1 } => shift(machine, imm, dest, src1, Shift::LogicalRight),
        Inst::SrliW { imm, dest, src1 } => {
            shift_w(machine, imm, dest, src1, Shift::LogicalRight)
        }
        Inst::Srai { imm, dest, src1 } => {
            shift(machine, imm, dest, src1, Shift::ArithmeticRight)
        }
        Inst::SraiW { imm, dest, src1 } => {
            shift_w(machine, imm, dest, src1, Shift::ArithmeticRight)
        }
        Inst::Sll { dest, src1, src2 } => runtime_shift_h(machine, dest, src1, src2, Shift::Left),
        Inst::SllW { dest, src1, src2 } => {
            runtime_shift_w(machine, dest, src1, src2, Shift::Left)
        }
        Inst::Srl { dest, src1, src2 } => {
            runtime_shift_h(machine, dest, src1, src2, Shift::LogicalRight)
        }
        Inst::SrlW { dest, src1, src2 } => {
            runtime_shift_w(machine, dest, src1, src2, Shift::LogicalRight)
        }
        Inst::Sra { dest, src1, src2 } => {
            runtime_shift_h(machine, dest, src1, src2, Shift::ArithmeticRight)
        }
        Inst::SraW { dest, src1, src2 } => {
            runtime_shift_w(machine, dest, src1, src2, Shift::ArithmeticRight)
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
        Inst::Lw { offset, dest, base } => load(machine, offset, dest, base, LoadKind::WordSigned),
        Inst::Lbu { offset, dest, base } => {
            load(machine, offset, dest, base, LoadKind::ByteUnsigned)
        }
        Inst::Lhu { offset, dest, base } => {
            load(machine, offset, dest, base, LoadKind::HalfUnsigned)
        }
        Inst::Lwu { offset, dest, base } => {
            load(machine, offset, dest, base, LoadKind::WordUnsigned)
        }
        Inst::Ld { offset, dest, base } => load(machine, offset, dest, base, LoadKind::Double),
        Inst::Sb { offset, src, base } => store(machine, offset, src, base, 8),
        Inst::Sh { offset, src, base } => store(machine, offset, src, base, 16),
        Inst::Sw { offset, src, base } => store(machine, offset, src, base, 32),
        Inst::Sd { offset, src, base } => store(machine, offset, src, base, 64),

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

fn set_less_than<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
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
        let result = compare_concrete::<BITS>(left, right, predicate) as u64;
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

fn set_less_than_immediate<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    dest: Reg,
    src1: Reg,
    imm: Imm,
    predicate: ComparePredicate,
) -> Result<Flow, ErtError<E>> {
    let immediate = imm.as_i32() as i64 as u64;
    machine.offs[dest.0 as usize] = None;
    if let Some(left) = machine.reg_consts[src1.0 as usize] {
        machine.write_constant(dest, compare_concrete::<BITS>(left, immediate, predicate) as u64);
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

fn compare_concrete<const BITS: usize>(left: u64, right: u64, predicate: ComparePredicate) -> bool {
    match predicate {
        ComparePredicate::Eq => left == right,
        ComparePredicate::Ne => left != right,
        ComparePredicate::GeU => left >= right,
        ComparePredicate::LtU => left < right,
        ComparePredicate::GeS => sext::<BITS>(left) >= sext::<BITS>(right),
        ComparePredicate::LtS => sext::<BITS>(left) < sext::<BITS>(right),
    }
}

fn next<W, E, const BITS: usize, R: RstackWord>(machine: &Machine<'_, W, E, BITS, R>) -> Result<Flow, ErtError<E>> {
    Ok(Flow::Next(machine.pc + machine.inst_len))
}

fn constant<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    dest: Reg,
    value: u64,
    pc: u64,
) -> Result<Flow, ErtError<E>> {
    machine.write_constant(dest, value);
    Ok(Flow::Next(pc + machine.inst_len))
}

fn add_immediate<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    imm: Imm,
    dest: Reg,
    src1: Reg,
) -> Result<Flow, ErtError<E>> {
    if dest == Reg::SP {
        machine.sp = machine.sp.wrapping_add_signed(i64::from(imm.as_i32()));
        for offset in machine.offs.iter_mut().flatten() {
            *offset = offset.wrapping_sub(i64::from(imm.as_i32()));
        }
    }
    if src1 != dest {
        machine.offs[dest.0 as usize] = None;
    }
    if src1 == Reg::SP {
        machine.offs[dest.0 as usize] = Some(i64::from(imm.as_i32()));
    } else if let Some(offset) = machine.offs[src1.0 as usize] {
        machine.offs[dest.0 as usize] = Some(offset.wrapping_add(i64::from(imm.as_i32())));
    }

    if let Some(value) = machine.reg_consts[src1.0 as usize] {
        machine.write_constant(dest, value.wrapping_add_signed(i64::from(imm.as_i32())));
        return next(machine);
    }

    let word = machine.word_from_constant(imm.as_i32() as i64 as u64);
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

/// `ADDIW`: add the sign-extended 12-bit immediate to the low 32 bits of
/// `src1`, then sign-extend the 32-bit result (RV64 only).
fn add_immediate_w<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    imm: Imm,
    dest: Reg,
    src1: Reg,
) -> Result<Flow, ErtError<E>> {
    debug_assert_eq!(BITS, 64, "ADDIW decodes only on RV64");
    machine.offs[dest.0 as usize] = None;
    if let Some(value) = machine.reg_consts[src1.0 as usize] {
        let low = (value as u32).wrapping_add_signed(imm.as_i32());
        machine.write_constant(dest, sext32(u64::from(low)));
        return next(machine);
    }
    let immediate = machine.word_from_constant(imm.as_i32() as i64 as u64);
    let low = add_bits(
        machine.t,
        &w_low(&machine.regs[src1.0 as usize]),
        &w_low(&immediate),
        machine.zero.clone(),
    )
    .map_err(ErtError::Emitted)?;
    machine.regs[dest.0 as usize] = w_extend(&low);
    machine.reg_consts[dest.0 as usize] = None;
    next(machine)
}

fn add<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    dest: Reg,
    src1: Reg,
    src2: Reg,
) -> Result<Flow, ErtError<E>> {
    if src1 != dest && src2 != dest {
        machine.offs[dest.0 as usize] = None;
    }
    if let Some(constant) = machine.reg_consts[src2.0 as usize] {
        if src1 == Reg::SP {
            machine.offs[dest.0 as usize] = Some(constant as i64);
        } else if let Some(offset) = machine.offs[src1.0 as usize] {
            machine.offs[dest.0 as usize] = Some(offset.wrapping_add_unsigned(constant));
        }
    }
    if let Some(constant) = machine.reg_consts[src1.0 as usize] {
        if src2 == Reg::SP {
            machine.offs[dest.0 as usize] = Some(constant as i64);
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

/// `ADDW`/`SUBW`: 32-bit addition on the low halves, sign-extended (RV64
/// only). The result is never a stack-form address, so `offs` is cleared.
fn add_w<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    dest: Reg,
    src1: Reg,
    src2: Reg,
    subtract: bool,
) -> Result<Flow, ErtError<E>> {
    debug_assert_eq!(BITS, 64, "*W register forms decode only on RV64");
    machine.offs[dest.0 as usize] = None;

    if let (Some(a), Some(b)) = (
        machine.reg_consts[src1.0 as usize],
        machine.reg_consts[src2.0 as usize],
    ) {
        let low = if subtract {
            (a as u32).wrapping_sub(b as u32)
        } else {
            (a as u32).wrapping_add(b as u32)
        };
        machine.write_constant(dest, sext32(u64::from(low)));
        return next(machine);
    }

    let right = if subtract {
        let mut inverted = w_low(&machine.regs[src2.0 as usize]);
        for bit in &mut inverted {
            *bit = machine
                .t
                .bitxor(bit.clone(), machine.one.clone())
                .map_err(ErtError::Emitted)?;
        }
        inverted
    } else {
        w_low(&machine.regs[src2.0 as usize])
    };
    let carry = if subtract {
        machine.one.clone()
    } else {
        machine.zero.clone()
    };
    let low = add_bits(machine.t, &w_low(&machine.regs[src1.0 as usize]), &right, carry)
        .map_err(ErtError::Emitted)?;
    machine.regs[dest.0 as usize] = w_extend(&low);
    machine.reg_consts[dest.0 as usize] = None;
    next(machine)
}

fn subtract<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    dest: Reg,
    src1: Reg,
    src2: Reg,
) -> Result<Flow, ErtError<E>> {
    if src1 != dest && src2 != dest {
        machine.offs[dest.0 as usize] = None;
    }
    if let Some(constant) = machine.reg_consts[src2.0 as usize] {
        if src1 == Reg::SP {
            machine.offs[dest.0 as usize] = Some((constant as i64).wrapping_neg());
        } else if let Some(offset) = machine.offs[src1.0 as usize] {
            machine.offs[dest.0 as usize] = Some(offset.wrapping_sub_unsigned(constant));
        }
    }
    if let Some(constant) = machine.reg_consts[src1.0 as usize] {
        if src2 == Reg::SP {
            machine.offs[dest.0 as usize] = Some((constant as i64).wrapping_neg());
        } else if let Some(offset) = machine.offs[src2.0 as usize] {
            machine.offs[dest.0 as usize] = Some(offset.wrapping_sub_unsigned(constant));
        }
    }

    let concrete = match (machine.offs[src1.0 as usize], machine.offs[src2.0 as usize]) {
        (Some(a), Some(b)) => Some(a.wrapping_sub(b) as u64),
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

    // `src1 - src2` as `src1 + !src2 + 1`. (A historical revision inverted
    // `src1` instead — `src2 - src1` — which symbolic coverage now pins
    // against.)
    let mut right = machine.regs[src2.0 as usize].clone();
    for bit in &mut right {
        *bit = machine
            .t
            .bitxor(bit.clone(), machine.one.clone())
            .map_err(ErtError::Emitted)?;
    }
    machine.regs[dest.0 as usize] = add_bits(
        machine.t,
        &machine.regs[src1.0 as usize],
        &right,
        machine.one.clone(),
    )
    .map_err(ErtError::Emitted)?;
    machine.reg_consts[dest.0 as usize] = None;
    next(machine)
}

fn fold(operation: BitOp, a: u64, b: u64) -> u64 {
    match operation {
        BitOp::And => a & b,
        BitOp::Or => a | b,
        BitOp::Xor => a ^ b,
    }
}

fn degenerate<const BITS: usize>(operation: BitOp, constant: u64) -> bool {
    match operation {
        BitOp::And => constant & word_mask::<BITS>() == 0,
        BitOp::Or => constant & word_mask::<BITS>() == word_mask::<BITS>(),
        BitOp::Xor => false,
    }
}

fn degenerate_value<const BITS: usize>(operation: BitOp) -> u64 {
    match operation {
        BitOp::And => 0,
        BitOp::Or => word_mask::<BITS>(),
        BitOp::Xor => unreachable!("Xor is never degenerate"),
    }
}

fn bitwise<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
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
        if degenerate::<BITS>(operation, constant) {
            machine.write_constant(dest, degenerate_value::<BITS>(operation));
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
    for bit in 0..BITS {
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

fn bitwise_immediate<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    imm: Imm,
    dest: Reg,
    src1: Reg,
    operation: BitOp,
) -> Result<Flow, ErtError<E>> {
    let immediate = imm.as_i32() as i64 as u64;
    machine.offs[dest.0 as usize] = None;
    if let Some(value) = machine.reg_consts[src1.0 as usize] {
        machine.write_constant(dest, fold(operation, value, immediate));
        return next(machine);
    }
    if degenerate::<BITS>(operation, immediate) {
        machine.write_constant(dest, degenerate_value::<BITS>(operation));
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

fn concrete_shift<const BITS: usize>(value: u64, amount: u32, direction: Shift) -> u64 {
    let mask = word_mask::<BITS>();
    match direction {
        Shift::Left => (value << amount) & mask,
        Shift::LogicalRight => value >> amount,
        Shift::ArithmeticRight => ((sext::<BITS>(value) >> amount) as u64) & mask,
        Shift::RotateRight => unreachable!("RISC-V has no rotate form"),
    }
}

fn shift<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    imm: Imm,
    dest: Reg,
    src1: Reg,
    direction: Shift,
) -> Result<Flow, ErtError<E>> {
    let amount = imm.as_u32() & (BITS as u32 - 1);
    let source = machine.regs[src1.0 as usize].clone();
    machine.offs[dest.0 as usize] = None;
    machine.reg_consts[dest.0 as usize] = machine.reg_consts[src1.0 as usize]
        .map(|value| concrete_shift::<BITS>(value, amount, direction));
    machine.regs[dest.0 as usize] = fixed_shift(&source, amount, direction, &machine.zero);
    next(machine)
}

/// `SLLIW`/`SRLIW`/`SRAIW`: 32-bit fixed shift with a sign-extended result
/// (RV64 only).
fn shift_w<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    imm: Imm,
    dest: Reg,
    src1: Reg,
    direction: Shift,
) -> Result<Flow, ErtError<E>> {
    debug_assert_eq!(BITS, 64, "*W shift forms decode only on RV64");
    let amount = imm.as_u32() & 31;
    let source = machine.regs[src1.0 as usize].clone();
    machine.offs[dest.0 as usize] = None;
    machine.reg_consts[dest.0 as usize] =
        machine.reg_consts[src1.0 as usize].map(|value| {
            let low = value as u32;
            let result = match direction {
                Shift::Left => low << amount,
                Shift::LogicalRight => low >> amount,
                Shift::ArithmeticRight => (low as i32 >> amount) as u32,
                Shift::RotateRight => unreachable!("RISC-V has no rotate form"),
            };
            sext32(u64::from(result))
        });
    let shifted = fixed_shift(&w_low(&source), amount, direction, &machine.zero);
    machine.regs[dest.0 as usize] = w_extend(&shifted);
    next(machine)
}

fn runtime_shift_h<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    dest: Reg,
    src1: Reg,
    src2: Reg,
    direction: Shift,
) -> Result<Flow, ErtError<E>> {
    let source = machine.regs[src1.0 as usize].clone();
    let amount_word = machine.regs[src2.0 as usize].clone();
    machine.offs[dest.0 as usize] = None;

    if let Some(amount) = machine.reg_consts[src2.0 as usize] {
        let amount = (amount as u32) & (BITS as u32 - 1);
        machine.reg_consts[dest.0 as usize] = machine.reg_consts[src1.0 as usize]
            .map(|value| concrete_shift::<BITS>(value, amount, direction));
        machine.regs[dest.0 as usize] = fixed_shift(&source, amount, direction, &machine.zero);
        return next(machine);
    }

    machine.reg_consts[dest.0 as usize] = None;
    machine.regs[dest.0 as usize] = runtime_shift(
        machine.t,
        &source,
        &amount_word,
        direction,
        &machine.zero,
    )
    .map_err(ErtError::Emitted)?;
    next(machine)
}

/// `SLLW`/`SRLW`/`SRAW`: 32-bit runtime shift over `rs2[4:0]`, sign-extended
/// (RV64 only).
fn runtime_shift_w<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    dest: Reg,
    src1: Reg,
    src2: Reg,
    direction: Shift,
) -> Result<Flow, ErtError<E>> {
    debug_assert_eq!(BITS, 64, "*W shift forms decode only on RV64");
    let source = machine.regs[src1.0 as usize].clone();
    let amount_word = machine.regs[src2.0 as usize].clone();
    machine.offs[dest.0 as usize] = None;

    if let Some(amount) = machine.reg_consts[src2.0 as usize] {
        machine.reg_consts[dest.0 as usize] =
            machine.reg_consts[src1.0 as usize].map(|value| {
                let low = value as u32;
                let amount = (amount as u32) & 31;
                let result = match direction {
                    Shift::Left => low << amount,
                    Shift::LogicalRight => low >> amount,
                    Shift::ArithmeticRight => (low as i32 >> amount) as u32,
                    Shift::RotateRight => unreachable!("RISC-V has no rotate form"),
                };
                sext32(u64::from(result))
            });
        let shifted = fixed_shift(
            &w_low(&source),
            (amount as u32) & 31,
            direction,
            &machine.zero,
        );
        machine.regs[dest.0 as usize] = w_extend(&shifted);
        return next(machine);
    }

    machine.reg_consts[dest.0 as usize] = None;
    let shifted = runtime_shift(
        machine.t,
        &w_low(&source),
        &w_low(&amount_word),
        direction,
        &machine.zero,
    )
    .map_err(ErtError::Emitted)?;
    machine.regs[dest.0 as usize] = w_extend(&shifted);
    next(machine)
}

fn select_word<W: Clone, E: core::error::Error, const N: usize, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
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

fn multiply<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
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
        machine.write_constant(dest, concrete_product::<BITS>(product, left, right));
        return next(machine);
    }
    if left_constant == Some(0) || right_constant == Some(0) {
        machine.write_constant(dest, 0);
        return next(machine);
    }

    let result = match product {
        Product::Low => {
            let (low, _) = if let Some(constant) = left_constant {
                multiply_by_constant(machine, &right, constant)?
            } else if let Some(constant) = right_constant {
                multiply_by_constant(machine, &left, constant)?
            } else {
                multiply_full(machine, &left, &right)?
            };
            low
        }
        Product::HighSigned | Product::HighSignedUnsigned | Product::HighUnsigned => {
            let (_, high) = if let Some(constant) = left_constant {
                multiply_by_constant(machine, &right, constant)?
            } else if let Some(constant) = right_constant {
                multiply_by_constant(machine, &left, constant)?
            } else {
                multiply_full(machine, &left, &right)?
            };
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

fn concrete_product<const BITS: usize>(product: Product, left: u64, right: u64) -> u64 {
    let mask = word_mask::<BITS>();
    match product {
        Product::Low => left.wrapping_mul(right) & mask,
        Product::HighSigned => {
            ((sext::<BITS>(left) as i128 * sext::<BITS>(right) as i128) >> BITS) as u64 & mask
        }
        Product::HighSignedUnsigned => {
            ((sext::<BITS>(left) as i128 * (right as u128) as i128) >> BITS) as u64 & mask
        }
        Product::HighUnsigned => ((left as u128 * right as u128) >> BITS) as u64 & mask,
    }
}

/// Long-multiply `multiplicand` (zero-extended to double width) by
/// `multiplier`, returning the full product as separate low and high words.
/// Splitting the double-width accumulator into two `BITS`-wide words keeps
/// the routine const-generic without unstable generic const expressions.
fn multiply_full<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    multiplicand: &[W; BITS],
    multiplier: &[W; BITS],
) -> Result<([W; BITS], [W; BITS]), ErtError<E>> {
    let mut acc_low: [W; BITS] = array::from_fn(|_| machine.zero.clone());
    let mut acc_high: [W; BITS] = array::from_fn(|_| machine.zero.clone());
    let mut addend_low = multiplicand.clone();
    let mut addend_high: [W; BITS] = array::from_fn(|_| machine.zero.clone());
    for multiplier_bit in multiplier {
        let (sum_low, carry) = cirrus_ert_core::add_bits_with_carry_out(
            machine.t,
            &acc_low,
            &addend_low,
            machine.zero.clone(),
        )
        .map_err(ErtError::Emitted)?;
        let sum_high = add_bits(machine.t, &acc_high, &addend_high, carry)
            .map_err(ErtError::Emitted)?;
        acc_low = select_word(machine, multiplier_bit.clone(), &sum_low, &acc_low)?;
        acc_high = select_word(machine, multiplier_bit.clone(), &sum_high, &acc_high)?;
        let top = addend_low[BITS - 1].clone();
        addend_low = shift_left_one(&addend_low, &machine.zero);
        addend_high = shift_left_one(&addend_high, &machine.zero);
        addend_high[0] = top;
    }
    Ok((acc_low, acc_high))
}

fn multiply_by_constant<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    multiplicand: &[W; BITS],
    multiplier: u64,
) -> Result<([W; BITS], [W; BITS]), ErtError<E>> {
    let mut acc_low: [W; BITS] = array::from_fn(|_| machine.zero.clone());
    let mut acc_high: [W; BITS] = array::from_fn(|_| machine.zero.clone());
    let mut addend_low = multiplicand.clone();
    let mut addend_high: [W; BITS] = array::from_fn(|_| machine.zero.clone());
    for bit in 0..BITS {
        if multiplier & (1 << bit) != 0 {
            let (sum_low, carry) = cirrus_ert_core::add_bits_with_carry_out(
                machine.t,
                &acc_low,
                &addend_low,
                machine.zero.clone(),
            )
            .map_err(ErtError::Emitted)?;
            acc_low = sum_low;
            acc_high = add_bits(machine.t, &acc_high, &addend_high, carry)
                .map_err(ErtError::Emitted)?;
        }
        let top = addend_low[BITS - 1].clone();
        addend_low = shift_left_one(&addend_low, &machine.zero);
        addend_high = shift_left_one(&addend_high, &machine.zero);
        addend_high[0] = top;
    }
    Ok((acc_low, acc_high))
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

fn correct_high_product<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    mut high: [W; BITS],
    left: &[W; BITS],
    right: &[W; BITS],
    left_constant: Option<u64>,
    right_constant: Option<u64>,
    product: Product,
) -> Result<[W; BITS], ErtError<E>> {
    if matches!(product, Product::HighSigned | Product::HighSignedUnsigned) {
        high = subtract_if_negative(machine, high, left, right, left_constant)?;
    }
    if matches!(product, Product::HighSigned) {
        high = subtract_if_negative(machine, high, right, left, right_constant)?;
    }
    Ok(high)
}

fn subtract_if_negative<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    value: [W; BITS],
    signed_operand: &[W; BITS],
    subtrahend: &[W; BITS],
    signed_constant: Option<u64>,
) -> Result<[W; BITS], ErtError<E>> {
    match signed_constant {
        Some(constant) if constant >> (BITS - 1) == 0 => Ok(value),
        Some(_) => subtract_word(machine, &value, subtrahend),
        None => {
            let difference = subtract_word(machine, &value, subtrahend)?;
            select_word(machine, signed_operand[BITS - 1].clone(), &difference, &value)
        }
    }
}

fn subtract_word<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    minuend: &[W; BITS],
    subtrahend: &[W; BITS],
) -> Result<[W; BITS], ErtError<E>> {
    let mut inverted = subtrahend.clone();
    for bit in &mut inverted {
        *bit = machine
            .t
            .bitxor(bit.clone(), machine.one.clone())
            .map_err(ErtError::Emitted)?;
    }
    add_bits(machine.t, minuend, &inverted, machine.one.clone()).map_err(ErtError::Emitted)
}

fn load<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    offset: Imm,
    dest: Reg,
    base: Reg,
    kind: LoadKind,
) -> Result<Flow, ErtError<E>> {
    match machine.load_address(base, offset)? {
        LoadAddress::Concrete(address) => {
            let value = match kind {
                LoadKind::ByteSigned => machine
                    .mem
                    .read64::<1>(address)
                    .map(|bytes| i8::from_le_bytes(bytes) as i64 as u64),
                LoadKind::HalfSigned => machine
                    .mem
                    .read64::<2>(address)
                    .map(|bytes| i16::from_le_bytes(bytes) as i64 as u64),
                LoadKind::WordSigned => machine
                    .mem
                    .read64::<4>(address)
                    .map(|bytes| i32::from_le_bytes(bytes) as i64 as u64),
                LoadKind::ByteUnsigned => machine
                    .mem
                    .read64::<1>(address)
                    .map(|bytes| u64::from(u8::from_le_bytes(bytes))),
                LoadKind::HalfUnsigned => machine
                    .mem
                    .read64::<2>(address)
                    .map(|bytes| u64::from(u16::from_le_bytes(bytes))),
                LoadKind::WordUnsigned => machine
                    .mem
                    .read64::<4>(address)
                    .map(|bytes| u64::from(u32::from_le_bytes(bytes))),
                LoadKind::Double => machine.mem.read64::<8>(address).map(u64::from_le_bytes),
            }
            .ok_or(ErtError::Unexpected)?;
            machine.write_constant(dest, value);
        }
        LoadAddress::Stack(offset) => {
            machine.offs[dest.0 as usize] = None;
            machine.reg_consts[dest.0 as usize] = None;
            let width = kind.width();
            if width > BITS {
                return Err(ErtError::Unexpected);
            }
            let sign_extend = kind.sign_extend();
            let stack_bits = machine.stack_bits(offset, width)?;
            for bit in 0..BITS {
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

fn store<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    offset: Imm,
    src: Reg,
    base: Reg,
    width: usize,
) -> Result<Flow, ErtError<E>> {
    if width > BITS {
        return Err(ErtError::Unexpected);
    }
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

fn jump_and_link<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    offset: Imm,
    dest: Reg,
) -> Result<Flow, ErtError<E>> {
    let target = machine.pc.wrapping_add_signed(i64::from(offset.as_i32()));
    let action = machine
        .t
        .call_hook(
            crate::CallEvent::Jal {
                caller_pc: machine.pc,
                target,
                link: dest,
            },
            &mut machine.regs[..],
            &mut machine.reg_consts[..],
            &mut machine.offs[..],
            &machine.zero,
            &machine.one,
        )
        .map_err(ErtError::Emitted)?;
    match action {
        crate::CallAction::ReturnNow => {
            // The hook replaced the call: no return push, no link write; the
            // caller continues at the following instruction.
            next(machine)
        }
        crate::CallAction::Proceed | crate::CallAction::Divert(_) => {
            push_return(machine, dest)?;
            if dest != Reg::ZERO {
                machine.write_constant(dest, machine.pc + machine.inst_len);
            }
            let target = match action {
                crate::CallAction::Divert(new_target) => new_target,
                _ => target,
            };
            Ok(Flow::Next(target))
        }
    }
}

fn jump_and_link_register<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    offset: Imm,
    base: Reg,
    dest: Reg,
) -> Result<Flow, ErtError<E>> {
    if dest == Reg::ZERO && base == Reg::RA && offset == Imm::ZERO {
        // The conventional return. The hook observes the return before the
        // private stack is popped; an empty stack still fails closed without
        // consulting the hook, since there is no meaningful target to show.
        if machine.rsp == 0 {
            return return_from_call(machine);
        }
        let target = machine.rstack[(machine.rsp - 1) as usize].into_u64();
        let action = machine
            .t
            .call_hook(
                crate::CallEvent::Return {
                    from_pc: machine.pc,
                    target,
                },
                &mut machine.regs[..],
                &mut machine.reg_consts[..],
                &mut machine.offs[..],
                &machine.zero,
                &machine.one,
            )
            .map_err(ErtError::Emitted)?;
        match action {
            crate::CallAction::Proceed => return return_from_call(machine),
            crate::CallAction::Divert(new_target) => {
                machine.rsp -= 1;
                return Ok(Flow::Next(new_target));
            }
            crate::CallAction::ReturnNow => return Err(ErtError::Unexpected),
        }
    }
    let offset64 = i64::from(offset.as_i32());
    match machine.reg_consts[base.0 as usize] {
        Some(base_value) => {
            let target = base_value.wrapping_add_signed(offset64) & !1;
            let action = machine
                .t
                .call_hook(
                    crate::CallEvent::Jalr {
                        caller_pc: machine.pc,
                        target,
                        base,
                        offset: offset64,
                        link: dest,
                    },
                    &mut machine.regs[..],
                    &mut machine.reg_consts[..],
                    &mut machine.offs[..],
                    &machine.zero,
                    &machine.one,
                )
                .map_err(ErtError::Emitted)?;
            match action {
                crate::CallAction::ReturnNow => next(machine),
                crate::CallAction::Proceed | crate::CallAction::Divert(_) => {
                    push_return(machine, dest)?;
                    if dest != Reg::ZERO {
                        machine.write_constant(dest, machine.pc + machine.inst_len);
                    }
                    let target = match action {
                        crate::CallAction::Divert(new_target) => new_target,
                        _ => target,
                    };
                    Ok(Flow::Next(target))
                }
            }
        }
        None => {
            // The historical behavior fails closed here. The hook is
            // consulted first so a host can resolve a known-indirect target.
            let action = machine
                .t
                .call_hook(
                    crate::CallEvent::UnresolvedJalr {
                        caller_pc: machine.pc,
                        base,
                        offset: offset64,
                        link: dest,
                    },
                    &mut machine.regs[..],
                    &mut machine.reg_consts[..],
                    &mut machine.offs[..],
                    &machine.zero,
                    &machine.one,
                )
                .map_err(ErtError::Emitted)?;
            match action {
                crate::CallAction::Proceed => Err(ErtError::Unexpected),
                crate::CallAction::ReturnNow => next(machine),
                crate::CallAction::Divert(new_target) => {
                    push_return(machine, dest)?;
                    if dest != Reg::ZERO {
                        machine.write_constant(dest, machine.pc + machine.inst_len);
                    }
                    Ok(Flow::Next(new_target))
                }
            }
        }
    }
}

fn push_return<W, E, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
    dest: Reg,
) -> Result<(), ErtError<E>> {
    if dest != Reg::ZERO {
        *machine
            .rstack
            .get_mut(machine.rsp as usize)
            .ok_or(ErtError::Unexpected)? = R::from_u64(machine.pc + machine.inst_len);
        machine.rsp += 1;
    }
    Ok(())
}

fn return_from_call<W, E, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
) -> Result<Flow, ErtError<E>> {
    machine.rsp = machine.rsp.checked_sub(1).ok_or(ErtError::Unexpected)?;
    Ok(Flow::Next(machine.rstack[machine.rsp as usize].into_u64()))
}

fn branch<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
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
            Branch::GreaterEqualSigned => sext::<BITS>(left) >= sext::<BITS>(right),
            Branch::LessThanSigned => sext::<BITS>(left) < sext::<BITS>(right),
        };
        return Ok(Flow::Next(if taken {
            machine.pc.wrapping_add_signed(i64::from(offset.as_i32()))
        } else {
            machine.pc + machine.inst_len
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
fn early_exit_loop_branch<W: Clone, E: core::error::Error, const BITS: usize, R: RstackWord>(
    machine: &mut Machine<'_, W, E, BITS, R>,
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
                if BITS == 64 {
                    rv_asm::Xlen::Rv64
                } else {
                    rv_asm::Xlen::Rv32
                },
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
