#![no_std]
#![warn(missing_docs)]

//! A deliberately narrow, fail-closed AArch64 ERT decoder.
//!
//! [`disarm64`] is used only as a classification guard. Every instruction
//! accepted here must also satisfy an ERT-owned raw mask and field extractor;
//! a successful broad decoder result never expands the supported ISA by
//! itself. The Phase 3 foundation contains audited control flow, move-wide
//! constants, and immediate arithmetic/NZCV semantics; unlisted arithmetic
//! and all memory forms remain fail-closed.
//!
//! Instruction field definitions follow Arm A-profile Architecture Reference
//! Manual DDI0487 A64 encoding tables: B/BL (`C4.1.4`), B.cond (`C4.1.5`),
//! CBZ/CBNZ (`C4.1.6`), TBZ/TBNZ (`C4.1.7`), BR/BLR/RET (`C4.1.8`), and SVC
//! (`C4.1.9`). The raw masks below are the support authority; `disarm64`
//! protects this hand-extracted subset against accepting an undecodable word.

use cirrus_core::ContextWithValue;
use cirrus_ert_core::{
    BitOp, ContextWithErtOps, RawMemory, Shift, add_bits, add_bits_with_carry_out, add_overflow,
    arm_condition, arm_condition_value, bitwise_word, constant_word, fixed_shift, invert_word,
    select_word, subtract_overflow, zero_word,
};
use disarm64::decoder;

/// A symbolic A64 state with 31 GPRs and separate SP.
///
/// Encoding register 31 is resolved by each semantic form: it is XZR in the
/// supported control forms and never aliases `sp` here. The concrete metadata
/// is intentionally retained beside every symbolic word for fail-closed
/// branch/address decisions.
#[derive(Clone)]
pub struct State<W> {
    /// `x0` through `x30`; x31 is not representable here.
    pub regs: [[W; 64]; 31],
    /// Known concrete values for `x0` through `x30`.
    pub constants: [Option<u64>; 31],
    /// Architectural stack pointer, distinct from x31/XZR.
    pub sp: u64,
    /// NZCV flag wires in N, Z, C, V order.
    pub nzcv: [W; 4],
    /// Concrete NZCV facts in N, Z, C, V order, when known.
    pub nzcv_constants: [Option<bool>; 4],
    /// Symbolic done wire accumulated by [`step`].
    pub done: W,
}

/// A symbolic A64 control-flow result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Flow<W> {
    /// Continue at this next virtual instruction pointer.
    Next([W; 64]),
    /// The bare-metal ERT `SVC #0` exit completed.
    Exit,
}

/// Construct the initial state with all GPRs and done set to `zero`.
pub fn initial_state<W: Clone>(zero: W) -> State<W> {
    State {
        regs: core::array::from_fn(|_| core::array::from_fn(|_| zero.clone())),
        constants: [None; 31],
        sp: 0,
        nzcv: core::array::from_fn(|_| zero.clone()),
        nzcv_constants: [Some(false); 4],
        done: zero,
    }
}

/// Execute the audited Phase 3 subset over a symbolic state.
///
/// Direct control uses constant targets; conditional forms emit a 64-bit
/// virtual-IP select. `CBZ`/`CBNZ` only reads `x0..x30`; `x31` remains XZR.
/// Unlisted data-processing and every memory form fail closed.
pub fn step<C, W>(
    context: &mut C,
    state: &mut State<W>,
    pc: u64,
    raw: u32,
    zero: &W,
    one: &W,
) -> Result<Flow<W>, DecodeError>
where
    C: ContextWithErtOps<bool, Wrapped = W> + ContextWithValue<bool, Wrapped = W>,
    W: Clone,
{
    let instruction = decode(pc, raw)?;
    let next = pc.wrapping_add(4);
    let constant = |value| constant_word(zero, one, value);
    match instruction {
        Instruction::Address { dest, target } => {
            state.regs[dest as usize] = constant(target);
            state.constants[dest as usize] = Some(target);
            Ok(Flow::Next(constant(next)))
        }
        Instruction::Branch { target } => Ok(Flow::Next(constant(target))),
        Instruction::BranchLink { target } => {
            state.regs[30] = constant(next);
            state.constants[30] = Some(next);
            Ok(Flow::Next(constant(target)))
        }
        Instruction::ConditionalBranch { condition, target } => {
            let condition = arm_condition(
                context,
                state.nzcv[0].clone(),
                state.nzcv[1].clone(),
                state.nzcv[2].clone(),
                state.nzcv[3].clone(),
                condition,
                one,
            )
            .map_err(|_| DecodeError::Unsupported(raw))?
            .ok_or(DecodeError::Malformed(raw))?;
            let target_word = constant(target);
            let fallthrough_word = constant(next);
            Ok(Flow::Next(
                select_word(context, condition, &target_word, &fallthrough_word)
                    .map_err(|_| DecodeError::Unsupported(raw))?,
            ))
        }
        Instruction::CompareBranch {
            register,
            nonzero,
            target,
            ..
        } => {
            if register == 31 {
                // x31 is XZR for CBZ/CBNZ, never SP: CBZ is therefore
                // always taken and CBNZ always falls through.
                return Ok(Flow::Next(constant(if nonzero { next } else { target })));
            }
            let word = &state.regs[register as usize];
            let is_zero = cirrus_ert_core::zero_word(context, word, one)
                .map_err(|_| DecodeError::Unsupported(raw))?;
            let condition = if nonzero {
                context
                    .bitxor(is_zero, one.clone())
                    .map_err(|_| DecodeError::Unsupported(raw))?
            } else {
                is_zero
            };
            let target_word = constant(target);
            let fallthrough_word = constant(next);
            Ok(Flow::Next(
                select_word(context, condition, &target_word, &fallthrough_word)
                    .map_err(|_| DecodeError::Unsupported(raw))?,
            ))
        }
        Instruction::MoveWide {
            dest,
            immediate,
            shift,
            keep,
            width64,
        } => {
            let mask = if width64 {
                u64::MAX
            } else {
                u64::from(u32::MAX)
            };
            let field_mask = 0xffffu64 << shift;
            let inserted = immediate << shift;
            let result = if keep {
                let previous = &state.regs[dest as usize];
                let mut output = core::array::from_fn(|_| zero.clone());
                for bit in 0..64 {
                    let retained = if (field_mask >> bit) & 1 == 0 {
                        previous[bit].clone()
                    } else {
                        zero.clone()
                    };
                    output[bit] = if (inserted >> bit) & 1 != 0 {
                        one.clone()
                    } else {
                        retained
                    };
                }
                output
            } else {
                constant(inserted & mask)
            };
            state.regs[dest as usize] = result;
            state.constants[dest as usize] = if keep {
                state.constants[dest as usize]
                    .map(|previous| ((previous & !field_mask) | inserted) & mask)
            } else {
                Some(inserted & mask)
            };
            Ok(Flow::Next(constant(next)))
        }
        Instruction::AddImmediate {
            dest,
            source,
            immediate,
            subtract,
            set_flags,
            width64,
        } => {
            let source_word = state.regs[source as usize].clone();
            let immediate_word = constant(immediate);
            let (result, carry_out) = arithmetic(
                context,
                &source_word,
                &immediate_word,
                subtract,
                width64,
                zero,
                one,
            )?;
            let source_constant = state.constants[source as usize];
            let result_constant = source_constant.map(|source| {
                let result = if subtract {
                    source.wrapping_sub(immediate)
                } else {
                    source.wrapping_add(immediate)
                };
                if width64 {
                    result
                } else {
                    result & u64::from(u32::MAX)
                }
            });
            if set_flags {
                update_nzcv(
                    context,
                    state,
                    &source_word,
                    &immediate_word,
                    &result,
                    carry_out,
                    subtract,
                    width64,
                    source_constant,
                    result_constant,
                    one,
                )?;
            }
            if dest != 31 {
                state.regs[dest as usize] = result;
                state.constants[dest as usize] = result_constant;
            }
            Ok(Flow::Next(constant(next)))
        }
        Instruction::MultiplyAdd {
            dest,
            left,
            right,
            addend,
            subtract,
            width64,
        } => {
            let product = multiply_low(
                context,
                &register_word(state, left, zero),
                &register_word(state, right, zero),
                width64,
                zero,
            )?;
            let addend_word = register_word(state, addend, zero);
            let (result, _) = arithmetic(
                context,
                &addend_word,
                &product,
                subtract,
                width64,
                zero,
                one,
            )?;
            let mask = if width64 {
                u64::MAX
            } else {
                u64::from(u32::MAX)
            };
            let result_constant = register_constant(state, left)
                .zip(register_constant(state, right))
                .zip(register_constant(state, addend))
                .map(|((left, right), addend)| {
                    let product = left.wrapping_mul(right) & mask;
                    (if subtract {
                        addend.wrapping_sub(product)
                    } else {
                        addend.wrapping_add(product)
                    }) & mask
                });
            if dest != 31 {
                state.regs[dest as usize] = result;
                state.constants[dest as usize] = result_constant;
            }
            Ok(Flow::Next(constant(next)))
        }
        Instruction::ConditionalSelect {
            dest,
            when_true,
            when_false,
            condition,
            width64,
        } => {
            let condition_wire = arm_condition(
                context,
                state.nzcv[0].clone(),
                state.nzcv[1].clone(),
                state.nzcv[2].clone(),
                state.nzcv[3].clone(),
                condition,
                one,
            )
            .map_err(|_| DecodeError::Unsupported(raw))?
            .ok_or(DecodeError::Malformed(raw))?;
            let selected = select_word(
                context,
                condition_wire,
                &register_word(state, when_true, zero),
                &register_word(state, when_false, zero),
            )
            .map_err(|_| DecodeError::Unsupported(raw))?;
            let selected = if width64 {
                selected
            } else {
                core::array::from_fn(|bit| {
                    if bit < 32 {
                        selected[bit].clone()
                    } else {
                        zero.clone()
                    }
                })
            };
            if dest != 31 {
                state.regs[dest as usize] = selected;
                state.constants[dest as usize] = arm_condition_value(
                    state.nzcv_constants[0],
                    state.nzcv_constants[1],
                    state.nzcv_constants[2],
                    state.nzcv_constants[3],
                    condition,
                )
                .ok_or(DecodeError::Malformed(raw))?
                .and_then(|condition| {
                    register_constant(state, if condition { when_true } else { when_false })
                })
                .map(|value| {
                    if width64 {
                        value
                    } else {
                        value & u64::from(u32::MAX)
                    }
                });
            }
            Ok(Flow::Next(constant(next)))
        }
        Instruction::LogicalRegister {
            dest,
            left,
            right,
            shift,
            amount,
            operation,
            set_flags,
            width64,
        } => {
            let left_word = register_word(state, left, zero);
            let right_word = shift_register(state, right, amount, shift, width64, zero);
            let result = bitwise_word(context, &left_word, &right_word, operation)
                .map_err(|_| DecodeError::Unsupported(raw))?;
            let result = if width64 {
                result
            } else {
                core::array::from_fn(|bit| {
                    if bit < 32 {
                        result[bit].clone()
                    } else {
                        zero.clone()
                    }
                })
            };
            let left_constant = register_constant(state, left);
            let right_constant = register_constant(state, right)
                .map(|value| shift_constant(value, amount, shift, width64));
            let mask = if width64 {
                u64::MAX
            } else {
                u64::from(u32::MAX)
            };
            let result_constant = left_constant.zip(right_constant).map(|(left, right)| match operation {
                BitOp::And => left & right,
                BitOp::Or => left | right,
                BitOp::Xor => left ^ right,
            } & mask);
            if set_flags {
                let z = if width64 {
                    zero_word(context, &result, one)
                } else {
                    let result32: [W; 32] = core::array::from_fn(|bit| result[bit].clone());
                    zero_word(context, &result32, one)
                }
                .map_err(|_| DecodeError::Unsupported(raw))?;
                state.nzcv = [
                    result[if width64 { 63 } else { 31 }].clone(),
                    z,
                    zero.clone(),
                    zero.clone(),
                ];
                state.nzcv_constants = result_constant.map_or([None; 4], |result| {
                    [
                        Some((result >> if width64 { 63 } else { 31 }) & 1 != 0),
                        Some(result == 0),
                        Some(false),
                        Some(false),
                    ]
                });
            }
            if dest != 31 {
                state.regs[dest as usize] = result;
                state.constants[dest as usize] = result_constant;
            }
            Ok(Flow::Next(constant(next)))
        }
        Instruction::AddRegister {
            dest,
            left,
            right,
            shift,
            amount,
            subtract,
            set_flags,
            width64,
        } => {
            let left_word = state.regs[left as usize].clone();
            let right_word = if width64 {
                fixed_shift(&state.regs[right as usize], amount, shift, zero)
            } else {
                let right32: [W; 32] =
                    core::array::from_fn(|bit| state.regs[right as usize][bit].clone());
                let shifted = fixed_shift(&right32, amount, shift, zero);
                core::array::from_fn(|bit| {
                    if bit < 32 {
                        shifted[bit].clone()
                    } else {
                        zero.clone()
                    }
                })
            };
            let (result, carry_out) = arithmetic(
                context,
                &left_word,
                &right_word,
                subtract,
                width64,
                zero,
                one,
            )?;
            let left_constant = state.constants[left as usize];
            let right_constant = state.constants[right as usize]
                .map(|value| shift_constant(value, amount, shift, width64));
            let result_constant = left_constant.zip(right_constant).map(|(left, right)| {
                let result = if subtract {
                    left.wrapping_sub(right)
                } else {
                    left.wrapping_add(right)
                };
                if width64 {
                    result
                } else {
                    result & u64::from(u32::MAX)
                }
            });
            if set_flags {
                update_nzcv(
                    context,
                    state,
                    &left_word,
                    &right_word,
                    &result,
                    carry_out,
                    subtract,
                    width64,
                    left_constant,
                    result_constant,
                    one,
                )?;
            }
            if dest != 31 {
                state.regs[dest as usize] = result;
                state.constants[dest as usize] = result_constant;
            }
            Ok(Flow::Next(constant(next)))
        }
        Instruction::TestBranch {
            register,
            bit,
            nonzero,
            target,
        } => {
            let tested = if register == 31 {
                zero.clone()
            } else {
                state.regs[register as usize][bit as usize].clone()
            };
            let condition = if nonzero {
                tested
            } else {
                context
                    .bitxor(tested, one.clone())
                    .map_err(|_| DecodeError::Unsupported(raw))?
            };
            let target_word = constant(target);
            let fallthrough_word = constant(next);
            Ok(Flow::Next(
                select_word(context, condition, &target_word, &fallthrough_word)
                    .map_err(|_| DecodeError::Unsupported(raw))?,
            ))
        }
        Instruction::BranchRegister { register } => {
            let target = state.constants[register as usize].ok_or(DecodeError::Unsupported(raw))?;
            if target & 3 != 0 {
                return Err(DecodeError::Malformed(raw));
            }
            Ok(Flow::Next(constant(target)))
        }
        Instruction::BranchLinkRegister { register } => {
            let target = state.constants[register as usize].ok_or(DecodeError::Unsupported(raw))?;
            if target & 3 != 0 {
                return Err(DecodeError::Malformed(raw));
            }
            state.regs[30] = constant(next);
            state.constants[30] = Some(next);
            Ok(Flow::Next(constant(target)))
        }
        Instruction::Return => {
            let target = state.constants[30].ok_or(DecodeError::Unsupported(raw))?;
            if target & 3 != 0 {
                return Err(DecodeError::Malformed(raw));
            }
            Ok(Flow::Next(constant(target)))
        }
        Instruction::LoadLiteral { .. }
        | Instruction::Load { .. }
        | Instruction::LoadExtend { .. } => Err(DecodeError::Unsupported(raw)),
        Instruction::SupervisorCall => {
            // The façade is bare-metal, not a Linux syscall emulator. Hash
            // service selectors need the runtime/handler seam; until that is
            // installed, accept only the declared all-ones exit selector.
            if state.constants[0] != Some(u64::MAX) {
                return Err(DecodeError::Unsupported(raw));
            }
            state.done = one.clone();
            Ok(Flow::Exit)
        }
    }
}

/// Execute an audited literal load, or delegate a non-memory form to [`step`].
///
/// Literal loads have a PC-derived concrete address and therefore need no
/// symbolic storage interface. Other load/store addressing modes remain
/// rejected until the caller-owned storage seam is introduced.
pub fn step_with_memory<C, W>(
    context: &mut C,
    state: &mut State<W>,
    pc: u64,
    raw: u32,
    memory: RawMemory<'_>,
    zero: &W,
    one: &W,
) -> Result<Flow<W>, DecodeError>
where
    C: ContextWithErtOps<bool, Wrapped = W> + ContextWithValue<bool, Wrapped = W>,
    W: Clone,
{
    match decode(pc, raw)? {
        Instruction::LoadLiteral {
            dest,
            width,
            signed,
            target,
        } => load_literal(state, dest, width, signed, target, memory, zero, one)?,
        Instruction::Load {
            dest,
            base,
            offset,
            width,
            signed,
        } => {
            let base = register_constant(state, base).ok_or(DecodeError::Unsupported(raw))?;
            let target = base.wrapping_add(offset);
            load_literal(state, dest, width, signed, target, memory, zero, one)?;
        }
        Instruction::LoadExtend {
            dest,
            base,
            offset,
            width,
            signed,
        } => {
            let base = register_constant(state, base).ok_or(DecodeError::Unsupported(raw))?;
            let target = base.wrapping_add_signed(offset);
            load_literal(state, dest, width, signed, target, memory, zero, one)?;
        }
        _ => return step(context, state, pc, raw, zero, one),
    }
    Ok(Flow::Next(constant_word(zero, one, pc.wrapping_add(4))))
}

#[allow(clippy::too_many_arguments)]
fn load_literal<W: Clone>(
    state: &mut State<W>,
    dest: u8,
    width: u8,
    signed: bool,
    target: u64,
    memory: RawMemory<'_>,
    zero: &W,
    one: &W,
) -> Result<(), DecodeError> {
    let value = match (width, signed) {
        (1, false) => memory
            .read64::<1>(target)
            .map(|bytes| u64::from(u8::from_le_bytes(bytes))),
        (2, false) => memory
            .read64::<2>(target)
            .map(|bytes| u64::from(u16::from_le_bytes(bytes))),
        (4, false) => memory
            .read64::<4>(target)
            .map(|bytes| u64::from(u32::from_le_bytes(bytes))),
        (8, false) => memory.read64::<8>(target).map(u64::from_le_bytes),
        (1, true) => memory
            .read64::<1>(target)
            .map(|bytes| i64::from(i8::from_le_bytes(bytes)) as u64),
        (2, true) => memory
            .read64::<2>(target)
            .map(|bytes| i64::from(i16::from_le_bytes(bytes)) as u64),
        (4, true) => memory
            .read64::<4>(target)
            .map(|bytes| i64::from(i32::from_le_bytes(bytes)) as u64),
        _ => return Err(DecodeError::Unsupported(0)),
    }
    .ok_or(DecodeError::Memory(target))?;
    if dest != 31 {
        state.regs[dest as usize] = constant_word(zero, one, value);
        state.constants[dest as usize] = Some(value);
    }
    Ok(())
}

fn arithmetic<C, W>(
    context: &mut C,
    left: &[W; 64],
    right: &[W; 64],
    subtract: bool,
    width64: bool,
    zero: &W,
    one: &W,
) -> Result<([W; 64], W), DecodeError>
where
    C: ContextWithErtOps<bool, Wrapped = W>,
    W: Clone,
{
    if width64 {
        let right = if subtract {
            invert_word(context, right, one.clone()).map_err(|_| DecodeError::Unsupported(0))?
        } else {
            right.clone()
        };
        add_bits_with_carry_out(
            context,
            left,
            &right,
            if subtract { one.clone() } else { zero.clone() },
        )
        .map_err(|_| DecodeError::Unsupported(0))
    } else {
        let left32: [W; 32] = core::array::from_fn(|bit| left[bit].clone());
        let right32: [W; 32] = core::array::from_fn(|bit| right[bit].clone());
        let right32 = if subtract {
            invert_word(context, &right32, one.clone()).map_err(|_| DecodeError::Unsupported(0))?
        } else {
            right32
        };
        let (low, carry) = add_bits_with_carry_out(
            context,
            &left32,
            &right32,
            if subtract { one.clone() } else { zero.clone() },
        )
        .map_err(|_| DecodeError::Unsupported(0))?;
        Ok((
            core::array::from_fn(|bit| {
                if bit < 32 {
                    low[bit].clone()
                } else {
                    zero.clone()
                }
            }),
            carry,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn update_nzcv<C, W>(
    context: &mut C,
    state: &mut State<W>,
    left: &[W; 64],
    right: &[W; 64],
    result: &[W; 64],
    carry: W,
    subtract: bool,
    width64: bool,
    left_constant: Option<u64>,
    result_constant: Option<u64>,
    one: &W,
) -> Result<(), DecodeError>
where
    C: ContextWithErtOps<bool, Wrapped = W>,
    W: Clone,
{
    let bits = if width64 { 64 } else { 32 };
    let n = result[bits - 1].clone();
    let z = if width64 {
        zero_word(context, result, one)
    } else {
        let result32: [W; 32] = core::array::from_fn(|bit| result[bit].clone());
        zero_word(context, &result32, one)
    }
    .map_err(|_| DecodeError::Unsupported(0))?;
    let v = if subtract {
        subtract_overflow(
            context,
            left[bits - 1].clone(),
            right[bits - 1].clone(),
            result[bits - 1].clone(),
        )
    } else {
        add_overflow(
            context,
            left[bits - 1].clone(),
            right[bits - 1].clone(),
            result[bits - 1].clone(),
        )
    }
    .map_err(|_| DecodeError::Unsupported(0))?;
    state.nzcv = [n, z, carry, v];
    let mask = if width64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    state.nzcv_constants =
        left_constant
            .zip(result_constant)
            .map_or([None; 4], |(left, result)| {
                let right = if subtract {
                    left.wrapping_sub(result)
                } else {
                    result.wrapping_sub(left)
                } & mask;
                let sign = 1u64 << (bits - 1);
                let carry = if subtract {
                    left & mask >= right
                } else {
                    (left & mask) > mask - right
                };
                let overflow = if subtract {
                    ((left ^ right) & (left ^ result) & sign) != 0
                } else {
                    ((left ^ result) & (right ^ result) & sign) != 0
                };
                [
                    Some(result & sign != 0),
                    Some(result & mask == 0),
                    Some(carry),
                    Some(overflow),
                ]
            });
    Ok(())
}

fn multiply_low<C, W>(
    context: &mut C,
    left: &[W; 64],
    right: &[W; 64],
    width64: bool,
    zero: &W,
) -> Result<[W; 64], DecodeError>
where
    C: ContextWithErtOps<bool, Wrapped = W>,
    W: Clone,
{
    if width64 {
        multiply_low_word(context, left, right, zero)
    } else {
        let left32: [W; 32] = core::array::from_fn(|bit| left[bit].clone());
        let right32: [W; 32] = core::array::from_fn(|bit| right[bit].clone());
        let low = multiply_low_word(context, &left32, &right32, zero)?;
        Ok(core::array::from_fn(|bit| {
            if bit < 32 {
                low[bit].clone()
            } else {
                zero.clone()
            }
        }))
    }
}

fn multiply_low_word<C, W, const N: usize>(
    context: &mut C,
    left: &[W; N],
    right: &[W; N],
    zero: &W,
) -> Result<[W; N], DecodeError>
where
    C: ContextWithErtOps<bool, Wrapped = W>,
    W: Clone,
{
    let mut accumulator: [W; N] = core::array::from_fn(|_| zero.clone());
    let mut addend = left.clone();
    for bit in 0..N {
        let candidate = add_bits(context, &accumulator, &addend, zero.clone())
            .map_err(|_| DecodeError::Unsupported(0))?;
        accumulator = select_word(context, right[bit].clone(), &candidate, &accumulator)
            .map_err(|_| DecodeError::Unsupported(0))?;
        addend = fixed_shift(&addend, 1, Shift::Left, zero);
    }
    Ok(accumulator)
}

fn register_word<W: Clone>(state: &State<W>, register: u8, zero: &W) -> [W; 64] {
    if register == 31 {
        core::array::from_fn(|_| zero.clone())
    } else {
        state.regs[register as usize].clone()
    }
}

fn register_constant<W>(state: &State<W>, register: u8) -> Option<u64> {
    if register == 31 {
        Some(0)
    } else {
        state.constants[register as usize]
    }
}

fn shift_register<W: Clone>(
    state: &State<W>,
    register: u8,
    amount: u32,
    shift: Shift,
    width64: bool,
    zero: &W,
) -> [W; 64] {
    let source = register_word(state, register, zero);
    if width64 {
        fixed_shift(&source, amount, shift, zero)
    } else {
        let source32: [W; 32] = core::array::from_fn(|bit| source[bit].clone());
        let shifted = fixed_shift(&source32, amount, shift, zero);
        core::array::from_fn(|bit| {
            if bit < 32 {
                shifted[bit].clone()
            } else {
                zero.clone()
            }
        })
    }
}

fn shift_constant(value: u64, amount: u32, shift: Shift, width64: bool) -> u64 {
    let mask = if width64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let value = value & mask;
    match shift {
        Shift::Left => value.wrapping_shl(amount) & mask,
        Shift::LogicalRight => value >> amount,
        Shift::ArithmeticRight => {
            let signed = if width64 {
                value as i64
            } else {
                i64::from(value as u32 as i32)
            };
            (signed >> amount) as u64 & mask
        }
        Shift::RotateRight => unreachable!("A64 add/sub shifted register excludes ROR"),
    }
}

/// A supported, audited A64 instruction form.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Instruction {
    /// Unsigned-immediate integer load over a concrete register base.
    Load {
        /// Destination `Wt`/`Xt`; 31 discards the loaded result.
        dest: u8,
        /// Base register; 31 denotes the architectural SP.
        base: u8,
        /// Unsigned immediate offset after the size scale.
        offset: u64,
        /// Access width in bytes (one, two, four, or eight).
        width: u8,
        /// Whether the loaded value is sign-extended.
        signed: bool,
    },
    /// Signed 9-bit unscaled integer load over a concrete register base.
    LoadExtend {
        /// Destination `Wt`/`Xt`; 31 discards the loaded result.
        dest: u8,
        /// Base register; 31 is rejected until the SP seam exists.
        base: u8,
        /// Sign-extended byte offset.
        offset: i64,
        /// Access width in bytes (one, two, four, or eight).
        width: u8,
        /// Whether the loaded value is sign-extended.
        signed: bool,
    },
    /// `LDR`/`LDRSW` literal with a PC-relative concrete address.
    LoadLiteral {
        /// Destination `Wt`/`Xt`; 31 discards the loaded result.
        dest: u8,
        /// Access width in bytes (four or eight).
        width: u8,
        /// Whether a four-byte value is sign-extended to 64 bits.
        signed: bool,
        /// Computed literal address.
        target: u64,
    },
    /// `ADR` or `ADRP`; target is the materialized PC-relative address.
    Address {
        /// Destination `Xd`, restricted to 0 through 30.
        dest: u8,
        /// Computed PC-relative address.
        target: u64,
    },
    /// `B imm26`; target is `pc + sign_extend(imm26 << 2)`.
    Branch {
        /// Destination fetch address.
        target: u64,
    },
    /// `BL imm26`; target is `pc + sign_extend(imm26 << 2)`.
    BranchLink {
        /// Destination fetch address.
        target: u64,
    },
    /// `B.cond imm19`; true branches to `target`, false falls through four bytes.
    ConditionalBranch {
        /// Arm NZCV condition code.
        condition: u8,
        /// Destination fetch address when true.
        target: u64,
    },
    /// `CBZ`/`CBNZ` over a 32- or 64-bit register.
    CompareBranch {
        /// Tested register, with 31 meaning the architectural zero register.
        register: u8,
        /// Whether this is CBNZ rather than CBZ.
        nonzero: bool,
        /// Whether the register is 64-bit (`Xn`) rather than 32-bit (`Wn`).
        width64: bool,
        /// Destination fetch address when the condition holds.
        target: u64,
    },
    /// `MOVZ` or `MOVK` immediate. `MOVN` remains unsupported in this cut.
    MoveWide {
        /// Destination `Wd`/`Xd`, restricted to 0 through 30.
        dest: u8,
        /// 16-bit immediate payload.
        immediate: u64,
        /// Bit position of the immediate payload (a multiple of 16).
        shift: u32,
        /// Whether this is MOVK (keep other bits) rather than MOVZ.
        keep: bool,
        /// Whether this is the 64-bit X-register form.
        width64: bool,
    },
    /// `ADD`/`SUB` immediate. Register-31 sources are rejected in this first
    /// cut because this encoding treats them as SP rather than XZR.
    AddImmediate {
        /// Destination `Wd`/`Xd`; 31 is accepted for flag-setting aliases.
        dest: u8,
        /// Source `Wn`/`Xn`, restricted to 0 through 30.
        source: u8,
        /// Zero-extended immediate after its optional 12-bit shift.
        immediate: u64,
        /// Whether this is subtraction rather than addition.
        subtract: bool,
        /// Whether the instruction writes NZCV (`ADDS`/`SUBS`).
        set_flags: bool,
        /// Whether this is the 64-bit X-register form.
        width64: bool,
    },
    /// `MADD` or `MSUB`, with `MUL` represented by an XZR/WZR addend.
    MultiplyAdd {
        /// Destination `Wd`/`Xd`; 31 discards the result.
        dest: u8,
        /// First multiplier, with 31 denoting XZR/WZR.
        left: u8,
        /// Second multiplier, with 31 denoting XZR/WZR.
        right: u8,
        /// Addend/minuend, with 31 denoting XZR/WZR.
        addend: u8,
        /// Whether this is MSUB rather than MADD.
        subtract: bool,
        /// Whether this is the 64-bit X-register form.
        width64: bool,
    },
    /// `CSEL`, selecting between two registers using NZCV.
    ConditionalSelect {
        /// Destination `Wd`/`Xd`; 31 discards the result.
        dest: u8,
        /// Register selected when the condition holds, with 31 denoting XZR/WZR.
        when_true: u8,
        /// Register selected when the condition does not hold, with 31 denoting XZR/WZR.
        when_false: u8,
        /// Arm NZCV condition code.
        condition: u8,
        /// Whether this is the 64-bit X-register form.
        width64: bool,
    },
    /// Logical shifted-register operation, including the flag-setting TST alias.
    LogicalRegister {
        /// Destination `Wd`/`Xd`; 31 discards the result.
        dest: u8,
        /// Left operand, with 31 denoting XZR/WZR.
        left: u8,
        /// Right operand before the fixed shift, with 31 denoting XZR/WZR.
        right: u8,
        /// Fixed right-operand shift or rotation.
        shift: Shift,
        /// Fixed shift amount.
        amount: u32,
        /// Logical operation.
        operation: BitOp,
        /// Whether this is the flag-setting ANDS/TST form.
        set_flags: bool,
        /// Whether this is the 64-bit X-register form.
        width64: bool,
    },
    /// `ADD`/`SUB` shifted register, including flag-setting aliases.
    AddRegister {
        /// Destination `Wd`/`Xd`; 31 is accepted for flag-setting aliases.
        dest: u8,
        /// Left operand, restricted to 0 through 30.
        left: u8,
        /// Right operand before the fixed shift, restricted to 0 through 30.
        right: u8,
        /// Fixed right-operand shift.
        shift: Shift,
        /// Fixed shift amount.
        amount: u32,
        /// Whether this is subtraction rather than addition.
        subtract: bool,
        /// Whether the instruction writes NZCV (`ADDS`/`SUBS`).
        set_flags: bool,
        /// Whether this is the 64-bit X-register form.
        width64: bool,
    },
    /// `TBZ`/`TBNZ`.
    TestBranch {
        /// Tested register, with 31 meaning the architectural zero register.
        register: u8,
        /// Bit number (0 through 63).
        bit: u8,
        /// Whether this is TBNZ rather than TBZ.
        nonzero: bool,
        /// Destination fetch address when the test holds.
        target: u64,
    },
    /// `BR Xn`.
    BranchRegister {
        /// Source register. Register 31 is rejected by the decoder.
        register: u8,
    },
    /// `BLR Xn`.
    BranchLinkRegister {
        /// Source register. Register 31 is rejected by the decoder.
        register: u8,
    },
    /// Canonical `RET X30` only.
    Return,
    /// `SVC #0`, reserved for the bare-metal ERT ABI.
    SupervisorCall,
}

/// A word was not an audited A64 form.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeError {
    /// `disarm64` rejected the instruction word.
    Invalid(u32),
    /// The decoder recognized it, but ERT has no audited semantic support.
    Unsupported(u32),
    /// A near-match used a reserved register, immediate, or exception form.
    Malformed(u32),
    /// An audited concrete memory access was unmapped.
    Memory(u64),
}

/// Decode one A64 instruction at `pc`.
///
/// The PC must be four-byte aligned. Every accepted form is independently
/// masked and decoded after `disarm64` accepts it, so changing the dependency
/// cannot silently change ERT support.
pub fn decode(pc: u64, raw: u32) -> Result<Instruction, DecodeError> {
    if pc & 3 != 0 {
        return Err(DecodeError::Malformed(raw));
    }
    decoder::decode(raw).ok_or(DecodeError::Invalid(raw))?;

    if raw & 0x3b20_0000 == 0x3900_0000 {
        let size = (raw >> 30) & 3;
        let v = raw & (1 << 26) != 0;
        let opc = (raw >> 22) & 3;
        let base = ((raw >> 5) & 31) as u8;
        let imm12 = u64::from((raw >> 10) & 0xfff);
        let width = 1u8 << size;
        let (signed, load) = match opc {
            0 => (false, false),
            1 => (false, true),
            2 if v => return Err(DecodeError::Unsupported(raw)),
            2 => (true, true),
            3 if size == 3 => return Err(DecodeError::Unsupported(raw)),
            3 => (true, true),
            _ => unreachable!("two-bit opc"),
        };
        if !load {
            return Err(DecodeError::Unsupported(raw));
        }
        if base == 31 {
            return Err(DecodeError::Unsupported(raw));
        }
        return Ok(Instruction::Load {
            dest: (raw & 31) as u8,
            base,
            offset: imm12 << size,
            width,
            signed,
        });
    }
    if raw & 0x3b20_0000 == 0x3800_0000 {
        let size = (raw >> 30) & 3;
        let v = raw & (1 << 26) != 0;
        let opc = (raw >> 22) & 3;
        let base = ((raw >> 5) & 31) as u8;
        let offset = sign_extend((raw >> 12) & 0x1ff, 9);
        let (signed, load) = match opc {
            0 => (false, false),
            1 => (false, true),
            2 if v => return Err(DecodeError::Unsupported(raw)),
            2 => (true, true),
            3 if size == 3 => return Err(DecodeError::Unsupported(raw)),
            3 => (true, true),
            _ => unreachable!("two-bit opc"),
        };
        if !load || base == 31 {
            return Err(DecodeError::Unsupported(raw));
        }
        return Ok(Instruction::LoadExtend {
            dest: (raw & 31) as u8,
            base,
            offset,
            width: 1u8 << size,
            signed,
        });
    }
    if raw & 0x3f00_0000 == 0x1800_0000 {
        let (width, signed) = match (raw >> 30) & 3 {
            0 => (4, false),
            1 => (8, false),
            2 => (4, true),
            _ => return Err(DecodeError::Unsupported(raw)),
        };
        let target = pc.wrapping_add_signed(sign_extend((raw >> 5) & 0x7f_ffff, 19) << 2);
        return Ok(Instruction::LoadLiteral {
            dest: (raw & 31) as u8,
            width,
            signed,
            target,
        });
    }
    if raw & 0x1f00_0000 == 0x1000_0000 {
        let dest = (raw & 31) as u8;
        if dest == 31 {
            return Err(DecodeError::Unsupported(raw));
        }
        let immediate = ((raw >> 29) & 3) | (((raw >> 5) & 0x7f_ffff) << 2);
        let page = raw & (1 << 31) != 0;
        let base = if page { pc & !0xfff } else { pc };
        let target =
            base.wrapping_add_signed(sign_extend(immediate, 21) << if page { 12 } else { 0 });
        return Ok(Instruction::Address { dest, target });
    }
    if raw & 0x7f80_0000 == 0x5280_0000 || raw & 0x7f80_0000 == 0x7280_0000 {
        let dest = (raw & 31) as u8;
        let width64 = raw & (1 << 31) != 0;
        let shift = ((raw >> 21) & 3) * 16;
        if dest == 31 || (!width64 && shift >= 32) {
            return Err(DecodeError::Unsupported(raw));
        }
        return Ok(Instruction::MoveWide {
            dest,
            immediate: u64::from((raw >> 5) & 0xffff),
            shift,
            keep: raw & 0x7f80_0000 == 0x7280_0000,
            width64,
        });
    }
    if raw & 0x1f00_0000 == 0x1100_0000 {
        let dest = (raw & 31) as u8;
        let source = ((raw >> 5) & 31) as u8;
        let set_flags = raw & (1 << 29) != 0;
        // Source register 31 is SP for this encoding. Destination 31 is XZR
        // for the flag-setting CMP/CMN aliases.
        if source == 31 || (dest == 31 && !set_flags) {
            return Err(DecodeError::Unsupported(raw));
        }
        let immediate = u64::from((raw >> 10) & 0xfff) << if raw & (1 << 22) != 0 { 12 } else { 0 };
        return Ok(Instruction::AddImmediate {
            dest,
            source,
            immediate,
            subtract: raw & (1 << 30) != 0,
            set_flags,
            width64: raw & (1 << 31) != 0,
        });
    }
    if raw & 0x7fe0_8000 == 0x1b00_0000 || raw & 0x7fe0_8000 == 0x1b00_8000 {
        return Ok(Instruction::MultiplyAdd {
            dest: (raw & 31) as u8,
            left: ((raw >> 5) & 31) as u8,
            addend: ((raw >> 10) & 31) as u8,
            right: ((raw >> 16) & 31) as u8,
            subtract: raw & (1 << 15) != 0,
            width64: raw & (1 << 31) != 0,
        });
    }
    if raw & 0x7fe0_0c00 == 0x1a80_0000 {
        let condition = ((raw >> 12) & 15) as u8;
        if condition >= 14 {
            return Err(DecodeError::Malformed(raw));
        }
        return Ok(Instruction::ConditionalSelect {
            dest: (raw & 31) as u8,
            when_true: ((raw >> 5) & 31) as u8,
            when_false: ((raw >> 16) & 31) as u8,
            condition,
            width64: raw & (1 << 31) != 0,
        });
    }
    if raw & 0x1f20_0000 == 0x0a00_0000 {
        let dest = (raw & 31) as u8;
        let left = ((raw >> 5) & 31) as u8;
        let right = ((raw >> 16) & 31) as u8;
        let width64 = raw & (1 << 31) != 0;
        let amount = (raw >> 10) & 63;
        let shift = match (raw >> 22) & 3 {
            0 => Shift::Left,
            1 => Shift::LogicalRight,
            2 => Shift::ArithmeticRight,
            3 => Shift::RotateRight,
            _ => unreachable!("two-bit shift field"),
        };
        if !width64 && amount >= 32 {
            return Err(DecodeError::Unsupported(raw));
        }
        let (operation, set_flags) = match (raw >> 29) & 3 {
            0 => (BitOp::And, false),
            1 => (BitOp::Or, false),
            2 => (BitOp::Xor, false),
            3 => (BitOp::And, true),
            _ => unreachable!("two-bit opcode field"),
        };
        return Ok(Instruction::LogicalRegister {
            dest,
            left,
            right,
            shift,
            amount,
            operation,
            set_flags,
            width64,
        });
    }
    if raw & 0x1f20_0000 == 0x0b00_0000 {
        let dest = (raw & 31) as u8;
        let left = ((raw >> 5) & 31) as u8;
        let right = ((raw >> 16) & 31) as u8;
        let set_flags = raw & (1 << 29) != 0;
        let width64 = raw & (1 << 31) != 0;
        let amount = (raw >> 10) & 63;
        let shift = match (raw >> 22) & 3 {
            0 => Shift::Left,
            1 => Shift::LogicalRight,
            2 => Shift::ArithmeticRight,
            _ => return Err(DecodeError::Malformed(raw)),
        };
        if left == 31 || right == 31 || (dest == 31 && !set_flags) || (!width64 && amount >= 32) {
            return Err(DecodeError::Unsupported(raw));
        }
        return Ok(Instruction::AddRegister {
            dest,
            left,
            right,
            shift,
            amount,
            subtract: raw & (1 << 30) != 0,
            set_flags,
            width64,
        });
    }
    if raw & 0x7c00_0000 == 0x1400_0000 {
        let target = pc.wrapping_add_signed(sign_extend(raw & 0x03ff_ffff, 26) << 2);
        return Ok(if raw & 0x8000_0000 == 0 {
            Instruction::Branch { target }
        } else {
            Instruction::BranchLink { target }
        });
    }
    if raw & 0xff00_0010 == 0x5400_0000 {
        let condition = (raw & 15) as u8;
        if condition >= 14 {
            return Err(DecodeError::Malformed(raw));
        }
        let target = pc.wrapping_add_signed(sign_extend((raw >> 5) & 0x7f_ffff, 19) << 2);
        return Ok(Instruction::ConditionalBranch { condition, target });
    }
    if raw & 0x7e00_0000 == 0x3400_0000 {
        let target = pc.wrapping_add_signed(sign_extend((raw >> 5) & 0x7f_ffff, 19) << 2);
        return Ok(Instruction::CompareBranch {
            register: (raw & 31) as u8,
            nonzero: raw & (1 << 24) != 0,
            width64: raw & (1 << 31) != 0,
            target,
        });
    }
    if raw & 0x7e00_0000 == 0x3600_0000 {
        let target = pc.wrapping_add_signed(sign_extend((raw >> 5) & 0x3fff, 14) << 2);
        return Ok(Instruction::TestBranch {
            register: (raw & 31) as u8,
            bit: (((raw >> 31) & 1) << 5 | ((raw >> 19) & 31)) as u8,
            nonzero: raw & (1 << 24) != 0,
            target,
        });
    }
    if raw & 0xffff_fc1f == 0xd61f_0000 {
        return register_branch(raw, |register| Instruction::BranchRegister { register });
    }
    if raw & 0xffff_fc1f == 0xd63f_0000 {
        return register_branch(raw, |register| Instruction::BranchLinkRegister { register });
    }
    if raw & 0xffff_fc1f == 0xd65f_0000 {
        return if ((raw >> 5) & 31) == 30 {
            Ok(Instruction::Return)
        } else {
            Err(DecodeError::Unsupported(raw))
        };
    }
    if raw & 0xffe0_001f == 0xd400_0001 {
        return if (raw >> 5) & 0xffff == 0 {
            Ok(Instruction::SupervisorCall)
        } else {
            Err(DecodeError::Unsupported(raw))
        };
    }
    Err(DecodeError::Unsupported(raw))
}

fn register_branch(
    raw: u32,
    make: impl FnOnce(u8) -> Instruction,
) -> Result<Instruction, DecodeError> {
    let register = ((raw >> 5) & 31) as u8;
    if register == 31 {
        Err(DecodeError::Malformed(raw))
    } else {
        Ok(make(register))
    }
}

fn sign_extend(value: u32, width: u32) -> i64 {
    (i64::from(value) << (64 - width)) >> (64 - width)
}

#[cfg(test)]
mod tests {
    use super::{
        BitOp, DecodeError, Flow, Instruction, RawMemory, Shift, decode, initial_state, step,
        step_with_memory,
    };

    fn word(value: u64) -> [bool; 64] {
        core::array::from_fn(|bit| (value >> bit) & 1 != 0)
    }

    #[test]
    fn branch_targets_and_link_bit_are_extracted_from_audited_masks() {
        assert_eq!(
            decode(0x1000, 0x1400_0002),
            Ok(Instruction::Branch { target: 0x1008 })
        );
        assert_eq!(
            decode(0x1000, 0x9400_0002),
            Ok(Instruction::BranchLink { target: 0x1008 })
        );
        assert_eq!(
            decode(0x1000, 0x17ff_fffe),
            Ok(Instruction::Branch { target: 0x0ff8 })
        );
    }

    #[test]
    fn conditional_and_test_branches_extract_pc_relative_targets() {
        assert_eq!(
            decode(0x2000, 0x5400_0040),
            Ok(Instruction::ConditionalBranch {
                condition: 0,
                target: 0x2008
            })
        );
        assert_eq!(
            decode(0x2000, 0xb500_0041),
            Ok(Instruction::CompareBranch {
                register: 1,
                nonzero: true,
                width64: true,
                target: 0x2008
            })
        );
    }

    #[test]
    fn register_branches_and_svc_are_narrowly_accepted() {
        assert_eq!(
            decode(0, 0xd61f_0060),
            Ok(Instruction::BranchRegister { register: 3 })
        );
        assert_eq!(decode(0, 0xd65f_03c0), Ok(Instruction::Return));
        assert_eq!(decode(0, 0xd400_0001), Ok(Instruction::SupervisorCall));
        assert_eq!(
            decode(0, 0xd400_0021),
            Err(DecodeError::Unsupported(0xd400_0021))
        );
    }

    #[test]
    fn add_sub_immediate_preserve_x31_xzr_and_w_zero_extension() {
        let mut state = initial_state(false);
        state.regs[1] = word(4);
        state.constants[1] = Some(4);
        assert_eq!(
            decode(0x1000, 0x9100_0420),
            Ok(Instruction::AddImmediate {
                dest: 0,
                source: 1,
                immediate: 1,
                subtract: false,
                set_flags: false,
                width64: true
            })
        );
        assert_eq!(
            step(&mut (), &mut state, 0x1000, 0x9100_0420, &false, &true),
            Ok(Flow::Next(word(0x1004)))
        );
        assert_eq!(state.regs[0], word(5));
        assert_eq!(state.constants[0], Some(5));
        state.regs[1] = word(0x1_0000_0000);
        state.constants[1] = Some(0x1_0000_0000);
        assert_eq!(
            step(&mut (), &mut state, 0x1004, 0x1100_0420, &false, &true),
            Ok(Flow::Next(word(0x1008)))
        );
        assert_eq!(state.regs[0], word(1));
        assert_eq!(state.constants[0], Some(1));
        assert_eq!(
            decode(0, 0x9100_043f),
            Err(DecodeError::Unsupported(0x9100_043f))
        );
    }

    #[test]
    fn literal_loads_are_concrete_pc_relative_and_fail_closed_when_unmapped() {
        let mut bytes = [0u8; 16];
        bytes[8..].copy_from_slice(&0xfeed_face_dead_beefu64.to_le_bytes());
        let memory = RawMemory::from_slice(&bytes);
        let mut state = initial_state(false);
        assert_eq!(
            decode(0, 0x5800_0040),
            Ok(Instruction::LoadLiteral {
                dest: 0,
                width: 8,
                signed: false,
                target: 8,
            })
        );
        assert_eq!(
            step_with_memory(&mut (), &mut state, 0, 0x5800_0040, memory, &false, &true),
            Ok(Flow::Next(word(4)))
        );
        assert_eq!(state.regs[0], word(0xfeed_face_dead_beef));
        assert_eq!(
            step_with_memory(&mut (), &mut state, 0, 0x1800_0041, memory, &false, &true),
            Ok(Flow::Next(word(4)))
        );
        assert_eq!(state.constants[1], Some(0xdead_beef));
        assert_eq!(
            step_with_memory(&mut (), &mut state, 0, 0x9800_0042, memory, &false, &true),
            Ok(Flow::Next(word(4)))
        );
        assert_eq!(state.constants[2], Some(0xffff_ffff_dead_beef));
        assert_eq!(
            step_with_memory(&mut (), &mut state, 0, 0x5800_0083, memory, &false, &true),
            Err(DecodeError::Memory(16))
        );
    }

    #[test]
    fn unsigned_immediate_loads_cover_widths_and_fail_closed_addresses() {
        let mut bytes = [0u8; 64];
        bytes[32] = 0x80;
        bytes[34..36].copy_from_slice(&0xbeefu16.to_le_bytes());
        bytes[36..40].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        bytes[48..56].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
        let memory = RawMemory::from_slice(&bytes);
        let mut state = initial_state(false);
        state.constants[1] = Some(32);
        assert_eq!(
            decode(0x1000, 0xf940_0020),
            Ok(Instruction::Load {
                dest: 0,
                base: 1,
                offset: 0,
                width: 8,
                signed: false,
            })
        );
        assert_eq!(
            step_with_memory(
                &mut (),
                &mut state,
                0x1000,
                0x3940_0022,
                memory,
                &false,
                &true
            ),
            Ok(Flow::Next(word(0x1004)))
        );
        assert_eq!(state.constants[2], Some(0x80));
        assert_eq!(
            step_with_memory(
                &mut (),
                &mut state,
                0x1004,
                0x7980_0423,
                memory,
                &false,
                &true
            ),
            Ok(Flow::Next(word(0x1008)))
        );
        assert_eq!(state.constants[3], Some(0xffff_ffff_ffff_beef));
        assert_eq!(
            step_with_memory(
                &mut (),
                &mut state,
                0x1008,
                0xb940_0424,
                memory,
                &false,
                &true
            ),
            Ok(Flow::Next(word(0x100c)))
        );
        assert_eq!(state.constants[4], Some(0xdead_beef));
        assert_eq!(state.constants[1], Some(32));
        assert_eq!(
            decode(0x100c, 0xf940_0825),
            Ok(Instruction::Load {
                dest: 5,
                base: 1,
                offset: 16,
                width: 8,
                signed: false,
            })
        );
        assert_eq!(
            step_with_memory(
                &mut (),
                &mut state,
                0x100c,
                0xf940_0825,
                memory,
                &false,
                &true
            ),
            Ok(Flow::Next(word(0x1010)))
        );
        assert_eq!(state.constants[5], Some(0x1122_3344_5566_7788));
        assert_eq!(
            step_with_memory(
                &mut (),
                &mut state,
                0x1010,
                0xb940_9826,
                memory,
                &false,
                &true
            ),
            Err(DecodeError::Memory(184))
        );
        assert_eq!(
            step_with_memory(
                &mut (),
                &mut state,
                0x1010,
                0x3900_0026,
                memory,
                &false,
                &true
            ),
            Err(DecodeError::Unsupported(0x3900_0026))
        );
    }

    #[test]
    fn unscaled_loads_use_signed_offsets_and_fail_closed_bases() {
        let mut bytes = [0u8; 32];
        bytes[8..16].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
        bytes[17] = 0x88;
        let memory = RawMemory::from_slice(&bytes);
        let mut state = initial_state(false);
        state.constants[1] = Some(12);
        assert_eq!(
            decode(0x1000, 0xf85f_e820),
            Ok(Instruction::LoadExtend {
                dest: 0,
                base: 1,
                offset: -2,
                width: 8,
                signed: false,
            })
        );
        assert_eq!(
            step_with_memory(
                &mut (),
                &mut state,
                0x1000,
                0xf85f_c820,
                memory,
                &false,
                &true
            ),
            Ok(Flow::Next(word(0x1004)))
        );
        assert_eq!(state.constants[0], Some(0x1122_3344_5566_7788));
        state.constants[1] = Some(8);
        assert_eq!(
            step_with_memory(
                &mut (),
                &mut state,
                0x1004,
                0x3840_9021,
                memory,
                &false,
                &true
            ),
            Ok(Flow::Next(word(0x1008)))
        );
        assert_eq!(state.constants[1], Some(0x88));
        assert_eq!(
            step_with_memory(
                &mut (),
                &mut state,
                0x1008,
                0xf840_905f,
                memory,
                &false,
                &true
            ),
            Err(DecodeError::Unsupported(0xf840_905f))
        );
    }

    #[test]
    fn svc_exit_requires_the_bare_metal_all_ones_selector() {
        let mut state = initial_state(false);
        assert_eq!(
            step(&mut (), &mut state, 0x1000, 0xd400_0001, &false, &true),
            Err(DecodeError::Unsupported(0xd400_0001))
        );
        state.regs[0] = word(u64::MAX);
        state.constants[0] = Some(u64::MAX);
        assert_eq!(
            step(&mut (), &mut state, 0x1000, 0xd400_0001, &false, &true),
            Ok(Flow::Exit)
        );
        assert!(state.done);
    }

    #[test]
    fn madd_msub_and_mul_aliases_use_low_width_products() {
        let mut state = initial_state(false);
        state.regs[1] = word(3);
        state.regs[2] = word(4);
        state.regs[3] = word(5);
        state.constants[1] = Some(3);
        state.constants[2] = Some(4);
        state.constants[3] = Some(5);
        assert_eq!(
            decode(0x1000, 0x9b02_0c20),
            Ok(Instruction::MultiplyAdd {
                dest: 0,
                left: 1,
                right: 2,
                addend: 3,
                subtract: false,
                width64: true,
            })
        );
        assert_eq!(
            step(&mut (), &mut state, 0x1000, 0x9b02_0c20, &false, &true),
            Ok(Flow::Next(word(0x1004)))
        );
        assert_eq!(state.regs[0], word(17));
        assert_eq!(
            step(&mut (), &mut state, 0x1004, 0x9b02_7c20, &false, &true),
            Ok(Flow::Next(word(0x1008)))
        );
        assert_eq!(state.regs[0], word(12));
        state.regs[1] = word(0x1_0000_0002);
        state.constants[1] = Some(0x1_0000_0002);
        assert_eq!(
            step(&mut (), &mut state, 0x1008, 0x1b02_7c20, &false, &true),
            Ok(Flow::Next(word(0x100c)))
        );
        assert_eq!(state.regs[0], word(8));
        assert_eq!(state.constants[0], Some(8));
        state.regs[1] = word(3);
        state.constants[1] = Some(3);
        assert_eq!(
            step(&mut (), &mut state, 0x100c, 0x9b02_8c20, &false, &true),
            Ok(Flow::Next(word(0x1010)))
        );
        assert_eq!(state.regs[0], word(u64::MAX - 6));
        assert_eq!(state.constants[0], Some(u64::MAX - 6));
    }

    #[test]
    fn csel_selects_from_nzcv_and_preserves_xzr() {
        let mut state = initial_state(false);
        state.regs[1] = word(0x11);
        state.regs[2] = word(0x22);
        state.constants[1] = Some(0x11);
        state.constants[2] = Some(0x22);
        assert_eq!(
            decode(0x1000, 0x9a82_0020),
            Ok(Instruction::ConditionalSelect {
                dest: 0,
                when_true: 1,
                when_false: 2,
                condition: 0,
                width64: true,
            })
        );
        assert_eq!(
            step(&mut (), &mut state, 0x1000, 0x9a82_0020, &false, &true),
            Ok(Flow::Next(word(0x1004)))
        );
        assert_eq!(state.regs[0], word(0x22));
        state.nzcv = [false, true, false, false];
        state.nzcv_constants = [Some(false), Some(true), Some(false), Some(false)];
        assert_eq!(
            step(&mut (), &mut state, 0x1004, 0x9a82_0020, &false, &true),
            Ok(Flow::Next(word(0x1008)))
        );
        assert_eq!(state.regs[0], word(0x11));
        assert_eq!(state.constants[0], Some(0x11));
        assert_eq!(
            step(&mut (), &mut state, 0x1008, 0x9a9f_03e0, &false, &true),
            Ok(Flow::Next(word(0x100c)))
        );
        assert_eq!(state.regs[0], word(0));
    }

    #[test]
    fn adr_and_adrp_materialize_audited_pc_relative_addresses() {
        let mut state = initial_state(false);
        assert_eq!(
            decode(0x1000, 0x1000_0040),
            Ok(Instruction::Address {
                dest: 0,
                target: 0x1008,
            })
        );
        assert_eq!(
            step(&mut (), &mut state, 0x1000, 0x1000_0040, &false, &true),
            Ok(Flow::Next(word(0x1004)))
        );
        assert_eq!(state.regs[0], word(0x1008));
        assert_eq!(
            step(&mut (), &mut state, 0x1234, 0xb000_0000, &false, &true),
            Ok(Flow::Next(word(0x1238)))
        );
        assert_eq!(state.regs[0], word(0x2000));
        assert_eq!(state.constants[0], Some(0x2000));
        assert_eq!(
            decode(0, 0x1000_001f),
            Err(DecodeError::Unsupported(0x1000_001f))
        );
    }

    #[test]
    fn logical_register_operations_support_xzr_shifts_and_tst() {
        let mut state = initial_state(false);
        state.regs[1] = word(0xf0);
        state.regs[2] = word(0x0f);
        state.constants[1] = Some(0xf0);
        state.constants[2] = Some(0x0f);
        assert_eq!(
            decode(0x1000, 0x8a02_0020),
            Ok(Instruction::LogicalRegister {
                dest: 0,
                left: 1,
                right: 2,
                shift: Shift::Left,
                amount: 0,
                operation: BitOp::And,
                set_flags: false,
                width64: true,
            })
        );
        assert_eq!(
            step(&mut (), &mut state, 0x1000, 0x8a02_0020, &false, &true),
            Ok(Flow::Next(word(0x1004)))
        );
        assert_eq!(state.regs[0], word(0));
        assert_eq!(
            step(&mut (), &mut state, 0x1004, 0xea02_003f, &false, &true),
            Ok(Flow::Next(word(0x1008)))
        );
        assert_eq!(
            state.nzcv_constants,
            [Some(false), Some(true), Some(false), Some(false)]
        );
        assert_eq!(
            step(&mut (), &mut state, 0x1008, 0xaa02_03e0, &false, &true),
            Ok(Flow::Next(word(0x100c)))
        );
        assert_eq!(state.regs[0], word(0x0f));
    }

    #[test]
    fn shifted_register_arithmetic_handles_shift_and_cmp_aliases() {
        let mut state = initial_state(false);
        state.regs[1] = word(3);
        state.regs[2] = word(2);
        state.constants[1] = Some(3);
        state.constants[2] = Some(2);
        assert_eq!(
            decode(0x1000, 0x8b02_0820),
            Ok(Instruction::AddRegister {
                dest: 0,
                left: 1,
                right: 2,
                shift: Shift::Left,
                amount: 2,
                subtract: false,
                set_flags: false,
                width64: true,
            })
        );
        assert_eq!(
            step(&mut (), &mut state, 0x1000, 0x8b02_0820, &false, &true),
            Ok(Flow::Next(word(0x1004)))
        );
        assert_eq!(state.regs[0], word(11));
        assert_eq!(state.constants[0], Some(11));
        state.regs[1] = word(0);
        state.regs[2] = word(0x8000_0000);
        state.constants[1] = Some(0);
        state.constants[2] = Some(0x8000_0000);
        assert_eq!(
            step(&mut (), &mut state, 0x1004, 0x0b82_0420, &false, &true),
            Ok(Flow::Next(word(0x1008)))
        );
        assert_eq!(state.regs[0], word(0xc000_0000));
        assert_eq!(state.constants[0], Some(0xc000_0000));
        state.regs[1] = word(3);
        state.regs[2] = word(2);
        state.constants[1] = Some(3);
        state.constants[2] = Some(2);
        assert_eq!(
            step(&mut (), &mut state, 0x1008, 0xeb02_003f, &false, &true),
            Ok(Flow::Next(word(0x100c)))
        );
        assert_eq!(state.nzcv_constants[1], Some(false));
        assert_eq!(state.nzcv_constants[2], Some(true));
    }

    #[test]
    fn flag_setting_arithmetic_drives_conditional_branches_and_cmp_aliases() {
        let mut state = initial_state(false);
        state.regs[1] = word(u64::MAX);
        state.constants[1] = Some(u64::MAX);
        assert_eq!(
            step(&mut (), &mut state, 0x1000, 0xb100_0420, &false, &true),
            Ok(Flow::Next(word(0x1004)))
        );
        assert_eq!(state.regs[0], word(0));
        assert_eq!(
            state.nzcv_constants,
            [Some(false), Some(true), Some(true), Some(false)]
        );
        assert_eq!(
            step(&mut (), &mut state, 0x1004, 0x5400_0040, &false, &true),
            Ok(Flow::Next(word(0x100c)))
        );
        state.regs[1] = word(1);
        state.constants[1] = Some(1);
        assert_eq!(
            step(&mut (), &mut state, 0x2000, 0xf100_043f, &false, &true),
            Ok(Flow::Next(word(0x2004)))
        );
        assert_eq!(state.constants[0], Some(0));
        assert_eq!(state.nzcv_constants[1], Some(true));
    }

    #[test]
    fn movz_movk_decode_and_update_only_the_selected_halfword() {
        let mut state = initial_state(false);
        assert_eq!(
            decode(0, 0xd282_4680),
            Ok(Instruction::MoveWide {
                dest: 0,
                immediate: 0x1234,
                shift: 0,
                keep: false,
                width64: true,
            })
        );
        assert_eq!(
            step(&mut (), &mut state, 0, 0xd282_4680, &false, &true),
            Ok(Flow::Next(word(4)))
        );
        assert_eq!(state.regs[0], word(0x1234));
        assert_eq!(
            step(&mut (), &mut state, 4, 0xf2a2_4680, &false, &true),
            Ok(Flow::Next(word(8)))
        );
        assert_eq!(state.regs[0], word(0x1234_1234));
        assert_eq!(
            decode(0, 0x5280_001f),
            Err(DecodeError::Unsupported(0x5280_001f))
        );
    }

    #[test]
    fn test_and_register_branches_preserve_x31_and_require_concrete_targets() {
        let mut state = initial_state(false);
        state.regs[1] = word(1 << 5);
        let taken = step(&mut (), &mut state, 0x2000, 0x3728_0041, &false, &true).unwrap();
        assert_eq!(taken, Flow::Next(word(0x2008)));
        let xzr = step(&mut (), &mut state, 0x2000, 0x3628_005f, &false, &true).unwrap();
        assert_eq!(xzr, Flow::Next(word(0x2008)));
        assert_eq!(
            step(&mut (), &mut state, 0x2000, 0xd61f_0020, &false, &true),
            Err(DecodeError::Unsupported(0xd61f_0020))
        );
        state.constants[1] = Some(0x3000);
        assert_eq!(
            step(&mut (), &mut state, 0x2000, 0xd61f_0020, &false, &true),
            Ok(Flow::Next(word(0x3000)))
        );
        assert_eq!(
            step(&mut (), &mut state, 0x2000, 0xd63f_0020, &false, &true),
            Ok(Flow::Next(word(0x3000)))
        );
        assert_eq!(state.constants[30], Some(0x2004));
        assert_eq!(
            step(&mut (), &mut state, 0x3000, 0xd65f_03c0, &false, &true),
            Ok(Flow::Next(word(0x2004)))
        );
    }

    #[test]
    fn symbolic_cbz_selects_the_audited_target_and_x31_is_xzr() {
        let mut state = initial_state(false);
        state.regs[1] = core::array::from_fn(|bit| bit == 0);
        let flow = step(&mut (), &mut state, 0x2000, 0xb500_0041, &false, &true).unwrap();
        assert_eq!(flow, Flow::Next(word(0x2008)));
        let zero_register = step(&mut (), &mut state, 0x2000, 0xb400_005f, &false, &true).unwrap();
        assert_eq!(zero_register, Flow::Next(word(0x2008)));
        let zero_register_nonzero =
            step(&mut (), &mut state, 0x2000, 0xb500_005f, &false, &true).unwrap();
        assert_eq!(zero_register_nonzero, Flow::Next(word(0x2004)));
    }

    #[test]
    fn alignment_and_recognized_but_unaudited_forms_fail_closed() {
        assert_eq!(
            decode(2, 0x1400_0000),
            Err(DecodeError::Malformed(0x1400_0000))
        );
        assert_eq!(
            decode(0, 0xd503_201f),
            Err(DecodeError::Unsupported(0xd503_201f))
        );
    }
}
