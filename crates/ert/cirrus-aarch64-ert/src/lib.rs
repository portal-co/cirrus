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
    ContextWithErtOps, Shift, add_bits_with_carry_out, add_overflow, arm_condition, constant_word,
    fixed_shift, invert_word, select_word, subtract_overflow, zero_word,
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
        Instruction::SupervisorCall => Ok(Flow::Exit),
    }
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
    use super::{DecodeError, Flow, Instruction, Shift, decode, initial_state, step};

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
