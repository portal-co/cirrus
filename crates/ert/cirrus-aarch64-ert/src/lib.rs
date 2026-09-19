#![no_std]
#![warn(missing_docs)]

//! A deliberately narrow, fail-closed AArch64 ERT decoder.
//!
//! [`disarm64`] is used only as a classification guard. Every instruction
//! accepted here must also satisfy an ERT-owned raw mask and field extractor;
//! a successful broad decoder result never expands the supported ISA by
//! itself. The initial Phase 3 cut contains only PC-relative/control-flow and
//! `SVC #0` forms, establishing the audited decoder boundary before symbolic
//! arithmetic and memory semantics are added.
//!
//! Instruction field definitions follow Arm A-profile Architecture Reference
//! Manual DDI0487 A64 encoding tables: B/BL (`C4.1.4`), B.cond (`C4.1.5`),
//! CBZ/CBNZ (`C4.1.6`), TBZ/TBNZ (`C4.1.7`), BR/BLR/RET (`C4.1.8`), and SVC
//! (`C4.1.9`). The raw masks below are the support authority; `disarm64`
//! protects this hand-extracted subset against accepting an undecodable word.

use disarm64::decoder;

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
    use super::{DecodeError, Instruction, decode};

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
                target: 0x2008,
            })
        );
        assert_eq!(
            decode(0x2000, 0xb500_0041),
            Ok(Instruction::CompareBranch {
                register: 1,
                nonzero: true,
                width64: true,
                target: 0x2008,
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
