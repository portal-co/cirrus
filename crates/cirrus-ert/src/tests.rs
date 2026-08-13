extern crate std;

use core::{array, convert::Infallible};

use rv_asm::{Imm, Inst, Reg, Xlen};
use std::vec::Vec;

use crate::{ErtError, ert_emit, ert_func, simple_add};

fn word(value: u32) -> [bool; 32] {
    array::from_fn(|bit| value & (1u32 << bit) != 0)
}

fn value(word: &[bool; 32]) -> u32 {
    word.iter()
        .enumerate()
        .fold(0, |value, (bit, set)| value | ((*set as u32) << bit))
}

fn program(instructions: impl IntoIterator<Item = Inst>) -> Vec<u8> {
    instructions
        .into_iter()
        .flat_map(|instruction| instruction.encode_normal(Xlen::Rv32).to_le_bytes())
        .collect()
}

fn run(
    mem: &mut [u8],
    regs: &mut [[bool; 32]; 32],
    constants: &mut [Option<u32>; 32],
    rstack: &mut [u32],
    vstack: &mut [bool],
) -> Result<(), ErtError<Infallible>> {
    let mut context = ();
    let mut hash = no_hash;
    ert_emit(
        &mut context,
        &mut hash,
        mem,
        rstack,
        vstack,
        0,
        regs,
        constants,
        false,
        true,
    )
}

fn no_hash(_: &[[bool; 32]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}

fn assert_success(result: Result<(), ErtError<Infallible>>) {
    assert!(result.is_ok());
}

fn exit_register(regs: &mut [[bool; 32]; 32], constants: &mut [Option<u32>; 32]) {
    regs[Reg::A0.0 as usize] = word(u32::MAX);
    constants[Reg::A0.0 as usize] = Some(u32::MAX);
}

#[test]
fn simple_add_returns_a_boolean_sum() {
    let mut context = ();
    let sum = simple_add(
        &mut context,
        &word(0xffff_ffff),
        &word(2),
        false,
        false,
        true,
    )
    .unwrap();

    assert_eq!(value(&sum), 1);
}

#[test]
fn arithmetic_immediates_and_shifts_preserve_concrete_tracking() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    regs[Reg::T1.0 as usize] = word(0x8000_0003);
    regs[Reg::T2.0 as usize] = word(5);
    constants[Reg::T1.0 as usize] = Some(0x8000_0003);
    constants[Reg::T2.0 as usize] = Some(5);
    regs[Reg::T6.0 as usize] = word(u32::MAX);
    constants[Reg::T6.0 as usize] = Some(u32::MAX);
    exit_register(&mut regs, &mut constants);
    let mut mem = program([
        Inst::Add {
            dest: Reg::T0,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Sub {
            dest: Reg::T3,
            src1: Reg::T2,
            src2: Reg::T1,
        },
        Inst::And {
            dest: Reg::T4,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Or {
            dest: Reg::T5,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Xor {
            dest: Reg::T0,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Andi {
            imm: Imm::new_i32(7),
            dest: Reg::T4,
            src1: Reg::T1,
        },
        Inst::Ori {
            imm: Imm::new_i32(4),
            dest: Reg::T5,
            src1: Reg::T2,
        },
        Inst::Xori {
            imm: Imm::new_i32(3),
            dest: Reg::T3,
            src1: Reg::T2,
        },
        Inst::Slli {
            imm: Imm::new_i32(1),
            dest: Reg::T2,
            src1: Reg::T2,
        },
        Inst::Srli {
            imm: Imm::new_i32(1),
            dest: Reg::T1,
            src1: Reg::T1,
        },
        Inst::Srai {
            imm: Imm::new_i32(1),
            dest: Reg::T6,
            src1: Reg::T6,
        },
        Inst::Ecall,
    ]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];

    assert_success(run(
        &mut mem,
        &mut regs,
        &mut constants,
        &mut rstack,
        &mut vstack,
    ));

    assert_eq!(constants[Reg::T0.0 as usize], Some(0x8000_0006));
    assert_eq!(constants[Reg::T3.0 as usize], Some(6));
    assert_eq!(constants[Reg::T4.0 as usize], Some(3));
    assert_eq!(constants[Reg::T5.0 as usize], Some(5));
    assert_eq!(constants[Reg::T2.0 as usize], Some(10));
    assert_eq!(constants[Reg::T1.0 as usize], Some(0x4000_0001));
    assert_eq!(constants[Reg::T6.0 as usize], Some(u32::MAX));
}

#[test]
fn constants_and_concrete_loads_update_register_metadata() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    regs[Reg::T0.0 as usize] = word(64);
    constants[Reg::T0.0 as usize] = Some(64);
    exit_register(&mut regs, &mut constants);
    let mut mem = program([
        Inst::Lui {
            uimm: Imm::new_u32(0x1234_5000),
            dest: Reg::S0,
        },
        Inst::Auipc {
            uimm: Imm::new_u32(0x1000),
            dest: Reg::T2,
        },
        Inst::Lb {
            offset: Imm::ZERO,
            dest: Reg::T3,
            base: Reg::T0,
        },
        Inst::Lh {
            offset: Imm::ZERO,
            dest: Reg::T4,
            base: Reg::T0,
        },
        Inst::Lbu {
            offset: Imm::ZERO,
            dest: Reg::T5,
            base: Reg::T0,
        },
        Inst::Lhu {
            offset: Imm::ZERO,
            dest: Reg::T6,
            base: Reg::T0,
        },
        Inst::Lw {
            offset: Imm::ZERO,
            dest: Reg::T1,
            base: Reg::T0,
        },
        Inst::Ecall,
    ]);
    mem.resize(96, 0);
    mem[64..68].copy_from_slice(&0x8001_80ffu32.to_le_bytes());
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];

    assert_success(run(
        &mut mem,
        &mut regs,
        &mut constants,
        &mut rstack,
        &mut vstack,
    ));

    assert_eq!(constants[Reg::S0.0 as usize], Some(0x1234_5000));
    assert_eq!(constants[Reg::T2.0 as usize], Some(0x1004));
    assert_eq!(constants[Reg::T3.0 as usize], Some(0xffff_ffff));
    assert_eq!(constants[Reg::T4.0 as usize], Some(0xffff_80ff));
    assert_eq!(constants[Reg::T5.0 as usize], Some(0xff));
    assert_eq!(constants[Reg::T6.0 as usize], Some(0x80ff));
    assert_eq!(constants[Reg::T1.0 as usize], Some(0x8001_80ff));
}

#[test]
fn symbolic_stack_memory_preserves_width_and_extension_rules() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    let source = word(0x8001_80ff);
    regs[Reg::T0.0 as usize] = source;
    exit_register(&mut regs, &mut constants);
    let mut mem = program([
        Inst::Addi {
            imm: Imm::new_i32(-1796),
            dest: Reg::SP,
            src1: Reg::SP,
        },
        Inst::Sb {
            offset: Imm::ZERO,
            src: Reg::T0,
            base: Reg::SP,
        },
        Inst::Sh {
            offset: Imm::new_i32(4),
            src: Reg::T0,
            base: Reg::SP,
        },
        Inst::Sw {
            offset: Imm::new_i32(8),
            src: Reg::T0,
            base: Reg::SP,
        },
        Inst::Lb {
            offset: Imm::ZERO,
            dest: Reg::T1,
            base: Reg::SP,
        },
        Inst::Lbu {
            offset: Imm::ZERO,
            dest: Reg::T2,
            base: Reg::SP,
        },
        Inst::Lh {
            offset: Imm::new_i32(4),
            dest: Reg::T3,
            base: Reg::SP,
        },
        Inst::Lhu {
            offset: Imm::new_i32(4),
            dest: Reg::T4,
            base: Reg::SP,
        },
        Inst::Lw {
            offset: Imm::new_i32(8),
            dest: Reg::T5,
            base: Reg::SP,
        },
        Inst::Addi {
            imm: Imm::new_i32(1796),
            dest: Reg::SP,
            src1: Reg::SP,
        },
        Inst::Ecall,
    ]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 2048];

    assert_success(run(
        &mut mem,
        &mut regs,
        &mut constants,
        &mut rstack,
        &mut vstack,
    ));

    assert_eq!(value(&regs[Reg::T1.0 as usize]), 0xffff_ffff);
    assert_eq!(value(&regs[Reg::T2.0 as usize]), 0xff);
    assert_eq!(value(&regs[Reg::T3.0 as usize]), 0xffff_80ff);
    assert_eq!(value(&regs[Reg::T4.0 as usize]), 0x80ff);
    assert_eq!(value(&regs[Reg::T5.0 as usize]), 0x8001_80ff);
    assert!(constants[Reg::T1.0 as usize].is_none());
    assert_eq!(&vstack[..8], &word(0x8001_80ff)[..8]);
}

#[test]
fn branches_and_call_return_follow_the_private_control_stack() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    regs[Reg::T1.0 as usize] = word(1);
    regs[Reg::T2.0 as usize] = word(2);
    constants[Reg::T1.0 as usize] = Some(1);
    constants[Reg::T2.0 as usize] = Some(2);
    exit_register(&mut regs, &mut constants);
    let mut mem = program([
        Inst::Beq {
            offset: Imm::new_i32(8),
            src1: Reg::T1,
            src2: Reg::T1,
        },
        Inst::Addi {
            imm: Imm::new_i32(1),
            dest: Reg::T0,
            src1: Reg::ZERO,
        },
        Inst::Bne {
            offset: Imm::new_i32(8),
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Addi {
            imm: Imm::new_i32(2),
            dest: Reg::T0,
            src1: Reg::ZERO,
        },
        Inst::Jal {
            offset: Imm::new_i32(12),
            dest: Reg::RA,
        },
        Inst::Addi {
            imm: Imm::new_i32(3),
            dest: Reg::T0,
            src1: Reg::ZERO,
        },
        Inst::Jal {
            offset: Imm::new_i32(8),
            dest: Reg::ZERO,
        },
        Inst::Jalr {
            offset: Imm::ZERO,
            base: Reg::RA,
            dest: Reg::ZERO,
        },
        Inst::Ecall,
    ]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];

    assert_success(run(
        &mut mem,
        &mut regs,
        &mut constants,
        &mut rstack,
        &mut vstack,
    ));

    assert_eq!(constants[Reg::T0.0 as usize], Some(3));
    assert_eq!(rstack[0], 20);
}

#[test]
fn a_not_taken_branch_falls_through() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    regs[Reg::T1.0 as usize] = word(1);
    regs[Reg::T2.0 as usize] = word(2);
    constants[Reg::T1.0 as usize] = Some(1);
    constants[Reg::T2.0 as usize] = Some(2);
    exit_register(&mut regs, &mut constants);
    let mut mem = program([
        Inst::Beq {
            offset: Imm::new_i32(8),
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Addi {
            imm: Imm::new_i32(9),
            dest: Reg::T0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
    ]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];

    assert_success(run(
        &mut mem,
        &mut regs,
        &mut constants,
        &mut rstack,
        &mut vstack,
    ));

    assert_eq!(constants[Reg::T0.0 as usize], Some(9));
}

#[test]
fn hash_ecall_exchanges_eight_words_with_the_callback() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    regs[Reg::A0.0 as usize] = word(0);
    constants[Reg::A0.0 as usize] = Some(0);
    for i in 0..8 {
        regs[Reg::A1.0 as usize + i] = word(i as u32 + 1);
        constants[Reg::A1.0 as usize + i] = Some(i as u32 + 1);
    }
    let mut mem = program([
        Inst::Ecall,
        Inst::Addi {
            imm: Imm::new_i32(-1),
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
    ]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];
    let mut observed = [[false; 32]; 8];
    let mut context = ();
    let mut hash = |words: &[[bool; 32]]| {
        observed.copy_from_slice(words);
        Ok::<_, Infallible>(array::from_fn(|byte| byte as u8))
    };

    assert_success(ert_emit(
        &mut context,
        &mut hash,
        &mut mem,
        &mut rstack,
        &mut vstack,
        0,
        &mut regs,
        &mut constants,
        false,
        true,
    ));

    assert_eq!(value(&observed[0]), 1);
    assert_eq!(value(&observed[7]), 8);
    assert_eq!(constants[Reg::A1.0 as usize], Some(0x0302_0100));
    assert_eq!(constants[Reg::S2.0 as usize], Some(8));
}

#[test]
fn ert_func_moves_register_and_stack_abi_values() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    let mut mem = program([Inst::Ecall]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 128];
    let args = array::from_fn(|i| {
        let constant = if i == 0 { u32::MAX } else { i as u32 };
        (word(constant), Some(constant))
    });
    let mut context = ();
    let mut hash = no_hash;

    let results = match ert_func::<_, _, 10, 10>(
        &mut context,
        &mut hash,
        &mut mem,
        &mut rstack,
        &mut vstack,
        0,
        &mut regs,
        &mut constants,
        false,
        true,
        args,
    ) {
        Ok(results) => results,
        Err(_) => panic!("well-formed ABI fixture must complete"),
    };

    assert_eq!(results[0].1, Some(u32::MAX));
    assert_eq!(results[7].1, Some(7));
    assert_eq!(value(&results[8].0), 8);
    assert_eq!(value(&results[9].0), 9);
    assert!(results[8].1.is_none());
    assert!(results[9].1.is_none());
}

#[test]
fn dynamic_control_and_invalid_words_are_reported() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    let mut mem = program([Inst::Beq {
        offset: Imm::new_i32(4),
        src1: Reg::T0,
        src2: Reg::ZERO,
    }]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];

    assert!(matches!(
        run(
            &mut mem,
            &mut regs,
            &mut constants,
            &mut rstack,
            &mut vstack,
        ),
        Err(ErtError::Unexpected)
    ));

    let mut invalid = [0u8; 4];
    assert!(matches!(
        run(
            &mut invalid,
            &mut regs,
            &mut constants,
            &mut rstack,
            &mut vstack,
        ),
        Err(ErtError::Decode(_))
    ));
}
