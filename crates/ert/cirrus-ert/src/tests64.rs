extern crate std;

use core::{array, convert::Infallible};

use rv_asm::{Imm, Inst, Reg, Xlen};
use std::vec::Vec;

use crate::{DefaultHandler, ErtError, RawMemory, RvDefaultHandler, ert64_emit, ert64_func};

fn word(value: u64) -> [bool; 64] {
    array::from_fn(|bit| (value >> bit) & 1 != 0)
}

fn value(word: &[bool; 64]) -> u64 {
    word.iter()
        .enumerate()
        .fold(0, |value, (bit, set)| value | ((*set as u64) << bit))
}

fn program(instructions: impl IntoIterator<Item = Inst>) -> Vec<u8> {
    instructions
        .into_iter()
        .flat_map(|instruction| instruction.encode_normal(Xlen::Rv64).to_le_bytes())
        .collect()
}

fn no_hash<C>(_: &mut C, _: &[[bool; 64]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}

fn exit_register(regs: &mut [[bool; 64]; 32], constants: &mut [Option<u64>; 32]) {
    regs[Reg::A0.0 as usize] = word(0xffff_ffff);
    constants[Reg::A0.0 as usize] = Some(0xffff_ffff);
}

fn run(
    mem: &[u8],
    regs: &mut [[bool; 64]; 32],
    constants: &mut [Option<u64>; 32],
    rstack: &mut [u64],
    vstack: &mut [bool],
) -> Result<(), ErtError<Infallible>> {
    let mut handler = RvDefaultHandler {
        inner: DefaultHandler {
            // The native Boolean context is the identity backend, matching the
            // RV32 suite: instruction semantics stay decoupled from IR builders.
            context: (),
            hash: no_hash,
        },
    };
    let storage_bits = vstack.len();
    ert64_emit(
        &mut handler,
        vstack,
        storage_bits,
        RawMemory::from(mem),
        rstack,
        0,
        regs,
        constants,
        false,
        true,
    )
}

fn run_ok(
    mem: &[u8],
    regs: &mut [[bool; 64]; 32],
    constants: &mut [Option<u64>; 32],
    vstack: &mut [bool],
) {
    let mut rstack = [0; 8];
    match run(mem, regs, constants, &mut rstack, vstack) {
        Ok(()) => {}
        Err(ErtError::Decode(_)) => panic!("RV64 program failed to decode"),
        Err(ErtError::Unexpected) => panic!("RV64 program violated the interpreter subset"),
        Err(ErtError::Emitted(error)) => match error {},
    }
}

#[test]
fn rv64_w_forms_sign_extend_concrete_and_symbolic() {
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    // T1 is symbolic: low half 5, with garbage above bit 31 that W-forms
    // must ignore.
    regs[Reg::T1.0 as usize] = word(0xdead_beef_0000_0005);
    regs[Reg::T2.0 as usize] = word(0xffff_ffff_ffff_fffc);
    constants[Reg::T2.0 as usize] = Some(0xffff_ffff_ffff_fffc);
    exit_register(&mut regs, &mut constants);
    let mem = program([
        // addiw t3, t1, -3 -> sext32(5 - 3) = 2
        Inst::AddiW {
            imm: Imm::new_i32(-3),
            dest: Reg::T3,
            src1: Reg::T1,
        },
        // addw t4, t1, t2 -> sext32(5 + (-4)) = 1
        Inst::AddW {
            dest: Reg::T4,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        // subw t5, t1, t2 -> sext32(5 - (-4)) = 9
        Inst::SubW {
            dest: Reg::T5,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        // addiw t6, t2, -1 -> sext32(-4 - 1) = -5 (sign-extended)
        Inst::AddiW {
            imm: Imm::new_i32(-1),
            dest: Reg::T6,
            src1: Reg::T2,
        },
        Inst::Ecall,
    ]);
    let mut vstack = [false; 512];
    run_ok(&mem, &mut regs, &mut constants, &mut vstack);

    assert_eq!(value(&regs[Reg::T3.0 as usize]), 2);
    assert_eq!(constants[Reg::T3.0 as usize], None);
    assert_eq!(value(&regs[Reg::T4.0 as usize]), 1);
    assert_eq!(value(&regs[Reg::T5.0 as usize]), 9);
    assert_eq!(value(&regs[Reg::T6.0 as usize]), (-5i64) as u64);
    assert_eq!(constants[Reg::T6.0 as usize], Some((-5i64) as u64));
}

#[test]
fn rv64_lui_and_auipc_sign_extend_their_u_immediates() {
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    exit_register(&mut regs, &mut constants);
    let mem = program([
        // PC = 0 here.
        Inst::Lui {
            uimm: Imm::new_u32(0x8000_0000),
            dest: Reg::T0,
        },
        Inst::Lui {
            uimm: Imm::new_u32(0x7fff_f000),
            dest: Reg::T1,
        },
        // PC = 8 here: auipc adds the sign-extended U-immediate.
        Inst::Auipc {
            uimm: Imm::new_u32(0xffff_f000),
            dest: Reg::T2,
        },
        Inst::Ecall,
    ]);
    let mut vstack = [false; 64];
    run_ok(&mem, &mut regs, &mut constants, &mut vstack);

    assert_eq!(value(&regs[Reg::T0.0 as usize]), 0xffff_ffff_8000_0000);
    assert_eq!(constants[Reg::T0.0 as usize], Some(0xffff_ffff_8000_0000));
    assert_eq!(value(&regs[Reg::T1.0 as usize]), 0x7fff_f000);
    assert_eq!(
        value(&regs[Reg::T2.0 as usize]),
        0xffff_ffff_ffff_f000u64.wrapping_add(8)
    );
}

#[test]
fn rv64_lw_sign_extends_and_lwu_zero_extends() {
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    regs[Reg::T0.0 as usize] = word(0x8000_0001);
    exit_register(&mut regs, &mut constants);
    let mem = program([
        Inst::Addi {
            imm: Imm::new_i32(-16),
            dest: Reg::SP,
            src1: Reg::SP,
        },
        Inst::Sw {
            offset: Imm::ZERO,
            src: Reg::T0,
            base: Reg::SP,
        },
        Inst::Lw {
            offset: Imm::ZERO,
            dest: Reg::T1,
            base: Reg::SP,
        },
        Inst::Lwu {
            offset: Imm::ZERO,
            dest: Reg::T2,
            base: Reg::SP,
        },
        Inst::Addi {
            imm: Imm::new_i32(16),
            dest: Reg::SP,
            src1: Reg::SP,
        },
        Inst::Ecall,
    ]);
    let mut vstack = [false; 512];
    run_ok(&mem, &mut regs, &mut constants, &mut vstack);

    assert_eq!(value(&regs[Reg::T1.0 as usize]), 0xffff_ffff_8000_0001);
    assert_eq!(value(&regs[Reg::T2.0 as usize]), 0x8000_0001);
}

#[test]
fn rv64_ld_and_sd_roundtrip_the_full_word() {
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    regs[Reg::T0.0 as usize] = word(0x8001_80ff_00ff_ffff);
    exit_register(&mut regs, &mut constants);
    let mem = program([
        Inst::Addi {
            imm: Imm::new_i32(-16),
            dest: Reg::SP,
            src1: Reg::SP,
        },
        Inst::Sd {
            offset: Imm::ZERO,
            src: Reg::T0,
            base: Reg::SP,
        },
        Inst::Ld {
            offset: Imm::ZERO,
            dest: Reg::T1,
            base: Reg::SP,
        },
        // A 32-bit load of the high half observes the true middle bytes.
        Inst::Lw {
            offset: Imm::new_i32(4),
            dest: Reg::T2,
            base: Reg::SP,
        },
        Inst::Addi {
            imm: Imm::new_i32(16),
            dest: Reg::SP,
            src1: Reg::SP,
        },
        Inst::Ecall,
    ]);
    let mut vstack = [false; 512];
    run_ok(&mem, &mut regs, &mut constants, &mut vstack);

    assert_eq!(value(&regs[Reg::T1.0 as usize]), 0x8001_80ff_00ff_ffff);
    assert_eq!(value(&regs[Reg::T2.0 as usize]), 0xffff_ffff_8001_80ff);
}

#[test]
fn rv64_shifts_use_a_six_stage_barrel() {
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    regs[Reg::T1.0 as usize] = word(1);
    // Symbolic shift amounts exercise the barrel stages; 32 and above only
    // exist on RV64.
    regs[Reg::T2.0 as usize] = word(33);
    regs[Reg::T3.0 as usize] = word(0x8000_0000_0000_0001);
    regs[Reg::T4.0 as usize] = word(63);
    exit_register(&mut regs, &mut constants);
    let mem = program([
        Inst::Slli {
            imm: Imm::new_i32(33),
            dest: Reg::T5,
            src1: Reg::T1,
        },
        Inst::Sll {
            dest: Reg::T6,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Sra {
            dest: Reg::A1,
            src1: Reg::T3,
            src2: Reg::T4,
        },
        Inst::Srai {
            imm: Imm::new_i32(63),
            dest: Reg::A2,
            src1: Reg::T3,
        },
        Inst::Srli {
            imm: Imm::new_i32(33),
            dest: Reg::A3,
            src1: Reg::T3,
        },
        Inst::Ecall,
    ]);
    let mut vstack = [false; 256];
    run_ok(&mem, &mut regs, &mut constants, &mut vstack);

    assert_eq!(value(&regs[Reg::T5.0 as usize]), 1u64 << 33);
    assert_eq!(value(&regs[Reg::T6.0 as usize]), 1u64 << 33);
    assert_eq!(
        value(&regs[Reg::A1.0 as usize]),
        (0x8000_0000_0000_0001u64 as i64 >> 63) as u64
    );
    assert_eq!(value(&regs[Reg::A2.0 as usize]), u64::MAX);
    assert_eq!(
        value(&regs[Reg::A3.0 as usize]),
        0x8000_0000_0000_0001u64 >> 33
    );
}

#[test]
fn rv64_w_shifts_operate_on_the_low_half() {
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    regs[Reg::T1.0 as usize] = word(0xffff_ffff_8000_0001);
    regs[Reg::T2.0 as usize] = word(9);
    exit_register(&mut regs, &mut constants);
    let mem = program([
        // slliw: low 0x8000_0001 << 3 -> 0x0000_0008, sign-extended positive.
        Inst::SlliW {
            imm: Imm::new_i32(3),
            dest: Reg::T3,
            src1: Reg::T1,
        },
        // srliw: low 0x8000_0001 >> 9 = 0x0040_0000.
        Inst::SrliW {
            imm: Imm::new_i32(9),
            dest: Reg::T4,
            src1: Reg::T1,
        },
        // sraiw: low 0x8000_0001 >>a 9 = 0xffc0_0000, sign-extended.
        Inst::SraiW {
            imm: Imm::new_i32(9),
            dest: Reg::T5,
            src1: Reg::T1,
        },
        // Register forms, symbolic amount.
        Inst::SllW {
            dest: Reg::T6,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::SraW {
            dest: Reg::A1,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Ecall,
    ]);
    let mut vstack = [false; 256];
    run_ok(&mem, &mut regs, &mut constants, &mut vstack);

    assert_eq!(value(&regs[Reg::T3.0 as usize]), 8);
    assert_eq!(value(&regs[Reg::T4.0 as usize]), 0x0040_0000);
    assert_eq!(value(&regs[Reg::T5.0 as usize]), 0xffff_ffff_ffc0_0000);
    assert_eq!(value(&regs[Reg::T6.0 as usize]), 0x200);
    assert_eq!(value(&regs[Reg::A1.0 as usize]), 0xffff_ffff_ffc0_0000);
}

#[test]
fn rv64_multiplication_matches_native_128_bit_products() {
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    // T1 stays symbolic so the long-multiplication path runs.
    regs[Reg::T1.0 as usize] = word(0x1234_5678_9abc_def0);
    regs[Reg::T2.0 as usize] = word(0x0fed_cba9_8765_4321);
    constants[Reg::T2.0 as usize] = Some(0x0fed_cba9_8765_4321);
    regs[Reg::T3.0 as usize] = word(7);
    exit_register(&mut regs, &mut constants);
    let mem = program([
        Inst::Mul {
            dest: Reg::T4,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Mulh {
            dest: Reg::T5,
            src1: Reg::T1,
            src2: Reg::T3,
        },
        Inst::Mulhu {
            dest: Reg::T6,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Mulhsu {
            dest: Reg::A1,
            src1: Reg::T3,
            src2: Reg::T2,
        },
        Inst::Ecall,
    ]);
    let mut vstack = [false; 256];
    run_ok(&mem, &mut regs, &mut constants, &mut vstack);

    let left = 0x1234_5678_9abc_def0u64;
    let right = 0x0fed_cba9_8765_4321u64;
    assert_eq!(value(&regs[Reg::T4.0 as usize]), left.wrapping_mul(right));
    assert_eq!(
        value(&regs[Reg::T5.0 as usize]),
        ((left as i64 as i128 * 7i128) >> 64) as u64
    );
    assert_eq!(
        value(&regs[Reg::T6.0 as usize]),
        ((left as u128 * right as u128) >> 64) as u64
    );
    assert_eq!(
        value(&regs[Reg::A1.0 as usize]),
        ((7i128 * (right as u128) as i128) >> 64) as u64
    );
}

#[test]
fn symbolic_sub_subtracts_src2_from_src1() {
    // Pins the historical operand mix-up: the symbolic subtract path once
    // computed `src2 - src1`.
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    regs[Reg::T1.0 as usize] = word(100);
    regs[Reg::T2.0 as usize] = word(58);
    exit_register(&mut regs, &mut constants);
    let mem = program([
        Inst::Sub {
            dest: Reg::T0,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Ecall,
    ]);
    let mut vstack = [false; 64];
    run_ok(&mem, &mut regs, &mut constants, &mut vstack);

    assert_eq!(value(&regs[Reg::T0.0 as usize]), 42);
}

#[test]
fn rv64_branches_compare_the_full_word() {
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    regs[Reg::T1.0 as usize] = word(1);
    constants[Reg::T1.0 as usize] = Some(1);
    regs[Reg::T2.0 as usize] = word(0x1_0000_0001);
    constants[Reg::T2.0 as usize] = Some(0x1_0000_0001);
    regs[Reg::T3.0 as usize] = word(u64::MAX);
    constants[Reg::T3.0 as usize] = Some(u64::MAX);
    exit_register(&mut regs, &mut constants);
    let mem = program([
        // Low halves are equal; only bit 32 differs, so BEQ must not branch.
        Inst::Beq {
            offset: Imm::new_i32(8),
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Addi {
            imm: Imm::new_i32(1),
            dest: Reg::T4,
            src1: Reg::ZERO,
        },
        // BLT signed: 1 < -1 is false, so skip the next instruction.
        Inst::Blt {
            offset: Imm::new_i32(8),
            src1: Reg::T1,
            src2: Reg::T3,
        },
        Inst::Addi {
            imm: Imm::new_i32(2),
            dest: Reg::T5,
            src1: Reg::ZERO,
        },
        // BLTU signed-negative reads unsigned: 1 < 2^64-1 is true.
        Inst::Bltu {
            offset: Imm::new_i32(8),
            src1: Reg::T1,
            src2: Reg::T3,
        },
        Inst::Addi {
            imm: Imm::new_i32(4),
            dest: Reg::T6,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
    ]);
    let mut vstack = [false; 64];
    run_ok(&mem, &mut regs, &mut constants, &mut vstack);

    assert_eq!(value(&regs[Reg::T4.0 as usize]), 1);
    assert_eq!(value(&regs[Reg::T5.0 as usize]), 2);
    assert_eq!(value(&regs[Reg::T6.0 as usize]), 0);
}

#[test]
fn rv64_hash_ecall_packs_four_64_bit_registers() {
    let hash = |_: &mut (), _: &[[bool; 64]]| -> Result<[u8; 32], Infallible> {
        Ok(array::from_fn(|index| index as u8))
    };
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    regs[Reg::T0.0 as usize] = word(0xdead_beef);
    exit_register(&mut regs, &mut constants);
    let mem = program([
        Inst::Addi {
            imm: Imm::ZERO,
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
        Inst::Addi {
            imm: Imm::new_i32(-1),
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
    ]);
    let mut handler = RvDefaultHandler {
        inner: DefaultHandler {
            context: (),
            hash,
        },
    };
    let mut vstack = [false; 64];
    let storage_bits = vstack.len();
    let mut rstack = [0; 8];
    assert!(
        ert64_emit(
            &mut handler,
            &mut vstack,
            storage_bits,
            RawMemory::from(&mem[..]),
            &mut rstack,
            0,
            &mut regs,
            &mut constants,
            false,
            true,
        )
        .is_ok()
    );

    assert_eq!(value(&regs[Reg::A1.0 as usize]), 0x0706_0504_0302_0100);
    assert_eq!(value(&regs[Reg::A2.0 as usize]), 0x0f0e_0d0c_0b0a_0908);
    assert_eq!(value(&regs[Reg::A3.0 as usize]), 0x1716_1514_1312_1110);
    assert_eq!(value(&regs[Reg::A4.0 as usize]), 0x1f1e_1d1c_1b1a_1918);
    assert_eq!(constants[Reg::A1.0 as usize], Some(0x0706_0504_0302_0100));
}

#[test]
fn rv64_abi_passes_stack_arguments_and_results() {
    // Ten 64-bit arguments: eight in registers, two in caller stack slots.
    // Nine results: eight registers plus one caller stack slot.
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    let mut rstack = [0; 8];
    let mut vstack = [false; 4096];
    let mem = program([
        // t2 = a0 + [sp] + [sp+8]
        Inst::Ld {
            offset: Imm::ZERO,
            dest: Reg::T0,
            base: Reg::SP,
        },
        Inst::Add {
            dest: Reg::T2,
            src1: Reg::A0,
            src2: Reg::T0,
        },
        Inst::Ld {
            offset: Imm::new_i32(8),
            dest: Reg::T1,
            base: Reg::SP,
        },
        Inst::Add {
            dest: Reg::T2,
            src1: Reg::T2,
            src2: Reg::T1,
        },
        // a1 carries the register result; a0 keeps the exit selector.
        Inst::Addi {
            imm: Imm::ZERO,
            dest: Reg::A1,
            src1: Reg::T2,
        },
        // The stack result slot aliases the first stack argument slot, whose
        // value has already been consumed.
        Inst::Addi {
            imm: Imm::new_i32(7),
            dest: Reg::T3,
            src1: Reg::T2,
        },
        Inst::Sd {
            offset: Imm::ZERO,
            src: Reg::T3,
            base: Reg::SP,
        },
        Inst::Addi {
            imm: Imm::new_i32(-1),
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
    ]);
    // a0 starts as the first argument (1); the program sums the two stack
    // arguments (9 and 10) alongside it and installs the exit selector
    // itself before the final ecall.
    let args: [([bool; 64], Option<u64>); 10] =
        array::from_fn(|index| (word((index + 1) as u64), Some((index + 1) as u64)));
    let mut handler = RvDefaultHandler {
        inner: DefaultHandler {
            context: (),
            hash: no_hash,
        },
    };
    let storage_bits = vstack.len();
    let results = ert64_func::<_, _, 10, 9, _>(
        &mut handler,
        &mut vstack,
        storage_bits,
        RawMemory::from(&mem[..]),
        &mut rstack,
        0,
        &mut regs,
        &mut constants,
        false,
        true,
        args,
    );
    let results = match results {
        Ok(results) => results,
        Err(ErtError::Decode(_)) => panic!("RV64 ABI program failed to decode"),
        Err(ErtError::Unexpected) => panic!("RV64 ABI program violated the interpreter subset"),
        Err(ErtError::Emitted(error)) => match error {},
    };

    assert_eq!(results[0].1, Some(u64::MAX));
    assert_eq!(results[1].0, word(20));
    assert_eq!(value(&results[8].0), 27);
}

// --- Compressed-instruction encodings (hand-assembled from the C-format
// field tables, cross-checked against rv-asm's decoder in the tests). ---

fn c_li(rd: u8, imm6: i8) -> u16 {
    let imm = imm6 as u16 & 0x3f;
    (0b010 << 13) | (((imm >> 5) & 1) << 12) | ((rd as u16) << 7) | ((imm & 0x1f) << 2) | 0b01
}

fn c_addi16sp(imm10: i16) -> u16 {
    let imm = imm10 as u16 & 0x3ff;
    let fields = ((imm >> 4) & 1) << 6
        | ((imm >> 6) & 1) << 5
        | ((imm >> 7) & 0b11) << 3
        | ((imm >> 5) & 1) << 2
        | ((imm >> 9) & 1) << 12;
    (0b011 << 13) | (2 << 7) | fields | 0b01
}

fn c_mv(rd: u8, rs2: u8) -> u16 {
    (0b1000 << 12) | ((rd as u16) << 7) | ((rs2 as u16) << 2) | 0b10
}

fn c_add(rd: u8, rs2: u8) -> u16 {
    (0b1001 << 12) | ((rd as u16) << 7) | ((rs2 as u16) << 2) | 0b10
}

fn c_jr(rs1: u8) -> u16 {
    (0b1000 << 12) | ((rs1 as u16) << 7) | 0b10
}

fn c_jalr(rs1: u8) -> u16 {
    (0b1001 << 12) | ((rs1 as u16) << 7) | 0b10
}

fn c_lwsp(rd: u8, uimm8: u16) -> u16 {
    let imm = uimm8 & 0xff;
    let fields = ((imm >> 5) & 1) << 12 | ((imm >> 2) & 0b111) << 4 | ((imm >> 6) & 0b11) << 2;
    (0b010 << 13) | ((rd as u16) << 7) | fields | 0b10
}

fn c_lw(rd: u8, rs1: u8, uimm7: u16) -> u16 {
    // CL format, llvm-mc-verified: uimm[5:3]@[12:10], uimm[2]@[6],
    // uimm[6]@[5]; compressed register fields are relative to x8.
    let imm = uimm7 & 0x7f;
    let fields = ((imm >> 3) & 0b111) << 10 | ((imm >> 2) & 1) << 6 | ((imm >> 6) & 1) << 5;
    (0b010 << 13) | fields | (((rs1 - 8) as u16) << 7) | (((rd - 8) as u16) << 2) | 0b00
}

fn c_beqz(rs1: u8, simm9: i16) -> u16 {
    let imm = simm9 as u16 & 0x1ff;
    // CB format, llvm-mc-verified: imm[8]@[12], imm[4:3]@[11:10],
    // imm[7:6]@[6:5], imm[2:1]@[4:3], imm[5]@[2]; the compressed register
    // field is relative to x8.
    let fields = ((imm >> 8) & 1) << 12
        | ((imm >> 3) & 0b11) << 10
        | ((imm >> 6) & 0b11) << 5
        | ((imm >> 1) & 0b11) << 3
        | ((imm >> 5) & 1) << 2;
    (0b110 << 13) | fields | (((rs1 - 8) as u16) << 7) | 0b01
}

fn c_swsp(rs2: u8, uimm8: u16) -> u16 {
    let imm = uimm8 & 0xff;
    let fields = ((imm >> 6) & 0b11) << 7 | ((imm >> 2) & 0b1111) << 9;
    (0b110 << 13) | fields | ((rs2 as u16) << 2) | 0b10
}

fn push16(mem: &mut Vec<u8>, code: u16) {
    mem.extend(code.to_le_bytes());
}

fn push32(mem: &mut Vec<u8>, instruction: Inst) {
    mem.extend(instruction.encode_normal(Xlen::Rv64).to_le_bytes());
}

const A0: u8 = 10;
const A1: u8 = 11;
const A2: u8 = 12;
const RA: u8 = 1;

#[test]
fn compressed_encodings_roundtrip_through_the_decoder() {
    assert_eq!(
        Inst::decode_compressed(c_li(A1, 5), Xlen::Rv64).unwrap(),
        Inst::Addi {
            imm: Imm::new_i32(5),
            dest: Reg(A1),
            src1: Reg::ZERO,
        }
    );
    assert_eq!(
        Inst::decode_compressed(c_addi16sp(-128), Xlen::Rv64).unwrap(),
        Inst::Addi {
            imm: Imm::new_i32(-128),
            dest: Reg::SP,
            src1: Reg::SP,
        }
    );
    assert_eq!(
        Inst::decode_compressed(c_mv(A1, A0), Xlen::Rv64).unwrap(),
        Inst::Add {
            dest: Reg(A1),
            src1: Reg::ZERO,
            src2: Reg(A0),
        }
    );
    assert_eq!(
        Inst::decode_compressed(c_add(A0, A1), Xlen::Rv64).unwrap(),
        Inst::Add {
            dest: Reg(A0),
            src1: Reg(A0),
            src2: Reg(A1),
        }
    );
    assert_eq!(
        Inst::decode_compressed(c_jalr(RA), Xlen::Rv64).unwrap(),
        Inst::Jalr {
            offset: Imm::ZERO,
            base: Reg::RA,
            dest: Reg::RA,
        }
    );
    assert_eq!(
        Inst::decode_compressed(c_lwsp(A1, 32), Xlen::Rv64).unwrap(),
        Inst::Lw {
            offset: Imm::new_u32(32),
            dest: Reg(A1),
            base: Reg::SP,
        }
    );
    assert_eq!(
        Inst::decode_compressed(c_beqz(A0, 8), Xlen::Rv64).unwrap(),
        Inst::Beq {
            offset: Imm::new_i32(8),
            src1: Reg(A0),
            src2: Reg::ZERO,
        }
    );
    assert_eq!(
        Inst::decode_compressed(c_swsp(A1, 16), Xlen::Rv64).unwrap(),
        Inst::Sw {
            offset: Imm::new_u32(16),
            src: Reg(A1),
            base: Reg::SP,
        }
    );
}

#[test]
fn compressed_and_normal_instructions_mix_across_calls_and_the_stack() {
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    exit_register(&mut regs, &mut constants);
    let mut mem: Vec<u8> = Vec::new();

    // Control-flow trace (compressed steps advance by 2; the callee is out
    // of line, past the mainline's exit):
    //   0: c.li a1, 5
    //   2: jal ra, +28 -> 30, ra = 6
    //   6: c.addi16sp sp, -16     (return landing)
    //   8: c.swsp a1, 8
    //  10: c.lwsp a2, 8
    //  12: sw a1, 4(sp)           (normal 4-byte instruction)
    //  16: addi s0, sp, 0         (compressed bases must be x8..x15)
    //  20: c.lw a5, 4(s0)
    //  22: c.beqz a0, +4          (a0 = exit selector, nonzero: not taken)
    //  24: c.addi16sp sp, 16
    //  26: ecall                  (sp balanced: exits)
    //  30: callee: c.add a1, a0   (a1 = 5 + 0xffff_ffff)
    //  32: c.jalr ra              (link-less return to 6)
    push16(&mut mem, c_li(A1, 5));
    push32(
        &mut mem,
        Inst::Jal {
            offset: Imm::new_i32(28),
            dest: Reg::RA,
        },
    );
    push16(&mut mem, c_addi16sp(-16));
    push16(&mut mem, c_swsp(A1, 8));
    push16(&mut mem, c_lwsp(A2, 8));
    push32(
        &mut mem,
        Inst::Sw {
            offset: Imm::new_i32(4),
            src: Reg(A1),
            base: Reg::SP,
        },
    );
    push32(
        &mut mem,
        Inst::Addi {
            imm: Imm::ZERO,
            dest: Reg::S0,
            src1: Reg::SP,
        },
    );
    push16(&mut mem, c_lw(15, 8, 4));
    push16(&mut mem, c_beqz(A0, 4));
    push16(&mut mem, c_addi16sp(16));
    push32(&mut mem, Inst::Ecall);
    push16(&mut mem, c_add(A1, A0));
    push16(&mut mem, c_jalr(RA));

    let mut vstack = [false; 512];
    run_ok(&mem, &mut regs, &mut constants, &mut vstack);

    // a1 keeps the full 64-bit sum through the call; the stack slots roundtrip
    // its low 32 bits (`c.swsp`/`sw`/`c.lwsp`/`c.lw` are word-width).
    let expected = 0x1_0000_0004u64;
    assert_eq!(value(&regs[A1 as usize]), expected);
    assert_eq!(value(&regs[A2 as usize]), 4);
    assert_eq!(value(&regs[15]), 4);
}

#[test]
fn compressed_step_by_step() {
    // Each stage ends with the exit ecall; a failure's stage number names the
    // first instruction that breaks.
    let c_add_code = c_add(A1, A0);
    let c_jalr_code = c_jalr(RA);
    let stages: std::vec::Vec<(&str, Vec<u8>)> = [
        ("c.li", {
            let mut m = Vec::new();
            push16(&mut m, c_li(A1, 5));
            push32(&mut m, Inst::Ecall);
            m
        }),
        ("jal-alone", {
            let mut m = Vec::new();
            push32(&mut m, Inst::Jal { offset: Imm::new_i32(8), dest: Reg::RA });
            push32(&mut m, Inst::Ecall);
            push32(&mut m, Inst::Ecall);
            m
        }),
        ("jal+c.add", {
            let mut m = Vec::new();
            push32(&mut m, Inst::Jal { offset: Imm::new_i32(8), dest: Reg::RA });
            push32(&mut m, Inst::Ecall);
            push16(&mut m, c_add_code);
            push16(&mut m, c_jalr_code);
            m
        }),
        ("c.addi16sp", {
            let mut m = Vec::new();
            push16(&mut m, c_addi16sp(-16));
            push16(&mut m, c_addi16sp(16));
            push32(&mut m, Inst::Ecall);
            m
        }),
        ("c.swsp+c.lwsp", {
            let mut m = Vec::new();
            push16(&mut m, c_addi16sp(-16));
            push16(&mut m, c_li(A1, 5));
            push16(&mut m, c_swsp(A1, 8));
            push16(&mut m, c_lwsp(A2, 8));
            push16(&mut m, c_addi16sp(16));
            push32(&mut m, Inst::Ecall);
            m
        }),
        ("c.beqz-not-taken", {
            let mut m = Vec::new();
            push16(&mut m, c_beqz(A0, 4));
            push32(&mut m, Inst::Ecall);
            m
        }),
    ]
    .into_iter()
    .collect();
    for (name, mem) in stages {
        let mut regs = [[false; 64]; 32];
        let mut constants = [None; 32];
        exit_register(&mut regs, &mut constants);
        let mut vstack = [false; 512];
        let mut rstack = [0; 8];
        let result = run(&mem, &mut regs, &mut constants, &mut rstack, &mut vstack);
        assert!(result.is_ok(), "stage {name} failed");
    }
}

