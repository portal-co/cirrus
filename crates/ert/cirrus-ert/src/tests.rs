extern crate std;

use core::{array, convert::Infallible};

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithStorage,
    ContextWithValue, HasError, StorageAddressBit,
};
use rv_asm::{Imm, Inst, Reg, Xlen};
use std::vec::Vec;

use crate::{
    ert_emit, ert_func, simple_add, DefaultHandler, ErtError, RawMemory, RvDefaultHandler,
};

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

fn bounded_memory(mem: &[u8]) -> RawMemory<'_> {
    RawMemory::from(mem)
}

fn run(
    mem: &[u8],
    regs: &mut [[bool; 32]; 32],
    constants: &mut [Option<u32>; 32],
    rstack: &mut [u32],
    vstack: &mut [bool],
) -> Result<(), ErtError<Infallible>> {
    let mut handler = RvDefaultHandler {
        inner: DefaultHandler {
            // The native Boolean context is the identity backend. Keeping the
            // facade tests here avoids coupling instruction semantics to the
            // Volar IR builder.
            context: (),
            hash: no_hash,
        },
    };
    let storage_bits = vstack.len();
    ert_emit(
        &mut handler,
        vstack,
        storage_bits,
        bounded_memory(mem),
        rstack,
        0,
        regs,
        constants,
        false,
        true,
    )
}

fn no_hash<C>(_: &mut C, _: &[[bool; 32]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}

fn assert_success(result: Result<(), ErtError<Infallible>>) {
    assert!(result.is_ok());
}

#[test]
fn unbounded_raw_memory_accepts_an_address_zero_base() {
    // SAFETY: this test only constructs the mapping; it never reads through it.
    let _ = unsafe { RawMemory::new(core::ptr::null(), None) };
}

#[test]
fn a_slice_converts_to_a_bounded_raw_memory_mapping() {
    let bytes = [1, 2, 3, 4];
    let memory = RawMemory::from(&bytes[..]);

    assert_eq!(memory.read::<4>(0), Some(bytes));
    assert_eq!(memory.read::<1>(4), None);
}

#[test]
fn bounded_raw_memory_rejects_a_truncated_instruction_fetch() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];

    // A null compressed halfword is a decode failure; a one-byte image is a
    // truncated fetch.
    assert!(matches!(
        run(&[0; 3], &mut regs, &mut constants, &mut rstack, &mut vstack),
        Err(ErtError::Decode(_))
    ));
    assert!(matches!(
        run(&[0x13], &mut regs, &mut constants, &mut rstack, &mut vstack),
        Err(ErtError::Unexpected)
    ));
}

#[test]
fn bounded_raw_memory_rejects_a_concrete_load_past_its_end() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    regs[Reg::A1.0 as usize] = word(12);
    constants[Reg::A1.0 as usize] = Some(12);
    let mem = program([
        Inst::Lw {
            offset: Imm::new_i32(0),
            dest: Reg::T0,
            base: Reg::A1,
        },
        Inst::Addi {
            imm: Imm::new_i32(-1),
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
    ]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];

    assert!(matches!(
        run(&mem, &mut regs, &mut constants, &mut rstack, &mut vstack),
        Err(ErtError::Unexpected)
    ));
}

#[derive(Default)]
struct CountingContext {
    bitand: usize,
    bitor: usize,
    bitxor: usize,
}

impl HasError for CountingContext {
    type Error = Infallible;
}

impl ContextWithValue<bool> for CountingContext {
    type Wrapped = bool;
}

impl ContextWithCreate<bool> for CountingContext {
    fn create(&mut self, value: bool) -> Result<bool, Self::Error> {
        Ok(value)
    }
}

impl ContextWithBitAnd<bool> for CountingContext {
    fn bitand(&mut self, left: bool, right: bool) -> Result<bool, Self::Error> {
        self.bitand += 1;
        Ok(left & right)
    }

    fn bitand_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Self::Error> {
        self.bitand += 1;
        *left &= right;
        Ok(())
    }
}

impl ContextWithBitOr<bool> for CountingContext {
    fn bitor(&mut self, left: bool, right: bool) -> Result<bool, Self::Error> {
        self.bitor += 1;
        Ok(left | right)
    }

    fn bitor_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Self::Error> {
        self.bitor += 1;
        *left |= right;
        Ok(())
    }
}

impl ContextWithBitXor<bool> for CountingContext {
    fn bitxor(&mut self, left: bool, right: bool) -> Result<bool, Self::Error> {
        self.bitxor += 1;
        Ok(left ^ right)
    }

    fn bitxor_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Self::Error> {
        self.bitxor += 1;
        *left ^= right;
        Ok(())
    }
}

impl ContextWithStorage<bool> for CountingContext {
    type Storage = [bool];

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<bool>],
    ) -> Result<bool, Self::Error> {
        Ok(storage[storage_index(address)])
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<bool>],
        value: bool,
    ) -> Result<(), Self::Error> {
        storage[storage_index(address)] = value;
        Ok(())
    }
}

fn storage_index(address: &[StorageAddressBit<bool>]) -> usize {
    address
        .iter()
        .enumerate()
        .fold(0usize, |index, (bit, address)| {
            index | ((address.wire as usize) << bit)
        })
}

fn run_counting(
    instructions: impl IntoIterator<Item = Inst>,
    regs: &mut [[bool; 32]; 32],
    constants: &mut [Option<u32>; 32],
) -> CountingContext {
    let mem = program(instructions);
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];
    let mut handler = RvDefaultHandler {
        inner: DefaultHandler {
            context: CountingContext::default(),
            hash: no_hash,
        },
    };
    let storage_bits = vstack.len();
    assert_success(ert_emit(
        &mut handler,
        &mut vstack,
        storage_bits,
        bounded_memory(&mem),
        &mut rstack,
        0,
        regs,
        constants,
        false,
        true,
    ));
    handler.inner.context
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
    let mem = program([
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
        &mem,
        &mut regs,
        &mut constants,
        &mut rstack,
        &mut vstack,
    ));

    assert_eq!(constants[Reg::T0.0 as usize], Some(0x8000_0006));
    assert_eq!(constants[Reg::T3.0 as usize], Some(6));
    assert_eq!(constants[Reg::T4.0 as usize], Some(3));
    assert_eq!(value(&regs[Reg::T4.0 as usize]), 3);
    assert_eq!(constants[Reg::T5.0 as usize], Some(5));
    assert_eq!(constants[Reg::T2.0 as usize], Some(10));
    assert_eq!(constants[Reg::T1.0 as usize], Some(0x4000_0001));
    assert_eq!(constants[Reg::T6.0 as usize], Some(u32::MAX));
    assert_eq!(value(&regs[Reg::T2.0 as usize]), 10);
    assert_eq!(value(&regs[Reg::T1.0 as usize]), 0x4000_0001);
    assert_eq!(value(&regs[Reg::T6.0 as usize]), u32::MAX);
}

#[test]
fn slt_forms_materialize_signed_and_unsigned_booleans() {
    let instructions = [
        Inst::Slt {
            dest: Reg::T0,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Sltu {
            dest: Reg::T3,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Slti {
            imm: Imm::new_i32(0),
            dest: Reg::T4,
            src1: Reg::T1,
        },
        Inst::Sltiu {
            imm: Imm::new_i32(-1),
            dest: Reg::T5,
            src1: Reg::T1,
        },
        Inst::Sltiu {
            imm: Imm::new_i32(-1),
            dest: Reg::T6,
            src1: Reg::T2,
        },
        Inst::Ecall,
    ];

    let mut concrete_regs = [[false; 32]; 32];
    let mut concrete_constants = [None; 32];
    concrete_regs[Reg::T1.0 as usize] = word(0x8000_0000);
    concrete_regs[Reg::T2.0 as usize] = word(0);
    concrete_constants[Reg::T1.0 as usize] = Some(0x8000_0000);
    concrete_constants[Reg::T2.0 as usize] = Some(0);
    exit_register(&mut concrete_regs, &mut concrete_constants);
    let concrete_mem = program(instructions);
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];
    assert_success(run(
        &concrete_mem,
        &mut concrete_regs,
        &mut concrete_constants,
        &mut rstack,
        &mut vstack,
    ));
    for (register, expected) in [
        (Reg::T0, 1),
        (Reg::T3, 0),
        (Reg::T4, 1),
        (Reg::T5, 1),
        (Reg::T6, 1),
    ] {
        assert_eq!(concrete_constants[register.0 as usize], Some(expected));
        assert_eq!(value(&concrete_regs[register.0 as usize]), expected);
    }

    let mut symbolic_regs = [[false; 32]; 32];
    let mut symbolic_constants = [None; 32];
    symbolic_regs[Reg::T1.0 as usize] = word(0x8000_0000);
    symbolic_regs[Reg::T2.0 as usize] = word(0);
    exit_register(&mut symbolic_regs, &mut symbolic_constants);
    let symbolic_mem = program(instructions);
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];
    assert_success(run(
        &symbolic_mem,
        &mut symbolic_regs,
        &mut symbolic_constants,
        &mut rstack,
        &mut vstack,
    ));
    for (register, expected) in [
        (Reg::T0, 1),
        (Reg::T3, 0),
        (Reg::T4, 1),
        (Reg::T5, 1),
        (Reg::T6, 1),
    ] {
        assert_eq!(value(&symbolic_regs[register.0 as usize]), expected);
        assert_eq!(symbolic_constants[register.0 as usize], None);
        assert_eq!(symbolic_regs[register.0 as usize][0], expected != 0);
        assert!(symbolic_regs[register.0 as usize][1..]
            .iter()
            .all(|bit| !bit));
    }
}

#[test]
fn runtime_shifts_use_low_five_bits_and_snapshot_aliased_sources() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    regs[Reg::T1.0 as usize] = word(0x4000_0001);
    regs[Reg::T2.0 as usize] = word(33);
    regs[Reg::T3.0 as usize] = word(0x8000_0003);
    regs[Reg::T4.0 as usize] = word(0x8000_0003);
    regs[Reg::T5.0 as usize] = word(1);
    exit_register(&mut regs, &mut constants);
    let mem = program([
        Inst::Sll {
            dest: Reg::T1,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Srl {
            dest: Reg::T2,
            src1: Reg::T3,
            src2: Reg::T2,
        },
        Inst::Sra {
            dest: Reg::T4,
            src1: Reg::T4,
            src2: Reg::T5,
        },
        Inst::Ecall,
    ]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];

    assert_success(run(
        &mem,
        &mut regs,
        &mut constants,
        &mut rstack,
        &mut vstack,
    ));

    assert_eq!(value(&regs[Reg::T1.0 as usize]), 0x8000_0002);
    assert_eq!(value(&regs[Reg::T2.0 as usize]), 0x4000_0001);
    assert_eq!(value(&regs[Reg::T4.0 as usize]), 0xc000_0001);
    assert!(constants[Reg::T1.0 as usize].is_none());
    assert!(constants[Reg::T2.0 as usize].is_none());
    assert!(constants[Reg::T4.0 as usize].is_none());
}

#[test]
fn concrete_shift_amounts_avoid_barrel_selector_gates() {
    let instructions = [
        Inst::Sll {
            dest: Reg::T0,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Ecall,
    ];
    let mut symbolic_regs = [[false; 32]; 32];
    let mut symbolic_constants = [None; 32];
    symbolic_regs[Reg::T1.0 as usize] = word(0x0123_4567);
    symbolic_regs[Reg::T2.0 as usize] = word(1);
    exit_register(&mut symbolic_regs, &mut symbolic_constants);
    let symbolic = run_counting(instructions, &mut symbolic_regs, &mut symbolic_constants);

    let mut constant_regs = [[false; 32]; 32];
    let mut constant_metadata = [None; 32];
    constant_regs[Reg::T1.0 as usize] = word(0x0123_4567);
    constant_regs[Reg::T2.0 as usize] = word(33);
    constant_metadata[Reg::T1.0 as usize] = Some(0x0123_4567);
    constant_metadata[Reg::T2.0 as usize] = Some(33);
    exit_register(&mut constant_regs, &mut constant_metadata);
    let constant = run_counting(instructions, &mut constant_regs, &mut constant_metadata);

    assert_eq!(symbolic.bitand, 160);
    assert_eq!(symbolic.bitxor, 320);
    assert_eq!(symbolic.bitor, 0);
    assert_eq!(constant.bitand, 0);
    assert_eq!(constant.bitxor, 0);
    assert_eq!(constant.bitor, 0);
    assert_eq!(value(&constant_regs[Reg::T0.0 as usize]), 0x0246_8ace);
    assert_eq!(constant_metadata[Reg::T0.0 as usize], Some(0x0246_8ace));
}

#[test]
fn a_constant_and_or_operand_avoids_bitwise_gates() {
    let instructions = [
        Inst::And {
            dest: Reg::T0,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Or {
            dest: Reg::T3,
            src1: Reg::T2,
            src2: Reg::T1,
        },
        Inst::Ecall,
    ];
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    let t1 = 0b1010_1100_1111_0000_0000_1111_0011_0101u32;
    regs[Reg::T1.0 as usize] = word(t1);
    regs[Reg::T2.0 as usize] = word(0x0f0f_0f0f);
    constants[Reg::T2.0 as usize] = Some(0x0f0f_0f0f);
    exit_register(&mut regs, &mut constants);
    let counts = run_counting(instructions, &mut regs, &mut constants);

    assert_eq!(counts.bitand, 0);
    assert_eq!(counts.bitor, 0);
    assert_eq!(value(&regs[Reg::T0.0 as usize]), t1 & 0x0f0f_0f0f);
    assert_eq!(constants[Reg::T0.0 as usize], None);
    assert_eq!(value(&regs[Reg::T3.0 as usize]), t1 | 0x0f0f_0f0f);
    assert_eq!(constants[Reg::T3.0 as usize], None);
}

#[test]
fn a_degenerate_and_or_mask_folds_to_a_full_constant() {
    let instructions = [
        Inst::And {
            dest: Reg::T0,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Or {
            dest: Reg::T3,
            src1: Reg::T1,
            src2: Reg::T4,
        },
        Inst::Ecall,
    ];
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    regs[Reg::T1.0 as usize] = word(0x1234_5678);
    regs[Reg::T2.0 as usize] = word(0);
    constants[Reg::T2.0 as usize] = Some(0);
    regs[Reg::T4.0 as usize] = word(u32::MAX);
    constants[Reg::T4.0 as usize] = Some(u32::MAX);
    exit_register(&mut regs, &mut constants);
    let counts = run_counting(instructions, &mut regs, &mut constants);

    assert_eq!(counts.bitand, 0);
    assert_eq!(counts.bitor, 0);
    assert_eq!(value(&regs[Reg::T0.0 as usize]), 0);
    assert_eq!(constants[Reg::T0.0 as usize], Some(0));
    assert_eq!(value(&regs[Reg::T3.0 as usize]), u32::MAX);
    assert_eq!(constants[Reg::T3.0 as usize], Some(u32::MAX));
}

fn signed_high(left: u32, right: u32) -> u32 {
    (((left as i32 as i64) * (right as i32 as i64)) >> 32) as u32
}

fn signed_unsigned_high(left: u32, right: u32) -> u32 {
    (((left as i32 as i64) * (right as u64 as i64)) >> 32) as u32
}

fn unsigned_high(left: u32, right: u32) -> u32 {
    ((left as u64 * right as u64) >> 32) as u32
}

#[test]
fn symbolic_multiplication_supports_low_and_high_product_variants() {
    let left = 0x8000_0000;
    let right = 0xffff_fffd;
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    regs[Reg::T1.0 as usize] = word(left);
    regs[Reg::T2.0 as usize] = word(right);
    exit_register(&mut regs, &mut constants);
    let mem = program([
        Inst::Mul {
            dest: Reg::T0,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Mulh {
            dest: Reg::T3,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Mulhsu {
            dest: Reg::T4,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Mulhu {
            dest: Reg::T5,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Ecall,
    ]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];

    assert_success(run(
        &mem,
        &mut regs,
        &mut constants,
        &mut rstack,
        &mut vstack,
    ));

    assert_eq!(value(&regs[Reg::T0.0 as usize]), left.wrapping_mul(right));
    assert_eq!(value(&regs[Reg::T3.0 as usize]), signed_high(left, right));
    assert_eq!(
        value(&regs[Reg::T4.0 as usize]),
        signed_unsigned_high(left, right)
    );
    assert_eq!(value(&regs[Reg::T5.0 as usize]), unsigned_high(left, right));
    for register in [Reg::T0, Reg::T3, Reg::T4, Reg::T5] {
        assert!(constants[register.0 as usize].is_none());
    }
}

#[test]
fn high_products_apply_each_required_signed_correction() {
    for (left, right) in [
        (0x7fff_fffe, 0x8000_0003),
        (0x8000_0002, 5),
        (0x1234_5678, 0x1020_3040),
    ] {
        let mut regs = [[false; 32]; 32];
        let mut constants = [None; 32];
        regs[Reg::T1.0 as usize] = word(left);
        regs[Reg::T2.0 as usize] = word(right);
        exit_register(&mut regs, &mut constants);
        let mem = program([
            Inst::Mulh {
                dest: Reg::T3,
                src1: Reg::T1,
                src2: Reg::T2,
            },
            Inst::Mulhsu {
                dest: Reg::T4,
                src1: Reg::T1,
                src2: Reg::T2,
            },
            Inst::Mulhu {
                dest: Reg::T5,
                src1: Reg::T1,
                src2: Reg::T2,
            },
            Inst::Ecall,
        ]);
        let mut rstack = [0; 8];
        let mut vstack = [false; 64];

        assert_success(run(
            &mem,
            &mut regs,
            &mut constants,
            &mut rstack,
            &mut vstack,
        ));

        assert_eq!(value(&regs[Reg::T3.0 as usize]), signed_high(left, right));
        assert_eq!(
            value(&regs[Reg::T4.0 as usize]),
            signed_unsigned_high(left, right)
        );
        assert_eq!(value(&regs[Reg::T5.0 as usize]), unsigned_high(left, right));
    }
}

#[test]
fn multiplication_snapshots_aliased_operands() {
    let left = 0x1234_5678;
    let right = 0x0001_0003;
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    regs[Reg::T1.0 as usize] = word(left);
    regs[Reg::T2.0 as usize] = word(right);
    regs[Reg::T3.0 as usize] = word(left);
    exit_register(&mut regs, &mut constants);
    let mem = program([
        Inst::Mul {
            dest: Reg::T1,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Mulh {
            dest: Reg::T2,
            src1: Reg::T3,
            src2: Reg::T2,
        },
        Inst::Ecall,
    ]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];

    assert_success(run(
        &mem,
        &mut regs,
        &mut constants,
        &mut rstack,
        &mut vstack,
    ));

    assert_eq!(value(&regs[Reg::T1.0 as usize]), left.wrapping_mul(right));
    assert_eq!(value(&regs[Reg::T2.0 as usize]), signed_high(left, right));
}

#[test]
fn concrete_multiplicands_use_specialized_product_paths() {
    let multiplicand = 0x1020_3040;
    let multiplier = 3;
    let instructions = [
        Inst::Mul {
            dest: Reg::T0,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Ecall,
    ];
    let mut symbolic_regs = [[false; 32]; 32];
    let mut symbolic_constants = [None; 32];
    symbolic_regs[Reg::T1.0 as usize] = word(multiplicand);
    symbolic_regs[Reg::T2.0 as usize] = word(multiplier);
    exit_register(&mut symbolic_regs, &mut symbolic_constants);
    let symbolic = run_counting(instructions, &mut symbolic_regs, &mut symbolic_constants);

    let mut right_constant_regs = [[false; 32]; 32];
    let mut right_constant_metadata = [None; 32];
    right_constant_regs[Reg::T1.0 as usize] = word(multiplicand);
    right_constant_regs[Reg::T2.0 as usize] = word(multiplier);
    right_constant_metadata[Reg::T2.0 as usize] = Some(multiplier);
    exit_register(&mut right_constant_regs, &mut right_constant_metadata);
    let right_constant = run_counting(
        instructions,
        &mut right_constant_regs,
        &mut right_constant_metadata,
    );

    let mut left_constant_regs = [[false; 32]; 32];
    let mut left_constant_metadata = [None; 32];
    left_constant_regs[Reg::T1.0 as usize] = word(multiplier);
    left_constant_regs[Reg::T2.0 as usize] = word(multiplicand);
    left_constant_metadata[Reg::T1.0 as usize] = Some(multiplier);
    exit_register(&mut left_constant_regs, &mut left_constant_metadata);
    let left_constant = run_counting(
        instructions,
        &mut left_constant_regs,
        &mut left_constant_metadata,
    );

    for regs in [&right_constant_regs, &left_constant_regs] {
        assert_eq!(
            value(&regs[Reg::T0.0 as usize]),
            multiplicand.wrapping_mul(multiplier)
        );
    }
    assert!(right_constant_metadata[Reg::T0.0 as usize].is_none());
    assert!(left_constant_metadata[Reg::T0.0 as usize].is_none());
    assert!(right_constant.bitand < symbolic.bitand);
    assert!(left_constant.bitand < symbolic.bitand);
    assert!(right_constant.bitxor < symbolic.bitxor);
    assert!(left_constant.bitxor < symbolic.bitxor);

    let mut concrete_regs = [[false; 32]; 32];
    let mut concrete_metadata = [None; 32];
    concrete_regs[Reg::T1.0 as usize] = word(0x8000_0000);
    concrete_regs[Reg::T2.0 as usize] = word(0xffff_fffd);
    concrete_metadata[Reg::T1.0 as usize] = Some(0x8000_0000);
    concrete_metadata[Reg::T2.0 as usize] = Some(0xffff_fffd);
    exit_register(&mut concrete_regs, &mut concrete_metadata);
    let mut concrete_mem = program([
        Inst::Mul {
            dest: Reg::T0,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Mulh {
            dest: Reg::T3,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Mulhsu {
            dest: Reg::T4,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Mulhu {
            dest: Reg::T5,
            src1: Reg::T1,
            src2: Reg::T2,
        },
        Inst::Ecall,
    ]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];
    assert_success(run(
        &mut concrete_mem,
        &mut concrete_regs,
        &mut concrete_metadata,
        &mut rstack,
        &mut vstack,
    ));
    assert_eq!(concrete_metadata[Reg::T0.0 as usize], Some(0x8000_0000));
    assert_eq!(concrete_metadata[Reg::T3.0 as usize], Some(1));
    assert_eq!(
        concrete_metadata[Reg::T4.0 as usize],
        Some(signed_unsigned_high(0x8000_0000, 0xffff_fffd))
    );
    assert_eq!(
        concrete_metadata[Reg::T5.0 as usize],
        Some(unsigned_high(0x8000_0000, 0xffff_fffd))
    );

    let mut zero_regs = [[false; 32]; 32];
    let mut zero_metadata = [None; 32];
    zero_regs[Reg::T1.0 as usize] = word(multiplicand);
    zero_regs[Reg::T2.0 as usize] = word(0);
    zero_metadata[Reg::T2.0 as usize] = Some(0);
    exit_register(&mut zero_regs, &mut zero_metadata);
    let zero = run_counting(instructions, &mut zero_regs, &mut zero_metadata);
    assert_eq!(zero_metadata[Reg::T0.0 as usize], Some(0));
    assert_eq!(zero.bitand, 0);
    assert_eq!(zero.bitxor, 0);
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
        &mem,
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
fn symbolic_sub_subtracts_src2_from_src1() {
    // Pins the historical operand mix-up: the symbolic subtract path once
    // computed `src2 - src1`.
    let mut regs = [[false; 32]; 32];
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
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];

    assert_success(run(
        &mem,
        &mut regs,
        &mut constants,
        &mut rstack,
        &mut vstack,
    ));

    assert_eq!(value(&regs[Reg::T0.0 as usize]), 42);
}

#[test]
fn symbolic_stack_memory_preserves_width_and_extension_rules() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    let source = word(0x8001_80ff);
    regs[Reg::T0.0 as usize] = source;
    exit_register(&mut regs, &mut constants);
    let mem = program([
        Inst::Addi {
            imm: Imm::new_i32(-16),
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
            imm: Imm::new_i32(16),
            dest: Reg::SP,
            src1: Reg::SP,
        },
        Inst::Ecall,
    ]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 2048];

    assert_success(run(
        &mem,
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
    let frame_start = (vstack.len() / 8 - 16) * 8;
    assert_eq!(
        &vstack[frame_start..frame_start + 8],
        &word(0x8001_80ff)[..8]
    );
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
    let mem = program([
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
        &mem,
        &mut regs,
        &mut constants,
        &mut rstack,
        &mut vstack,
    ));

    assert_eq!(constants[Reg::T0.0 as usize], Some(3));
    assert_eq!(rstack[0], 20);
}

#[test]
fn jalr_calls_use_concrete_targets_and_the_private_return_stack() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    regs[Reg::T0.0 as usize] = word(12);
    constants[Reg::T0.0 as usize] = Some(12);
    let mem = program([
        Inst::Jalr {
            offset: Imm::ZERO,
            base: Reg::T0,
            dest: Reg::RA,
        },
        Inst::Addi {
            imm: Imm::new_i32(-1),
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
        Inst::Jalr {
            offset: Imm::ZERO,
            base: Reg::RA,
            dest: Reg::ZERO,
        },
    ]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];

    assert_success(run(
        &mem,
        &mut regs,
        &mut constants,
        &mut rstack,
        &mut vstack,
    ));

    assert_eq!(rstack[0], 4);
    assert_eq!(constants[Reg::RA.0 as usize], Some(4));
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
    let mem = program([
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
        &mem,
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
    let mem = program([
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
    let hash = |_: &mut (), words: &[[bool; 32]]| {
        observed.copy_from_slice(words);
        Ok::<_, Infallible>(array::from_fn(|byte| byte as u8))
    };
    let mut handler = RvDefaultHandler {
        inner: DefaultHandler { context: (), hash },
    };

    let storage_bits = vstack.len();
    assert_success(ert_emit(
        &mut handler,
        &mut vstack,
        storage_bits,
        bounded_memory(&mem),
        &mut rstack,
        0,
        &mut regs,
        &mut constants,
        false,
        true,
    ));

    assert_eq!(value(&observed[0]), 1);
    assert_eq!(value(&observed[7]), 8);
    for i in 0..8 {
        let expected = u32::from_le_bytes(array::from_fn(|byte| (i * 4 + byte) as u8));
        assert_eq!(constants[Reg::A1.0 as usize + i], Some(expected));
    }
}

#[test]
fn a_concrete_load_at_the_detect_address_returns_the_overridden_word() {
    let mem = program([
        Inst::Addi {
            imm: Imm::new_i32(0x100),
            dest: Reg::T0,
            src1: Reg::ZERO,
        },
        Inst::Lw {
            offset: Imm::ZERO,
            dest: Reg::T1,
            base: Reg::T0,
        },
        Inst::Addi {
            imm: Imm::new_i32(-1),
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
    ]);
    let memory = RawMemory::from(mem.as_slice()).with_ert_detect(0x100, 0xdead_beef);
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];
    let mut handler = RvDefaultHandler {
        inner: DefaultHandler {
            context: (),
            hash: no_hash,
        },
    };

    let storage_bits = vstack.len();
    assert_success(ert_emit(
        &mut handler,
        &mut vstack,
        storage_bits,
        memory,
        &mut rstack,
        0,
        &mut regs,
        &mut constants,
        false,
        true,
    ));

    assert_eq!(constants[Reg::T1.0 as usize], Some(0xdead_beef));
}

#[test]
fn ert_func_moves_register_and_stack_abi_values() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    let mem = program([Inst::Ecall]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 128];
    let args = array::from_fn(|i| {
        let constant = if i == 0 { u32::MAX } else { i as u32 };
        (word(constant), Some(constant))
    });
    let mut handler = RvDefaultHandler {
        inner: DefaultHandler {
            context: (),
            hash: no_hash,
        },
    };

    let storage_bits = vstack.len();
    let results = match ert_func::<_, _, 10, 10, _>(
        &mut handler,
        &mut vstack,
        storage_bits,
        bounded_memory(&mem),
        &mut rstack,
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
fn ert_func_rejects_a_symbolic_stack_too_small_for_abi_words() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    let mem = program([Inst::Ecall]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 31];
    let args = array::from_fn(|i| {
        let constant = if i == 0 { u32::MAX } else { i as u32 };
        (word(constant), Some(constant))
    });
    let mut handler = RvDefaultHandler {
        inner: DefaultHandler {
            context: (),
            hash: no_hash,
        },
    };

    let storage_bits = vstack.len();
    assert!(matches!(
        ert_func::<_, _, 9, 0, _>(
            &mut handler,
            &mut vstack,
            storage_bits,
            bounded_memory(&mem),
            &mut rstack,
            0,
            &mut regs,
            &mut constants,
            false,
            true,
            args,
        ),
        Err(ErtError::Unexpected)
    ));
}

#[test]
fn dynamic_control_and_invalid_words_are_reported() {
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    let mem = program([Inst::Beq {
        offset: Imm::new_i32(4),
        src1: Reg::T0,
        src2: Reg::ZERO,
    }]);
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];

    assert!(matches!(
        run(&mem, &mut regs, &mut constants, &mut rstack, &mut vstack,),
        Err(ErtError::Unexpected)
    ));

    let invalid = [0u8; 4];
    assert!(matches!(
        run(
            &invalid,
            &mut regs,
            &mut constants,
            &mut rstack,
            &mut vstack,
        ),
        Err(ErtError::Decode(_))
    ));
}

#[test]
fn compressed_instructions_execute_with_two_byte_steps() {
    // Hand-assembled C.LI/C.MV/C.ADD/C.JR covering the CI and CR formats,
    // including a compressed conventional return.
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    exit_register(&mut regs, &mut constants);
    let mut mem = std::vec::Vec::new();
    mem.extend(0x4585u16.to_le_bytes()); // c.li a1, 1
    mem.extend(0x85aau16.to_le_bytes()); // c.mv a1, a0  (a0 = exit selector)
    mem.extend(0x8082u16.to_le_bytes()); // c.jr ra — underflow: must fail closed
    let mut rstack = [0; 8];
    let mut vstack = [false; 64];
    assert!(matches!(
        run(&mem, &mut regs, &mut constants, &mut rstack, &mut vstack),
        Err(ErtError::Unexpected)
    ));
}
