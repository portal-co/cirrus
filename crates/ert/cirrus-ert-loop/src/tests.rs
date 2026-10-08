extern crate std;

use core::{array, convert::Infallible};

use cirrus_ert::{DefaultHandler, ErtError, RawMemory, RvDefaultHandler, ert64_emit};
use rv_asm::{Imm, Inst, Reg, Xlen};
use std::vec::Vec;

use crate::{IndirectTargets, LoopedMachine, looped_ert64_func};

fn word<const BITS: usize>(value: u64) -> [bool; BITS] {
    array::from_fn(|bit| (value >> bit) & 1 != 0)
}

fn value<const BITS: usize>(word: &[bool; BITS]) -> u64 {
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

fn no_hash(_: &mut (), _: &[[bool; 64]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}

fn handler() -> PlaintextHandler {
    RvDefaultHandler {
        inner: DefaultHandler {
            context: (),
            hash: no_hash,
        },
    }
}

/// Build a looped machine over a fresh stack/candidate table at pc 0.
type PlaintextHandler = RvDefaultHandler<
    DefaultHandler<(), fn(&mut (), &[[bool; 64]]) -> Result<[u8; 32], Infallible>>,
>;

/// The looped machine over a fresh stack/candidate table at pc 0.
fn machine64<'a>(
    mem: &'a [u8],
    vstack: &'a mut [bool],
    rstack: &'a mut [u64],
    candidates: &'a mut [u64],
    handler: &'a mut PlaintextHandler,
) -> LoopedMachine<'a, PlaintextHandler, bool, Infallible, 64, u64> {
    let storage_bits = vstack.len();
    LoopedMachine::new(
        handler,
        vstack,
        storage_bits,
        RawMemory::from(mem),
        rstack,
        0,
        [],
        candidates,
        &[],
        false,
        true,
    )
    .unwrap()
}

/// The same image through the single-pass interpreter.
fn run_single_concrete(
    mem: &[u8],
    a0: u64,
) -> Result<([[bool; 64]; 32], [Option<u64>; 32]), ErtError<Infallible>> {
    let (mut regs, mut constants) = {
        let regs = [[false; 64]; 32];
        let constants = [None; 32];
        (regs, constants)
    };
    regs[Reg::A0.0 as usize] = word(a0);
    constants[Reg::A0.0 as usize] = Some(a0);
    regs[Reg::T0.0 as usize] = word(a0);
    constants[Reg::T0.0 as usize] = Some(a0);
    let mut handler = handler();
    let mut vstack = [false; 4096];
    let storage_bits = vstack.len();
    let mut rstack = [0u64; 32];
    ert64_emit(
        &mut handler,
        &mut vstack,
        storage_bits,
        RawMemory::from(mem),
        &mut rstack,
        0,
        &mut regs,
        &mut constants,
        false,
        true,
    )?;
    Ok((regs, constants))
}

fn run_single(
    mem: &[u8],
    a0: Option<u64>,
) -> Result<([[bool; 64]; 32], [Option<u64>; 32]), ErtError<Infallible>> {
    let mut handler = handler();
    let mut vstack = [false; 4096];
    let storage_bits = vstack.len();
    let mut rstack = [0u64; 32];
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    if let Some(a0) = a0 {
        regs[Reg::A0.0 as usize] = word(a0);
    }
    ert64_emit(
        &mut handler,
        &mut vstack,
        storage_bits,
        RawMemory::from(mem),
        &mut rstack,
        0,
        &mut regs,
        &mut constants,
        false,
        true,
    )?;
    Ok((regs, constants))
}

#[test]
fn concrete_control_flow_matches_the_single_pass_interpreter() {
    // A call whose callee has a concrete-bound loop, then a concrete branch.
    let mem = program([
        Inst::Jal {
            offset: Imm::new_i32(24),
            dest: Reg::RA,
        },
        // Return landing at 4:
        Inst::Addi {
            imm: Imm::new_i32(1),
            dest: Reg::T2,
            src1: Reg::A1,
        },
        Inst::Beq {
            offset: Imm::new_i32(8),
            src1: Reg::A0,
            src2: Reg::T0,
        },
        Inst::Addi {
            imm: Imm::new_i32(100),
            dest: Reg::T2,
            src1: Reg::ZERO,
        },
        Inst::Addi {
            imm: Imm::new_i32(-1),
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
        // Callee at 24: a concrete-bound loop, a1 = 0+1+1.
        Inst::Addi {
            imm: Imm::ZERO,
            dest: Reg::A1,
            src1: Reg::ZERO,
        },
        Inst::Addi {
            imm: Imm::new_i32(2),
            dest: Reg::T1,
            src1: Reg::ZERO,
        },
        // loop at 32:
        Inst::Addi {
            imm: Imm::new_i32(1),
            dest: Reg::A1,
            src1: Reg::A1,
        },
        Inst::Blt {
            offset: Imm::new_i32(-4),
            src1: Reg::A1,
            src2: Reg::T1,
        },
        Inst::Jalr {
            offset: Imm::ZERO,
            base: Reg::RA,
            dest: Reg::ZERO,
        },
    ]);
    let mut handler = handler();
    let mut vstack = [false; 4096];
    let mut rstack = [0u64; 32];
    let mut candidates = [0u64; 16];
    let mut machine = machine64(
        &mem,
        &mut vstack,
        &mut rstack,
        &mut candidates,
        &mut handler,
    );
    machine.set_register(Reg::A0, word(3), Some(3));
    machine.set_register(Reg::T0, word(3), Some(3));
    assert!(machine.run(100, |wire| *wire).unwrap());

    let (single_regs, _) = run_single_concrete(&mem, 3).expect("single-pass runs");
    for register in [Reg::A1, Reg::T2] {
        assert_eq!(
            machine.regs()[register.0 as usize],
            single_regs[register.0 as usize],
            "register {register:?} must match the single-pass interpreter"
        );
    }
}

#[test]
fn a_secret_trip_count_loop_runs_to_done() {
    // a1 counts up to the symbolic a0; the branch stays symbolic.
    let mem = program([
        Inst::Addi {
            imm: Imm::ZERO,
            dest: Reg::A1,
            src1: Reg::ZERO,
        },
        // loop at 4:
        Inst::Addi {
            imm: Imm::new_i32(1),
            dest: Reg::A1,
            src1: Reg::A1,
        },
        Inst::Blt {
            offset: Imm::new_i32(-4),
            src1: Reg::A1,
            src2: Reg::A0,
        },
        // exit at 12:
        Inst::Addi {
            imm: Imm::new_i32(-1),
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
    ]);
    let mut handler = handler();
    let mut vstack = [false; 4096];
    let storage_bits = vstack.len();
    let mut rstack = [0u64; 32];
    let mut candidates = [0u64; 16];
    // a0 = 4 on the wires, symbolic to the machine, through the ABI entry.
    let results = looped_ert64_func::<_, _, 1, 2, _>(
        &mut handler,
        &mut vstack,
        storage_bits,
        RawMemory::from(&mem[..]),
        &mut rstack,
        0,
        [(word(4), None)],
        &mut candidates,
        &[],
        false,
        true,
        100,
        |wire| *wire,
    )
    .expect("looped run finishes");
    assert_eq!(value(&results[1].0), 4);
    // The single-pass interpreter fails closed on the same image.
    assert!(matches!(
        run_single(&mem, Some(4)),
        Err(ErtError::Unexpected)
    ));
}

#[test]
fn divergent_stack_writes_are_predicated() {
    // beqz a0, +8 skips a predicated store; both paths then load the slot.
    // beqz a0, +16 skips a predicated frame store; both paths then load it.
    let mem = program([
        Inst::Beq {
            offset: Imm::new_i32(16),
            src1: Reg::A0,
            src2: Reg::ZERO,
        },
        Inst::Addi {
            imm: Imm::new_i32(-16),
            dest: Reg::SP,
            src1: Reg::SP,
        },
        Inst::Sw {
            offset: Imm::ZERO,
            src: Reg::A2,
            base: Reg::SP,
        },
        Inst::Addi {
            imm: Imm::new_i32(16),
            dest: Reg::SP,
            src1: Reg::SP,
        },
        // landing at 16:
        Inst::Lw {
            offset: Imm::new_i32(-16),
            dest: Reg::A3,
            base: Reg::SP,
        },
        Inst::Addi {
            imm: Imm::new_i32(-1),
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
    ]);
    for (input, expected) in [(1u64, 0x1234u64), (0u64, 0u64)] {
        let mut handler = handler();
        let mut vstack = [false; 4096];
        let mut rstack = [0u64; 32];
        let mut candidates = [0u64; 16];
        let mut machine = machine64(
            &mem,
            &mut vstack,
            &mut rstack,
            &mut candidates,
            &mut handler,
        );
        machine.set_register(Reg::A0, word(input), None);
        machine.set_register(Reg::A2, word(0x1234), Some(0x1234));
        assert!(machine.run(100, |wire| *wire).unwrap());
        assert_eq!(
            value(&machine.regs()[Reg::A3.0 as usize]),
            expected,
            "input {input}"
        );
    }
}

#[test]
fn a_bounded_indirect_call_dispatches_to_declared_targets() {
    // jalr ra, 0(t0) with t0 symbolic; the declaration pins target 16.
    let mem = program([
        Inst::Jalr {
            offset: Imm::ZERO,
            base: Reg::T0,
            dest: Reg::RA,
        },
        // Return landing at 4:
        Inst::Addi {
            imm: Imm::new_i32(-1),
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
        Inst::Ecall,
        // Callee at 16:
        Inst::Addi {
            imm: Imm::new_i32(77),
            dest: Reg::A1,
            src1: Reg::ZERO,
        },
        Inst::Jalr {
            offset: Imm::ZERO,
            base: Reg::RA,
            dest: Reg::ZERO,
        },
    ]);
    let mut handler = handler();
    let mut vstack = [false; 4096];
    let mut rstack = [0u64; 32];
    let mut candidates = [0u64; 16];
    let indirect = [IndirectTargets {
        pc: 0,
        targets: &[16],
    }];
    let storage_bits = vstack.len();
    let mut machine = LoopedMachine::<_, bool, Infallible, 64, u64>::new(
        &mut handler,
        &mut vstack,
        storage_bits,
        RawMemory::from(&mem[..]),
        &mut rstack,
        0,
        [],
        &mut candidates,
        &indirect,
        false,
        true,
    )
    .unwrap();
    machine.set_register(Reg::T0, word(16), None);
    assert!(machine.run(100, |wire| *wire).unwrap());
    assert_eq!(value(&machine.regs()[Reg::A1.0 as usize]), 77);

    // Without the declaration, the same image fails closed.
    let mut handler = self::handler();
    let mut vstack = [false; 4096];
    let mut rstack = [0u64; 32];
    let mut candidates = [0u64; 16];
    let storage_bits = vstack.len();
    let mut machine = LoopedMachine::<_, bool, Infallible, 64, u64>::new(
        &mut handler,
        &mut vstack,
        storage_bits,
        RawMemory::from(&mem[..]),
        &mut rstack,
        0,
        [],
        &mut candidates,
        &[],
        false,
        true,
    )
    .unwrap();
    machine.set_register(Reg::T0, word(16), None);
    assert!(matches!(
        machine.run(100, |wire| *wire),
        Err(ErtError::Unexpected)
    ));
}

#[test]
fn budget_truncation_is_deterministic() {
    let mem = program([
        Inst::Addi {
            imm: Imm::ZERO,
            dest: Reg::A1,
            src1: Reg::ZERO,
        },
        Inst::Addi {
            imm: Imm::new_i32(1),
            dest: Reg::A1,
            src1: Reg::A1,
        },
        Inst::Blt {
            offset: Imm::new_i32(-4),
            src1: Reg::A1,
            src2: Reg::A0,
        },
        Inst::Addi {
            imm: Imm::new_i32(-1),
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
    ]);
    let mut handler = handler();
    let mut vstack = [false; 4096];
    let mut rstack = [0u64; 32];
    let mut candidates = [0u64; 16];
    let mut machine = machine64(
        &mem,
        &mut vstack,
        &mut rstack,
        &mut candidates,
        &mut handler,
    );
    machine.set_register(Reg::A0, word(9), None);
    // Two steps cannot finish a 9-iteration loop.
    assert!(!machine.run(2, |wire| *wire).unwrap());
    // Resuming with a generous budget finishes, deterministically.
    assert!(machine.run(100, |wire| *wire).unwrap());
    assert_eq!(value(&machine.regs()[Reg::A1.0 as usize]), 9);
}

#[test]
fn rv32_looped_entry_runs() {
    // A 32-bit secret-trip-count loop through `looped_ert_func`.
    let mem: Vec<u8> = [
        Inst::Addi {
            imm: Imm::ZERO,
            dest: Reg::A1,
            src1: Reg::ZERO,
        },
        Inst::Addi {
            imm: Imm::new_i32(1),
            dest: Reg::A1,
            src1: Reg::A1,
        },
        Inst::Blt {
            offset: Imm::new_i32(-4),
            src1: Reg::A1,
            src2: Reg::A0,
        },
        Inst::Addi {
            imm: Imm::new_i32(-1),
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
    ]
    .into_iter()
    .flat_map(|instruction| instruction.encode_normal(Xlen::Rv32).to_le_bytes())
    .collect();
    let mut handler32 = RvDefaultHandler {
        inner: DefaultHandler {
            context: (),
            hash: |_: &mut (), _: &[[bool; 32]]| -> Result<[u8; 32], Infallible> { Ok([0; 32]) },
        },
    };
    let mut vstack = [false; 4096];
    let storage_bits = vstack.len();
    let mut rstack = [0u32; 32];
    let mut candidates = [0u64; 16];
    let mut machine = LoopedMachine::<_, bool, Infallible, 32, u32>::new(
        &mut handler32,
        &mut vstack,
        storage_bits,
        RawMemory::from(&mem[..]),
        &mut rstack,
        0,
        [],
        &mut candidates,
        &[],
        false,
        true,
    )
    .unwrap();
    machine.set_register(Reg::A0, word(7), None);
    assert!(machine.run(100, |wire| *wire).unwrap());
    assert_eq!(value(&machine.regs()[Reg::A1.0 as usize]), 7);
}

#[cfg(feature = "precompute")]
#[test]
fn precomputed_program_matches_live_execution() {
    use crate::LoopedProgram;
    let mem = program([
        Inst::Addi {
            imm: Imm::ZERO,
            dest: Reg::A1,
            src1: Reg::ZERO,
        },
        Inst::Addi {
            imm: Imm::new_i32(1),
            dest: Reg::A1,
            src1: Reg::A1,
        },
        Inst::Blt {
            offset: Imm::new_i32(-4),
            src1: Reg::A1,
            src2: Reg::A0,
        },
        Inst::Addi {
            imm: Imm::new_i32(-1),
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
    ]);
    let program = LoopedProgram::compile(&RawMemory::from(&mem[..]), 0, &[], 16).unwrap();
    assert_eq!(program.boundaries().len(), 1);
    assert_eq!(program.boundaries()[&8], (4, 12));

    let run_with = |with_program: bool| -> u64 {
        let mut handler = handler();
        let mut vstack = [false; 4096];
        let mut rstack = [0u64; 32];
        let mut candidates = [0u64; 16];
        let mut machine = machine64(
            &mem,
            &mut vstack,
            &mut rstack,
            &mut candidates,
            &mut handler,
        );
        if with_program {
            machine.set_program(&program);
        }
        machine.set_register(Reg::A0, word(5), None);
        assert!(machine.run(100, |wire| *wire).unwrap());
        value(&machine.regs()[Reg::A1.0 as usize])
    };
    assert_eq!(run_with(false), 5);
    assert_eq!(run_with(true), 5);
}

/// The recorded execution of the looped emulator must replay on a plaintext
/// backend with the same outputs (the `run_step_loop` contract).
#[test]
fn recorder_run_replays_on_plaintext() {
    use cirrus_core::ContextWithCreate;
    use cirrus_recompile_core::Recorder;
    use cirrus_volar_boolar::MuxTreeContext;

    let mem = program([
        Inst::Addi {
            imm: Imm::ZERO,
            dest: Reg::A1,
            src1: Reg::ZERO,
        },
        Inst::Addi {
            imm: Imm::new_i32(1),
            dest: Reg::A1,
            src1: Reg::A1,
        },
        Inst::Blt {
            offset: Imm::new_i32(-4),
            src1: Reg::A1,
            src2: Reg::A0,
        },
        Inst::Addi {
            imm: Imm::new_i32(-1),
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
    ]);

    let mut handler = RvDefaultHandler {
        inner: DefaultHandler {
            context: MuxTreeContext::new(Recorder::new()),
            hash: |_: &mut MuxTreeContext<Recorder>,
                   _: &[[cirrus_recompile_core::Idx; 64]]|
             -> Result<[u8; 32], Infallible> { Ok([0; 32]) },
        },
    };
    // Constants and the symbolic a0 input slots.
    let zero = handler.inner.context.create(false).unwrap();
    let one = handler.inner.context.create(true).unwrap();
    let a0_wires: [cirrus_recompile_core::Idx; 64] =
        array::from_fn(|_| handler.inner.context.create(false).unwrap());
    let a0_slots: std::vec::Vec<_> = a0_wires.to_vec();
    let mut vstack = [zero; 4096];
    let mut rstack = [0u64; 32];
    let mut candidates = [0u64; 16];
    let storage_bits = vstack.len();
    let mut machine = LoopedMachine::<_, cirrus_recompile_core::Idx, Infallible, 64, u64>::new(
        &mut handler,
        &mut vstack,
        storage_bits,
        RawMemory::from(&mem[..]),
        &mut rstack,
        0,
        [],
        &mut candidates,
        &[],
        zero,
        one,
    )
    .unwrap();
    machine.set_register(Reg::A0, a0_wires, None);
    // Record a bounded number of steps (the trace replays as far as it goes).
    machine.run(24, |_| false).unwrap();

    let a1_wires = machine.regs()[Reg::A1.0 as usize];
    let done_wire = *machine.done_wire();
    let mut outputs: std::vec::Vec<_> = a1_wires.to_vec();
    outputs.push(done_wire);
    let recorder = handler.inner.context.into_inner();
    let recorded = recorder.finish(a0_slots.clone(), outputs);

    // Replay with a0 = 4: a1 must count to 4 and done must be set.
    let mut inputs = std::vec![false; 64];
    for bit in 0..64 {
        inputs[bit] = (4u64 >> bit) & 1 != 0;
    }
    let results =
        cirrus_recompile_rt::execute(&mut (), &recorded, &inputs).expect("plaintext replay");
    let a1_value = results[..64]
        .iter()
        .enumerate()
        .fold(0u64, |value, (bit, set)| value | ((*set as u64) << bit));
    assert_eq!(a1_value, 4, "plaintext replay of the recorded trace");
    assert!(results[64], "done must be set after replay");
}
