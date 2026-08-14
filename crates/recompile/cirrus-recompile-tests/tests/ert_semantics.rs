//! Cross-backend semantics-preservation test, using a *real* ERT-recorded
//! program (not a hand-built one) as the input every backend lowers.
//!
//! This is the "including with the ERT" check the project's backends are
//! meant to satisfy: [`cirrus_recompile_core::Recorder`] is plugged into
//! `cirrus_ert::ert_func` -- the same generic entry point `()`/`GC`/
//! `Evaluator` already use -- to record the trace a small RV32 `add`/`and`/
//! `xor` program builds. That one recorded [`Program`] is then lowered by
//! all three new backends (Rust source + `rustc`, LLVM JIT, AArch64 machine
//! code) and by the two reference interpreters
//! (`cirrus_recompile_core::interpret`, `cirrus_recompile_rt::execute`), and
//! every one of them is checked against `cirrus_ert::ert_func`'s own native
//! (`Context = ()`) execution -- the existing, independent oracle -- for
//! several concrete inputs. A mismatch localizes cleanly to whichever single
//! backend produced it, since they all consume the identical `Program`.

use core::array;
use core::convert::Infallible;

use cirrus_core::{ContextWithCreate, HasError};
use cirrus_ert::{DefaultHandler, RawMemory, ert_func};
use cirrus_recompile_core::{Idx, Program, Recorder};
use rv_asm::{Imm, Inst, Reg, Xlen};

fn word(value: u32) -> [bool; 32] {
    array::from_fn(|bit| (value >> bit) & 1 != 0)
}

fn value(word: &[bool; 32]) -> u32 {
    word.iter()
        .enumerate()
        .fold(0u32, |acc, (bit, set)| acc | ((*set as u32) << bit))
}

fn encode(instructions: impl IntoIterator<Item = Inst>) -> Vec<u8> {
    instructions
        .into_iter()
        .flat_map(|instruction| instruction.encode_normal(Xlen::Rv32).to_le_bytes())
        .collect()
}

/// `a2 = a0 + a1; a3 = a0 & a1; a4 = a0 ^ a1; a0 = -1; ecall` -- a small but
/// nontrivial program (the `add` alone is a 32-stage ripple-carry circuit)
/// that exercises `BitAnd`/`BitOr`/`BitXor` without ever branching on the
/// (symbolic) data inputs, only on the `a0 = -1` sentinel this program sets
/// itself immediately before the exit `ecall` -- concrete regardless of
/// `a0`/`a1`'s own symbolic-ness, exactly as `cirrus-ert`'s "branch only on
/// concrete values" contract requires.
fn program_image() -> Vec<u8> {
    encode([
        Inst::Add { dest: Reg::A2, src1: Reg::A0, src2: Reg::A1 },
        Inst::And { dest: Reg::A3, src1: Reg::A0, src2: Reg::A1 },
        Inst::Xor { dest: Reg::A4, src1: Reg::A0, src2: Reg::A1 },
        Inst::Addi { imm: Imm::new_i32(-1), dest: Reg::A0, src1: Reg::ZERO },
        Inst::Ecall,
    ])
}

fn no_hash_bool(_: &[[bool; 32]]) -> Result<[u8; 32], Infallible> {
    unreachable!("this test program never issues the hash ECALL")
}

/// Run the program natively (`Context = ()`) for one concrete `(a, b)` --
/// the independent oracle every backend is checked against.
fn native_golden(mem: &[u8], a: u32, b: u32) -> (u32, u32, u32) {
    let mut regs = [[false; 32]; 32];
    let mut reg_consts = [None; 32];
    let mut rstack = [0u32; 8];
    let mut vstack = [false; 128];
    let mut context = ();
    let mut hash = no_hash_bool;
    let mut handler = DefaultHandler {
        context: &mut context,
        hash: &mut hash,
    };

    let results = ert_func::<_, _, 2, 5>(
        &mut handler,
        RawMemory::from(mem),
        &mut rstack,
        &mut vstack,
        0,
        &mut regs,
        &mut reg_consts,
        false,
        true,
        [(word(a), Some(a)), (word(b), Some(b))],
    )
    .unwrap_or_else(|_| panic!("native ERT execution of the fixture program must succeed"));

    (value(&results[2].0), value(&results[3].0), value(&results[4].0))
}

fn no_hash_idx(_: &[[Idx; 32]]) -> Result<[u8; 32], <Recorder as HasError>::Error> {
    unreachable!("this test program never issues the hash ECALL")
}

/// Record the same program's trace by plugging [`Recorder`] into the same
/// `ert_func` entry point, with fully symbolic (concrete-shadow-`None`)
/// inputs -- this is the "recompile once, reuse for many secret inputs"
/// artifact every backend below lowers.
fn record_program(mem: &[u8]) -> Program {
    let mut recorder = Recorder::new();
    let zero = recorder.create(false).unwrap();
    let one = recorder.create(true).unwrap();
    let a_bits: [Idx; 32] = array::from_fn(|_| recorder.create(false).unwrap());
    let b_bits: [Idx; 32] = array::from_fn(|_| recorder.create(false).unwrap());

    let mut regs = [[Idx(0); 32]; 32];
    let mut reg_consts = [None; 32];
    let mut rstack = [0u32; 8];
    let mut vstack = [Idx(0); 128];
    let mut hash = no_hash_idx;
    let mut handler = DefaultHandler {
        context: &mut recorder,
        hash: &mut hash,
    };

    let results = ert_func::<_, _, 2, 5>(
        &mut handler,
        RawMemory::from(mem),
        &mut rstack,
        &mut vstack,
        0,
        &mut regs,
        &mut reg_consts,
        zero,
        one,
        [(a_bits, None), (b_bits, None)],
    )
    .unwrap_or_else(|_| panic!("recording the fixture program's trace must succeed"));

    let mut inputs = Vec::with_capacity(64);
    inputs.extend_from_slice(&a_bits);
    inputs.extend_from_slice(&b_bits);

    let mut outputs = Vec::with_capacity(96);
    outputs.extend_from_slice(&results[2].0); // a2 = a + b
    outputs.extend_from_slice(&results[3].0); // a3 = a & b
    outputs.extend_from_slice(&results[4].0); // a4 = a ^ b

    recorder.finish(inputs, outputs)
}

fn bits_of(program: &Program, a: u32, b: u32) -> Vec<bool> {
    let a_bits = word(a);
    let b_bits = word(b);
    (0..32)
        .map(|i| a_bits[i])
        .chain((0..32).map(|i| b_bits[i]))
        .collect::<Vec<_>>()
        .into_iter()
        .take(program.inputs.len())
        .collect()
}

fn triple_of(outputs: &[bool]) -> (u32, u32, u32) {
    let word_at = |offset: usize| -> u32 {
        (0..32).fold(0u32, |acc, bit| acc | ((outputs[offset + bit] as u32) << bit))
    };
    (word_at(0), word_at(32), word_at(64))
}

const CASES: [(u32, u32); 4] = [(0, 0), (1, 2), (0xFFFF_FFFF, 1), (0x1234_5678, 0x0F0F_0F0F)];

#[test]
fn recorded_ert_program_matches_native_execution_via_ir_interpreter() {
    let mem = program_image();
    let program = record_program(&mem);
    for &(a, b) in &CASES {
        let golden = native_golden(&mem, a, b);
        let outputs = cirrus_recompile_core::interpret(&program, &bits_of(&program, a, b));
        assert_eq!(triple_of(&outputs), golden, "interpret() mismatch for ({a:#x}, {b:#x})");
    }
}

#[test]
fn recorded_ert_program_matches_native_execution_via_pinned_rt() {
    let mem = program_image();
    let program = record_program(&mem);
    for &(a, b) in &CASES {
        let golden = native_golden(&mem, a, b);
        let outputs = cirrus_recompile_rt::execute(&mut (), &program, &bits_of(&program, a, b));
        assert_eq!(triple_of(&outputs), golden, "cirrus_recompile_rt::execute mismatch for ({a:#x}, {b:#x})");
    }
}

#[test]
fn recorded_ert_program_matches_native_execution_via_rust_backend() {
    let mem = program_image();
    let program = record_program(&mem);
    let compiled = cirrus_rust_codegen::CompiledProgram::compile(
        &program,
        "cirrus_recompile_tests_rust_fn",
        &cirrus_rust_codegen::BackendTarget::plaintext(),
    );
    for &(a, b) in &CASES {
        let golden = native_golden(&mem, a, b);
        let outputs = compiled.run_plaintext(&program, &bits_of(&program, a, b));
        assert_eq!(triple_of(&outputs), golden, "Rust backend mismatch for ({a:#x}, {b:#x})");
    }
}

fn plaintext_pinned() -> cirrus_llvm::PinnedAddresses {
    cirrus_llvm::PinnedAddresses {
        create: cirrus_recompile_rt::plaintext::create as *const () as usize,
        bitand: cirrus_recompile_rt::plaintext::bitand as *const () as usize,
        bitor: cirrus_recompile_rt::plaintext::bitor as *const () as usize,
        bitxor: cirrus_recompile_rt::plaintext::bitxor as *const () as usize,
        mux: cirrus_recompile_rt::plaintext::mux as *const () as usize,
    }
}

#[test]
fn recorded_ert_program_matches_native_execution_via_llvm_backend() {
    let mem = program_image();
    let program = record_program(&mem);
    let context = inkwell::context::Context::create();
    let compiled = cirrus_llvm::CompiledProgram::compile(
        &context,
        &program,
        "cirrus_recompile_tests_llvm_fn",
        &plaintext_pinned(),
    );
    for &(a, b) in &CASES {
        let golden = native_golden(&mem, a, b);
        let outputs = compiled.run_plaintext(&program, &bits_of(&program, a, b));
        assert_eq!(triple_of(&outputs), golden, "LLVM backend mismatch for ({a:#x}, {b:#x})");
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn recorded_ert_program_matches_native_execution_via_asm_backend() {
    use cirrus_asm::recompile::{PinnedAddresses, compile_aarch64};

    struct ExecMem {
        ptr: *mut libc::c_void,
        len: usize,
    }
    impl ExecMem {
        fn new(code: &[u8]) -> Self {
            let len = code.len();
            let ptr = unsafe {
                libc::mmap(
                    core::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            assert_ne!(ptr, libc::MAP_FAILED, "mmap failed");
            unsafe { core::ptr::copy_nonoverlapping(code.as_ptr(), ptr as *mut u8, len) };
            let status = unsafe { libc::mprotect(ptr, len, libc::PROT_READ | libc::PROT_EXEC) };
            assert_eq!(status, 0, "mprotect failed");
            Self { ptr, len }
        }
        unsafe fn call<Backend, Wrapped>(&self, backend: *mut Backend, buf: *mut Wrapped) {
            let function: unsafe extern "C" fn(*mut Backend, *mut Wrapped) =
                unsafe { core::mem::transmute(self.ptr) };
            unsafe { function(backend, buf) };
        }
    }
    impl Drop for ExecMem {
        fn drop(&mut self) {
            unsafe { libc::munmap(self.ptr, self.len) };
        }
    }

    let mem = program_image();
    let program = record_program(&mem);
    let pinned = PinnedAddresses {
        create: cirrus_recompile_rt::plaintext::create as *const () as usize,
        bitand: cirrus_recompile_rt::plaintext::bitand as *const () as usize,
        bitor: cirrus_recompile_rt::plaintext::bitor as *const () as usize,
        bitxor: cirrus_recompile_rt::plaintext::bitxor as *const () as usize,
        mux: cirrus_recompile_rt::plaintext::mux as *const () as usize,
    };
    let code = compile_aarch64(&program, &pinned);
    let exec = ExecMem::new(&code);

    for &(a, b) in &CASES {
        let golden = native_golden(&mem, a, b);
        let mut buf = vec![false; program.ops.len()];
        for (&idx, bit) in program.inputs.iter().zip(bits_of(&program, a, b)) {
            buf[idx.get()] = bit;
        }
        unsafe { exec.call(&mut () as *mut (), buf.as_mut_ptr()) };
        let outputs: Vec<bool> = program.outputs.iter().map(|idx| buf[idx.get()]).collect();
        assert_eq!(triple_of(&outputs), golden, "asm backend mismatch for ({a:#x}, {b:#x})");
    }
}
