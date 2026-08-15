extern crate std;

use core::{array, convert::Infallible};

use cirrus_core::{ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithValue, HasError};
use cirrus_ert_core::{EarlyExitLoopOptions, EcallOutcome, Handler};
use rv_asm::{Imm, Inst, Reg, Xlen};
use std::vec::Vec;

use crate::{ErtError, RawMemory, RvHandler, ert_emit};

fn value(word: &[bool; 32]) -> u32 {
    word.iter()
        .enumerate()
        .fold(0, |value, (bit, set)| value | ((*set as u32) << bit))
}

fn program(instructions: &[Inst]) -> Vec<u8> {
    instructions
        .iter()
        .flat_map(|instruction| instruction.encode_normal(Xlen::Rv32).to_le_bytes())
        .collect()
}

fn addr(index: usize) -> i32 {
    (index * 4) as i32
}

/// A minimal handler that exits on concrete `a0 == 0xffff_ffff` (matching
/// `DefaultHandler`'s convention) and, when `enabled` is set, opts into the
/// early-exit-loop recognizer under test.
struct TestHandler {
    enabled: bool,
}

impl HasError for TestHandler {
    type Error = Infallible;
}

impl ContextWithValue<bool> for TestHandler {
    type Wrapped = bool;
}

impl ContextWithBitAnd<bool> for TestHandler {
    fn bitand(&mut self, a: bool, b: bool) -> Result<bool, Infallible> {
        Ok(a & b)
    }
    fn bitand_assign(&mut self, a: &mut bool, b: bool) -> Result<(), Infallible> {
        *a &= b;
        Ok(())
    }
}

impl ContextWithBitOr<bool> for TestHandler {
    fn bitor(&mut self, a: bool, b: bool) -> Result<bool, Infallible> {
        Ok(a | b)
    }
    fn bitor_assign(&mut self, a: &mut bool, b: bool) -> Result<(), Infallible> {
        *a |= b;
        Ok(())
    }
}

impl ContextWithBitXor<bool> for TestHandler {
    fn bitxor(&mut self, a: bool, b: bool) -> Result<bool, Infallible> {
        Ok(a ^ b)
    }
    fn bitxor_assign(&mut self, a: &mut bool, b: bool) -> Result<(), Infallible> {
        *a ^= b;
        Ok(())
    }
}

impl Handler<bool> for TestHandler {
    fn ecall(
        &mut self,
        _regs: &mut [[bool; 32]],
        reg_consts: &mut [Option<u32>],
        _offsets: &mut [Option<i32>],
        _zero: &bool,
        _one: &bool,
    ) -> Result<EcallOutcome, Infallible> {
        match reg_consts[Reg::A0.0 as usize] {
            Some(0xffff_ffff) => Ok(EcallOutcome::Exit),
            _ => Ok(EcallOutcome::Unexpected),
        }
    }

    fn early_exit_loop_options(&self) -> EarlyExitLoopOptions {
        EarlyExitLoopOptions {
            enabled: self.enabled,
            max_lookahead_instructions: 64,
        }
    }
}

impl RvHandler<bool> for TestHandler {}

/// `T2 = 1; for i in 0..len { if a[i] != b[i] { T2 = 0; break; } }`, built
/// into the canonical rotated-loop shape a real `-O2` toolchain emits: a
/// concrete-bounded backward branch as the latch, and a single
/// secret-dependent forward branch as the early exit, both landing on the
/// same merge point.
///
/// `a`/`b` are stored on the *symbolic* stack (via `sb`, SP-relative), not
/// embedded in the program image: a load from a concrete `RawMemory`
/// address resolves through the interpreter's existing concrete fast path
/// regardless of what the recognizer does, which would make a test built
/// that way pass or fail for the wrong reason. Stack loads are always
/// symbolic here (see `cirrus-ert`'s `Machine::load_address`), so the
/// compare genuinely cannot resolve without the recognizer.
fn memcmp_loop_program(a: &[u8], b: &[u8]) -> Vec<u8> {
    let len = a.len() as i32;
    let mut instrs = Vec::new();

    instrs.push(Inst::Addi {
        imm: Imm::new_i32(-2 * len),
        dest: Reg::SP,
        src1: Reg::SP,
    });
    for (k, &byte) in a.iter().enumerate() {
        instrs.push(Inst::Addi {
            imm: Imm::new_i32(byte as i32),
            dest: Reg::S0,
            src1: Reg::ZERO,
        });
        instrs.push(Inst::Sb {
            offset: Imm::new_i32(k as i32),
            src: Reg::S0,
            base: Reg::SP,
        });
    }
    for (k, &byte) in b.iter().enumerate() {
        instrs.push(Inst::Addi {
            imm: Imm::new_i32(byte as i32),
            dest: Reg::S0,
            src1: Reg::ZERO,
        });
        instrs.push(Inst::Sb {
            offset: Imm::new_i32(len + k as i32),
            src: Reg::S0,
            base: Reg::SP,
        });
    }

    instrs.push(Inst::Addi {
        imm: Imm::new_i32(0),
        dest: Reg::T0,
        src1: Reg::ZERO,
    }); // i = 0
    instrs.push(Inst::Addi {
        imm: Imm::new_i32(len),
        dest: Reg::T1,
        src1: Reg::ZERO,
    }); // n = len
    instrs.push(Inst::Addi {
        imm: Imm::new_i32(1),
        dest: Reg::T2,
        src1: Reg::ZERO,
    }); // result = 1
    instrs.push(Inst::Addi {
        imm: Imm::new_i32(len),
        dest: Reg::S1,
        src1: Reg::SP,
    }); // base_b = sp + len

    let h = instrs.len();
    instrs.push(Inst::Add {
        dest: Reg::T3,
        src1: Reg::SP,
        src2: Reg::T0,
    }); // addr_a = sp + i
    instrs.push(Inst::Add {
        dest: Reg::T4,
        src1: Reg::S1,
        src2: Reg::T0,
    }); // addr_b = base_b + i
    instrs.push(Inst::Lbu {
        offset: Imm::ZERO,
        dest: Reg::T5,
        base: Reg::T3,
    });
    instrs.push(Inst::Lbu {
        offset: Imm::ZERO,
        dest: Reg::T6,
        base: Reg::T4,
    });

    let beq = instrs.len();
    instrs.push(Inst::Beq {
        offset: Imm::ZERO,
        src1: Reg::T5,
        src2: Reg::T6,
    }); // patched below

    instrs.push(Inst::Addi {
        imm: Imm::new_i32(0),
        dest: Reg::T2,
        src1: Reg::ZERO,
    }); // result = 0   <- exit prelude

    let jal = instrs.len();
    instrs.push(Inst::Jal {
        offset: Imm::ZERO,
        dest: Reg::ZERO,
    }); // patched below

    let cont = instrs.len();
    instrs.push(Inst::Addi {
        imm: Imm::new_i32(1),
        dest: Reg::T0,
        src1: Reg::T0,
    }); // i += 1

    let blt = instrs.len();
    instrs.push(Inst::Blt {
        offset: Imm::ZERO,
        src1: Reg::T0,
        src2: Reg::T1,
    }); // patched below

    let done = instrs.len();
    instrs.push(Inst::Addi {
        imm: Imm::new_i32(2 * len),
        dest: Reg::SP,
        src1: Reg::SP,
    }); // restore sp -- required for a balanced-stack exit
    instrs.push(Inst::Addi {
        imm: Imm::new_i32(-1),
        dest: Reg::A0,
        src1: Reg::ZERO,
    });
    instrs.push(Inst::Ecall);

    instrs[beq] = Inst::Beq {
        offset: Imm::new_i32(addr(cont) - addr(beq)),
        src1: Reg::T5,
        src2: Reg::T6,
    };
    instrs[jal] = Inst::Jal {
        offset: Imm::new_i32(addr(done) - addr(jal)),
        dest: Reg::ZERO,
    };
    instrs[blt] = Inst::Blt {
        offset: Imm::new_i32(addr(h) - addr(blt)),
        src1: Reg::T0,
        src2: Reg::T1,
    };

    program(&instrs)
}

fn run_memcmp(a: &[u8], b: &[u8], enabled: bool) -> Result<u32, ErtError<Infallible>> {
    assert_eq!(a.len(), b.len());
    let mem = memcmp_loop_program(a, b);

    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    let mut rstack = [0u32; 8];
    let mut vstack = [false; 4096];
    let mut handler = TestHandler { enabled };

    ert_emit(
        &mut handler,
        RawMemory::from(mem.as_slice()),
        &mut rstack,
        &mut vstack,
        0,
        &mut regs,
        &mut constants,
        false,
        true,
    )?;

    // The recognizer's mux clears `reg_consts[T2]` once a mismatch could
    // have flipped it (the result is now genuinely data-dependent), so the
    // actual boolean lives in `regs[T2]`'s bit pattern, not the concrete
    // shadow.
    Ok(value(&regs[Reg::T2.0 as usize]))
}

#[test]
fn disabled_by_default_hard_errors_on_the_secret_dependent_compare() {
    assert!(matches!(
        run_memcmp(b"abcd", b"abcd", false),
        Err(ErtError::Unexpected)
    ));
}

#[test]
fn enabled_recognizes_equal_buffers_across_every_length() {
    for len in 1..=8usize {
        let data = std::vec![1u8; len];
        let result = match run_memcmp(&data, &data, true) {
            Ok(result) => result,
            Err(_) => panic!("recognized loop should not error, len={len}"),
        };
        assert_eq!(result, 1, "len={len}");
    }
}

#[test]
fn enabled_recognizes_a_mismatch_at_every_position() {
    let len = 6usize;
    for mismatch_at in 0..len {
        let a: Vec<u8> = (0..len as u8).collect();
        let mut b = a.clone();
        b[mismatch_at] = b[mismatch_at].wrapping_add(1);
        let result = match run_memcmp(&a, &b, true) {
            Ok(result) => result,
            Err(_) => panic!("recognized loop should not error, mismatch_at={mismatch_at}"),
        };
        assert_eq!(result, 0, "mismatch_at={mismatch_at}");
    }
}

/// Two independent secret-dependent branches in one loop body: the idiom
/// contract requires exactly one candidate, so this must keep hard-erroring
/// even with the recognizer enabled.
fn two_candidate_branches_program(a: &[u8], b: &[u8]) -> Vec<u8> {
    let len = a.len() as i32;
    let mut instrs = Vec::new();

    instrs.push(Inst::Addi {
        imm: Imm::new_i32(-2 * len),
        dest: Reg::SP,
        src1: Reg::SP,
    });
    for (k, &byte) in a.iter().enumerate() {
        instrs.push(Inst::Addi {
            imm: Imm::new_i32(byte as i32),
            dest: Reg::S0,
            src1: Reg::ZERO,
        });
        instrs.push(Inst::Sb {
            offset: Imm::new_i32(k as i32),
            src: Reg::S0,
            base: Reg::SP,
        });
    }
    for (k, &byte) in b.iter().enumerate() {
        instrs.push(Inst::Addi {
            imm: Imm::new_i32(byte as i32),
            dest: Reg::S0,
            src1: Reg::ZERO,
        });
        instrs.push(Inst::Sb {
            offset: Imm::new_i32(len + k as i32),
            src: Reg::S0,
            base: Reg::SP,
        });
    }

    instrs.push(Inst::Addi {
        imm: Imm::new_i32(0),
        dest: Reg::T0,
        src1: Reg::ZERO,
    }); // i = 0
    instrs.push(Inst::Addi {
        imm: Imm::new_i32(len),
        dest: Reg::T1,
        src1: Reg::ZERO,
    }); // n = len
    instrs.push(Inst::Addi {
        imm: Imm::new_i32(len),
        dest: Reg::S1,
        src1: Reg::SP,
    }); // base_b = sp + len

    let h = instrs.len();
    instrs.push(Inst::Add {
        dest: Reg::T3,
        src1: Reg::SP,
        src2: Reg::T0,
    });
    instrs.push(Inst::Add {
        dest: Reg::T4,
        src1: Reg::S1,
        src2: Reg::T0,
    });
    instrs.push(Inst::Lbu {
        offset: Imm::ZERO,
        dest: Reg::T5,
        base: Reg::T3,
    });
    instrs.push(Inst::Lbu {
        offset: Imm::ZERO,
        dest: Reg::T6,
        base: Reg::T4,
    });

    let beq = instrs.len();
    instrs.push(Inst::Beq {
        offset: Imm::ZERO,
        src1: Reg::T5,
        src2: Reg::T6,
    }); // first candidate branch, patched below

    let bne = instrs.len();
    instrs.push(Inst::Bne {
        offset: Imm::ZERO,
        src1: Reg::T5,
        src2: Reg::T6,
    }); // second (unreachable) candidate branch, patched below

    let cont = instrs.len();
    instrs.push(Inst::Addi {
        imm: Imm::new_i32(1),
        dest: Reg::T0,
        src1: Reg::T0,
    });

    let blt = instrs.len();
    instrs.push(Inst::Blt {
        offset: Imm::ZERO,
        src1: Reg::T0,
        src2: Reg::T1,
    }); // patched below

    let done = instrs.len();
    instrs.push(Inst::Addi {
        imm: Imm::new_i32(-1),
        dest: Reg::A0,
        src1: Reg::ZERO,
    });
    instrs.push(Inst::Ecall);

    instrs[beq] = Inst::Beq {
        offset: Imm::new_i32(addr(cont) - addr(beq)),
        src1: Reg::T5,
        src2: Reg::T6,
    };
    instrs[bne] = Inst::Bne {
        offset: Imm::new_i32(addr(cont) - addr(bne)),
        src1: Reg::T5,
        src2: Reg::T6,
    };
    instrs[blt] = Inst::Blt {
        offset: Imm::new_i32(addr(h) - addr(blt)),
        src1: Reg::T0,
        src2: Reg::T1,
    };
    let _ = done;

    program(&instrs)
}

#[test]
fn a_second_branch_on_the_continue_path_is_rejected() {
    let mem = two_candidate_branches_program(&[1, 2], &[1, 2]);
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    let mut rstack = [0u32; 8];
    let mut vstack = [false; 4096];
    let mut handler = TestHandler { enabled: true };

    let result = ert_emit(
        &mut handler,
        RawMemory::from(mem.as_slice()),
        &mut rstack,
        &mut vstack,
        0,
        &mut regs,
        &mut constants,
        false,
        true,
    );

    assert!(matches!(result, Err(ErtError::Unexpected)));
}
