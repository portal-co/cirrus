//! Execution test for the AArch64 recompile backend: compile a recorded
//! trace to real machine code, run it from freshly allocated executable
//! memory, and check it reproduces the reference interpreter's output.
//!
//! This host is Apple Silicon (aarch64-apple-darwin), so this is the one of
//! the three new backends whose *compiled artifact* can be executed
//! natively here rather than only structurally validated -- the LLVM and
//! Rust backends are validated the same way, but via their own in-process
//! JIT / `rustc`-compiled dylib rather than raw JIT'd bytes.

#![cfg(target_arch = "aarch64")]

use cirrus_asm::recompile::{PinnedAddresses, compile_aarch64, compile_prepared_aarch64};
use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux,
};
use cirrus_recompile_core::{
    Idx, LoopInvocation, OptimizationOptions, PreparedLoop, PreparedOp, PreparedProgram,
    PreparedSlot, Recorder, Statement, StatementRange, interpret_prepared,
};

/// A W^X-compliant executable-memory allocation: written while
/// `PROT_READ|PROT_WRITE`, then made `PROT_READ|PROT_EXEC` and never
/// writable again -- no `MAP_JIT`/toggling needed since this buffer is
/// written exactly once and never modified after it becomes executable.
struct ExecMem {
    ptr: *mut libc::c_void,
    len: usize,
}

impl ExecMem {
    fn new(code: &[u8]) -> Self {
        let len = code.len();
        // SAFETY: a fresh anonymous private mapping; `len` is nonzero for
        // every program this test compiles.
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
        // SAFETY: `ptr` was just mapped read-write for exactly `len` bytes.
        unsafe { core::ptr::copy_nonoverlapping(code.as_ptr(), ptr as *mut u8, len) };
        // SAFETY: dropping write permission after the copy above, before
        // this memory is ever executed, is exactly the W^X-compliant order.
        let status = unsafe { libc::mprotect(ptr, len, libc::PROT_READ | libc::PROT_EXEC) };
        assert_eq!(status, 0, "mprotect failed");
        Self { ptr, len }
    }

    /// Call the compiled `void(void *backend, void *buf)` function this
    /// buffer holds.
    ///
    /// # Safety
    ///
    /// `self` must hold a valid AAPCS64 `void(void *, void *)` function
    /// starting at its first byte, and `backend`/`buf` must be valid for
    /// whatever reads/writes that function performs.
    unsafe fn call<Backend, Wrapped>(&self, backend: *mut Backend, buf: *mut Wrapped) {
        let function: unsafe extern "C" fn(*mut Backend, *mut Wrapped) =
            unsafe { core::mem::transmute(self.ptr) };
        unsafe { function(backend, buf) };
    }
}

impl Drop for ExecMem {
    fn drop(&mut self) {
        // SAFETY: `self.ptr`/`self.len` are exactly the mapping `new` made.
        unsafe { libc::munmap(self.ptr, self.len) };
    }
}

#[test]
fn compiled_aarch64_matches_reference_interpreter() {
    let mut recorder = Recorder::new();
    let a = recorder.create(false).unwrap();
    let b = recorder.create(false).unwrap();
    let and = ContextWithBitAnd::bitand(&mut recorder, a, b).unwrap();
    let or = ContextWithBitOr::bitor(&mut recorder, a, b).unwrap();
    let xor = ContextWithBitXor::bitxor(&mut recorder, a, b).unwrap();
    let mux = ContextWithMux::mux(&mut recorder, a, or, xor).unwrap();
    let program = recorder.finish(vec![a, b], vec![and, or, xor, mux]);

    let pinned = PinnedAddresses {
        create: cirrus_recompile_rt::plaintext::create as *const () as usize,
        bitand: cirrus_recompile_rt::plaintext::bitand as *const () as usize,
        bitor: cirrus_recompile_rt::plaintext::bitor as *const () as usize,
        bitxor: cirrus_recompile_rt::plaintext::bitxor as *const () as usize,
        mux: cirrus_recompile_rt::plaintext::mux as *const () as usize,
    };
    let code = compile_aarch64(&program, &pinned);
    let exec = ExecMem::new(&code);

    for &(x, y) in &[(false, false), (false, true), (true, false), (true, true)] {
        let expected = cirrus_recompile_rt::execute(&mut (), &program, &[x, y]).unwrap();

        let mut buf = vec![false; program.ops.len()];
        buf[a.get()] = x;
        buf[b.get()] = y;
        // SAFETY: `buf` has exactly `program.ops.len()` bools, matching
        // every buffer-slot index `code` was compiled against, and `()` is
        // the `plaintext` backend's zero-sized `Backend` type.
        unsafe { exec.call(&mut () as *mut (), buf.as_mut_ptr()) };
        let actual: Vec<bool> = program.outputs.iter().map(|idx| buf[idx.get()]).collect();

        assert_eq!(actual, expected, "mismatch for inputs ({x}, {y})");
    }
}

#[test]
fn compiled_aarch64_matches_reference_interpreter_after_slot_compaction() {
    // x1 dies before x3 is computed, so `Program::compact` should reuse its
    // physical slot for x3 -- this exercises the compacted artifact through
    // the exact same real backend the non-compacted test above uses, to
    // confirm compaction needs no `cirrus-asm`-side changes.
    let mut recorder = Recorder::new();
    let a = recorder.create(false).unwrap();
    let b = recorder.create(false).unwrap();
    let c = recorder.create(false).unwrap();
    let d = recorder.create(false).unwrap();
    let x1 = ContextWithBitAnd::bitand(&mut recorder, a, b).unwrap();
    let x2 = ContextWithBitXor::bitxor(&mut recorder, x1, c).unwrap();
    let x3 = ContextWithBitAnd::bitand(&mut recorder, c, d).unwrap();
    let out = ContextWithBitOr::bitor(&mut recorder, x2, x3).unwrap();
    let program = recorder.finish(vec![a, b, c, d], vec![out]);
    let compacted = program.compact(&OptimizationOptions::default());
    assert!(compacted.slots < program.len());

    let pinned = PinnedAddresses {
        create: cirrus_recompile_rt::plaintext::create as *const () as usize,
        bitand: cirrus_recompile_rt::plaintext::bitand as *const () as usize,
        bitor: cirrus_recompile_rt::plaintext::bitor as *const () as usize,
        bitxor: cirrus_recompile_rt::plaintext::bitxor as *const () as usize,
        mux: cirrus_recompile_rt::plaintext::mux as *const () as usize,
    };
    let exec = ExecMem::new(&compile_prepared_aarch64(&compacted, &pinned));

    for &(va, vb, vc, vd) in &[
        (false, false, false, false),
        (true, false, true, false),
        (true, true, false, true),
        (false, true, true, true),
    ] {
        let sample = [va, vb, vc, vd];
        let mut buf = vec![false; compacted.slots];
        for (&idx, &value) in compacted.inputs.iter().zip(sample.iter()) {
            buf[idx.get()] = value;
        }
        unsafe { exec.call(&mut () as *mut (), buf.as_mut_ptr()) };
        let actual: Vec<bool> = compacted.outputs.iter().map(|idx| buf[idx.get()]).collect();
        assert_eq!(actual, interpret_prepared(&compacted, &sample));
    }
}

#[test]
fn compiled_aarch64_executes_nested_table_scopes() {
    let prepared = PreparedProgram::new(
        6,
        vec![Idx(0), Idx(1)],
        vec![Idx(2), Idx(3), Idx(4), Idx(5)],
        StatementRange::new(0, 1),
        vec![
            Statement::Loop(PreparedLoop {
                body: StatementRange::new(1, 3),
                invocations: vec![LoopInvocation {
                    first_row: 0,
                    iterations: 2,
                }],
                fields_per_iteration: 1,
                table: vec![2, 3],
            }),
            Statement::Op(PreparedOp::BitXor {
                a: PreparedSlot::Static(Idx(0)),
                b: PreparedSlot::Static(Idx(1)),
                out: PreparedSlot::Table { depth: 0, field: 0 },
            }),
            Statement::Loop(PreparedLoop {
                body: StatementRange::new(3, 4),
                invocations: vec![
                    LoopInvocation {
                        first_row: 0,
                        iterations: 1,
                    },
                    LoopInvocation {
                        first_row: 1,
                        iterations: 1,
                    },
                ],
                fields_per_iteration: 1,
                table: vec![4, 5],
            }),
            Statement::Op(PreparedOp::BitAnd {
                a: PreparedSlot::Table { depth: 1, field: 0 },
                b: PreparedSlot::Static(Idx(1)),
                out: PreparedSlot::Table { depth: 0, field: 0 },
            }),
        ],
    )
    .unwrap();
    let pinned = PinnedAddresses {
        create: cirrus_recompile_rt::plaintext::create as *const () as usize,
        bitand: cirrus_recompile_rt::plaintext::bitand as *const () as usize,
        bitor: cirrus_recompile_rt::plaintext::bitor as *const () as usize,
        bitxor: cirrus_recompile_rt::plaintext::bitxor as *const () as usize,
        mux: cirrus_recompile_rt::plaintext::mux as *const () as usize,
    };
    let exec = ExecMem::new(&compile_prepared_aarch64(&prepared, &pinned));

    for &(x, y) in &[(false, false), (false, true), (true, false), (true, true)] {
        let mut buf = vec![false; prepared.slots];
        buf[0] = x;
        buf[1] = y;
        unsafe { exec.call(&mut () as *mut (), buf.as_mut_ptr()) };
        let actual: Vec<bool> = prepared.outputs.iter().map(|idx| buf[idx.get()]).collect();
        assert_eq!(actual, interpret_prepared(&prepared, &[x, y]));
    }
}
