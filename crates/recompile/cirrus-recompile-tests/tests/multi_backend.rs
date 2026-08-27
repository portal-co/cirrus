//! Multi-backend demonstration: the *same* recorded [`Program`], compiled
//! and run against three different pinned-function backends -- plaintext, a
//! garbled-circuit garbler, and its evaluator -- all through real,
//! `rustc`-compiled generated Rust source (the same
//! `generate_source`/`compile` pipeline every other test uses). This is the
//! "allow usage of any backend (plaintext, garbled circuit, etc.)" contract
//! made concrete: [`BackendTarget`] only names a Rust module path plus a
//! lifetime list, so the emitted call sites are identical no matter which
//! backend module ends up linked against them -- only which addresses
//! `cirrus-asm`/`cirrus-llvm` map, or which module `cirrus-rust-codegen`
//! generates a path to, changes between runs. The `gc`/`eval` backends
//! below are defined in *this* crate's `src/lib.rs`, not in
//! `cirrus-recompile-rt` -- proof that
//! `cirrus_recompile_rt::define_pinned_backend!` is usable from any crate,
//! not just the one that defines it.
//!
//! The garble/evaluate half additionally exercises "garbled circuits in
//! generated code": the tables a real generated-and-compiled artifact
//! produces while garbling are fed to a second real generated-and-compiled
//! artifact that evaluates them, and the result is checked against the same
//! program's plaintext execution -- the same round-trip check
//! `cirrus-garbled-circuit`'s own test suite uses, just with the garbling
//! and evaluation steps themselves performed by compiled code instead of by
//! directly calling `GC`/`Evaluator`.

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux,
    Pusher,
};
use cirrus_garbled_circuit::{GC, GarblingRecord, Label};
use cirrus_recompile_core::Program;
use cirrus_recompile_tests::{EvalBackend, GcBackend, LABEL_BYTES, LabelDigest};
use cirrus_rust_codegen::{BackendTarget, CompiledProgram};

struct VecPusher<T>(Vec<T>);
impl<T> Pusher<T> for VecPusher<T> {
    fn push(&mut self, x: T) {
        self.0.push(x);
    }
}

/// A small circuit exercising every op `cirrus_recompile_core::Op` has.
/// `zero`/`one` are marked as inputs alongside the real secret inputs `a`/
/// `b`: the evaluator backend cannot manufacture *any* wire's label from
/// nothing (see [`EvalBackend`]'s docs), so every constant slot must be
/// supplied externally by the caller, exactly like a genuine secret input.
fn sample_program() -> Program {
    let mut recorder = cirrus_recompile_core::Recorder::new();
    let zero = recorder.create(false).unwrap();
    let one = recorder.create(true).unwrap();
    let a = recorder.create(false).unwrap();
    let b = recorder.create(false).unwrap();
    let and = ContextWithBitAnd::bitand(&mut recorder, a, b).unwrap();
    let or = ContextWithBitOr::bitor(&mut recorder, a, b).unwrap();
    let xor = ContextWithBitXor::bitxor(&mut recorder, a, b).unwrap();
    let mux = ContextWithMux::mux(&mut recorder, a, one, zero).unwrap();
    recorder.finish(vec![zero, one, a, b], vec![and, or, xor, mux])
}

#[test]
fn rust_backend_runs_the_same_program_against_plaintext_and_garbled_circuit_backends() {
    let program = sample_program();

    let plaintext_target = BackendTarget::plaintext();
    let plaintext_compiled = CompiledProgram::compile(
        &program,
        "cirrus_recompile_tests_multi_backend_plaintext",
        &plaintext_target,
    );

    let gc_target = BackendTarget {
        module_path: "cirrus_recompile_tests::gc_backend".to_string(),
        lifetimes: vec!["'a".to_string(), "'b".to_string()],
        extra_crates: vec!["cirrus_recompile_tests".to_string()],
    };
    let gc_compiled = CompiledProgram::compile(
        &program,
        "cirrus_recompile_tests_multi_backend_gc",
        &gc_target,
    );

    let eval_target = BackendTarget {
        module_path: "cirrus_recompile_tests::eval_backend".to_string(),
        lifetimes: vec!["'a".to_string()],
        extra_crates: vec!["cirrus_recompile_tests".to_string()],
    };
    let eval_compiled = CompiledProgram::compile(
        &program,
        "cirrus_recompile_tests_multi_backend_eval",
        &eval_target,
    );

    for &(av, bv) in &[(false, false), (false, true), (true, false), (true, true)] {
        let plaintext = cirrus_recompile_core::interpret(&program, &[false, true, av, bv]);

        // --- plaintext backend: the always-available `Context = ()` path ---
        let plaintext_actual = plaintext_compiled.run_plaintext(&program, &[false, true, av, bv]);
        assert_eq!(
            plaintext_actual, plaintext,
            "plaintext backend mismatch for ({av}, {bv})"
        );

        // --- garble: run the "gc"-targeted compiled artifact for real ---
        let mut tables = VecPusher(Vec::new());
        let mut gc_backend = GcBackend::new(GC::<LabelDigest, LABEL_BYTES> {
            queue: &mut tables,
            seed: Default::default(),
            delta: Default::default(),
        });
        gc_backend.gc.delta[0] = 1;

        // The garbler's own labels for the input wires are picked by the
        // caller directly (a garbler never learns the true bit -- see
        // `Label`'s docs), exactly like a real ERT program's data inputs.
        let garbler_zero = Label::new([0x11; LABEL_BYTES]);
        let garbler_one = Label::new([0x22; LABEL_BYTES]);
        let garbler_a = Label::new([0x33; LABEL_BYTES]);
        let garbler_b = Label::new([0x44; LABEL_BYTES]);
        let mut garble_buf = vec![Label::new([0u8; LABEL_BYTES]); program.ops.len()];
        for (&idx, label) in
            program
                .inputs
                .iter()
                .zip([garbler_zero, garbler_one, garbler_a, garbler_b])
        {
            garble_buf[idx.get()] = label;
        }
        // SAFETY: `gc_backend`/`garble_buf` are exactly `gc_target`'s
        // `Backend`/`Wrapped` types, and `garble_buf` has `program.ops.len()`
        // elements.
        unsafe {
            gc_compiled.run_raw(
                &mut gc_backend as *mut GcBackend<'_, '_, LabelDigest, LABEL_BYTES>,
                garble_buf.as_mut_ptr(),
            );
        }
        let garbled: Vec<Label<LABEL_BYTES>> = program
            .outputs
            .iter()
            .map(|idx| garble_buf[idx.get()])
            .collect();

        let selected = |zero_label: Label<LABEL_BYTES>, bit: bool| -> [u8; LABEL_BYTES] {
            if bit {
                core::array::from_fn(|i| zero_label.zero_label()[i] ^ gc_backend.gc.delta[i])
            } else {
                zero_label.zero_label()
            }
        };

        // --- evaluate: run the "eval"-targeted compiled artifact for real ---
        let mut eval_backend = EvalBackend::new(cirrus_garbled_circuit::Evaluator::new(Box::new(
            tables.0.into_iter().map(GarblingRecord::Table),
        )));
        let mut eval_buf = vec![[0u8; LABEL_BYTES]; program.ops.len()];
        for (&idx, selected_label) in program.inputs.iter().zip([
            selected(garbler_zero, false),
            selected(garbler_one, true),
            selected(garbler_a, av),
            selected(garbler_b, bv),
        ]) {
            eval_buf[idx.get()] = selected_label;
        }
        // SAFETY: `eval_backend`/`eval_buf` are exactly `eval_target`'s
        // `Backend`/`Wrapped` types, and `eval_buf` has `program.ops.len()`
        // elements.
        unsafe {
            eval_compiled.run_raw(
                &mut eval_backend as *mut EvalBackend<'_, LABEL_BYTES>,
                eval_buf.as_mut_ptr(),
            );
        }
        let evaluated: Vec<[u8; LABEL_BYTES]> = program
            .outputs
            .iter()
            .map(|idx| eval_buf[idx.get()])
            .collect();

        for (i, &plaintext_bit) in plaintext.iter().enumerate() {
            assert_eq!(
                evaluated[i],
                selected(garbled[i], plaintext_bit),
                "garble/evaluate output {i} mismatch for inputs ({av}, {bv})"
            );
        }
    }
}
