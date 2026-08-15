//! Shared test support for `cirrus-r1cs-backend`/`cirrus-groth16`:
//! `sample_program` covers every `Op` variant with a small, fast-to-prove
//! circuit, mirroring `cirrus-recompile-tests/tests/multi_backend.rs`'s
//! own `sample_program`.

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux,
};
use cirrus_recompile_core::{Program, Recorder};

/// A program with two constant-like inputs (`zero`, `one`) and two witness
/// inputs (`a`, `b`), outputting one wire for each `Op` variant this
/// workspace defines (`and`, `or`, `xor`, `mux`).
///
/// `zero`/`one`/`a`/`b` are all recorded as `program.inputs` -- supplied
/// directly by the caller, not via `Op::Create` -- exactly like
/// `multi_backend.rs`'s `sample_program` and `cirrus-recompile-tests`'s own
/// `GcBackend`/`EvalBackend` round-trip test.
pub fn sample_program() -> Program {
    let mut recorder = Recorder::new();
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
