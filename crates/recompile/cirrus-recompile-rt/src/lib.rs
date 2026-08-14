#![warn(missing_docs)]

//! Pinned runtime functions: the fixed ABI every code-generating backend
//! (assembly, LLVM, Rust) targets.
//!
//! A backend never manipulates a circuit value directly; it only ever
//! manipulates [`Idx`](cirrus_recompile_core::Idx)es into a scratch buffer,
//! and defers the actual Boolean operation to one of five pinned functions
//! -- `create`/`bitand`/`bitor`/`bitxor`/`mux` -- that read and write the
//! buffer by index. [`define_pinned_backend!`] generates that set of five
//! functions for any concrete `Backend: ContextWithBitAnd<bool> + ... `
//! (`cirrus-core`'s own `Context` trait family), so the *same* recompiled
//! artifact's call sites work unchanged no matter which backend they are
//! linked/mapped against -- swap which module's functions a compiled
//! program calls, and the identical machine code, LLVM IR, or Rust source
//! runs natively, garbles a circuit, or evaluates one.
//!
//! Every pinned function takes the backend and the buffer as two separate
//! pointer parameters (`backend: *mut Backend`, `buf: *mut Backend::Wrapped`)
//! -- the backend carries whatever mutable state that `Context` needs (a
//! garbler's table sink and digest seed, an evaluator's table stream, ...),
//! while the buffer is a plain array of `Backend::Wrapped` values, one slot
//! per [`Op`](cirrus_recompile_core::Op). [`plaintext`] instantiates this
//! for the native `bool` backend (`Context = ()`).
//!
//! `define_pinned_backend!` is `#[macro_export]`ed so *any* crate can define
//! its own backend the same way `plaintext` is defined here -- see
//! `cirrus-recompile-tests` for a worked example wrapping
//! `cirrus-garbled-circuit`'s `GC`/`Evaluator`. Call it as
//! `cirrus_recompile_rt::define_pinned_backend!(...)`; it references
//! `cirrus-core` through `$crate::cirrus_core`, this crate's own re-export
//! (see below), so it resolves correctly regardless of what the invoking
//! crate's own `cirrus-core` dependency (if any) happens to be named.

/// Re-exported so [`define_pinned_backend!`] can name `cirrus-core`'s traits
/// via `$crate::cirrus_core::...` from any invoking crate, hygienically --
/// it never needs the invoker to depend on `cirrus-core` under that exact
/// name (or at all).
pub use cirrus_core;

use cirrus_recompile_core::{Op, PreparedProgram, PreparedStep, Program, ScheduledOp};

/// Define one backend's pinned functions: `create`, `bitand`, `bitor`,
/// `bitxor`, `mux`, generic only over the lifetimes named in `[$lt,*]` --
/// every type/const parameter of `$backend` must already be concrete, since
/// an exported function must be monomorphic (lifetimes are erased and don't
/// participate in that, so a backend being "generic modulo lifetimes" is
/// exactly the shape this macro accepts). `$tag` names this backend's
/// exported symbols (`cirrus_rt_bitand_$tag`, etc.).
///
/// Usable from any crate: `cirrus_recompile_rt::define_pinned_backend!(pub
/// mod my_backend for ['a] MyBackend<'a> as "my_backend");`. `$backend` only
/// needs to implement `cirrus-core`'s `ContextWithBitAnd`/`ContextWithBitOr`/
/// `ContextWithBitXor`/`ContextWithCreate`/`ContextWithMux`, all over `bool`
/// -- see `plaintext`, defined with this same macro just below, for the
/// simplest possible instance.
#[macro_export]
macro_rules! define_pinned_backend {
    ($vis:vis mod $modname:ident for [$($lt:lifetime),*] $backend:ty as $tag:literal) => {
        #[doc = concat!("Pinned functions for the \"", $tag, "\" backend.")]
        $vis mod $modname {
            #[allow(unused_imports)]
            use super::*;

            /// This backend's concrete [`cirrus_core`] `Context` type.
            pub type Backend<$($lt),*> = $backend;
            /// This backend's scratch-buffer element type.
            pub type Wrapped<$($lt),*> =
                <Backend<$($lt),*> as $crate::cirrus_core::ContextWithValue<bool>>::Wrapped;

            /// Materialize a known Boolean constant into `buf[out]`.
            ///
            /// # Safety
            ///
            /// `backend` must be valid for a mutable borrow; `buf` must be
            /// valid for a write at `out`.
            #[unsafe(export_name = concat!("cirrus_rt_create_", $tag))]
            pub unsafe extern "C" fn create<$($lt),*>(
                backend: *mut Backend<$($lt),*>,
                buf: *mut Wrapped<$($lt),*>,
                val: u8,
                out: u32,
            ) {
                unsafe {
                    let result = $crate::cirrus_core::ContextWithCreate::create(&mut *backend, val != 0)
                        .unwrap_or_else(|_| panic!(concat!("cirrus_rt_create_", $tag, " failed")));
                    buf.add(out as usize).write(result);
                }
            }

            /// `buf[out] = backend.bitand(buf[a], buf[b])`.
            ///
            /// # Safety
            ///
            /// `backend` must be valid for a mutable borrow; `buf` must be
            /// valid for reads at `a`, `b` and a write at `out`.
            #[unsafe(export_name = concat!("cirrus_rt_bitand_", $tag))]
            pub unsafe extern "C" fn bitand<$($lt),*>(
                backend: *mut Backend<$($lt),*>,
                buf: *mut Wrapped<$($lt),*>,
                a: u32,
                b: u32,
                out: u32,
            ) {
                unsafe {
                    let av = buf.add(a as usize).read();
                    let bv = buf.add(b as usize).read();
                    let result = $crate::cirrus_core::ContextWithBitAnd::bitand(&mut *backend, av, bv)
                        .unwrap_or_else(|_| panic!(concat!("cirrus_rt_bitand_", $tag, " failed")));
                    buf.add(out as usize).write(result);
                }
            }

            /// `buf[out] = backend.bitor(buf[a], buf[b])`.
            ///
            /// # Safety
            ///
            /// `backend` must be valid for a mutable borrow; `buf` must be
            /// valid for reads at `a`, `b` and a write at `out`.
            #[unsafe(export_name = concat!("cirrus_rt_bitor_", $tag))]
            pub unsafe extern "C" fn bitor<$($lt),*>(
                backend: *mut Backend<$($lt),*>,
                buf: *mut Wrapped<$($lt),*>,
                a: u32,
                b: u32,
                out: u32,
            ) {
                unsafe {
                    let av = buf.add(a as usize).read();
                    let bv = buf.add(b as usize).read();
                    let result = $crate::cirrus_core::ContextWithBitOr::bitor(&mut *backend, av, bv)
                        .unwrap_or_else(|_| panic!(concat!("cirrus_rt_bitor_", $tag, " failed")));
                    buf.add(out as usize).write(result);
                }
            }

            /// `buf[out] = backend.bitxor(buf[a], buf[b])`.
            ///
            /// # Safety
            ///
            /// `backend` must be valid for a mutable borrow; `buf` must be
            /// valid for reads at `a`, `b` and a write at `out`.
            #[unsafe(export_name = concat!("cirrus_rt_bitxor_", $tag))]
            pub unsafe extern "C" fn bitxor<$($lt),*>(
                backend: *mut Backend<$($lt),*>,
                buf: *mut Wrapped<$($lt),*>,
                a: u32,
                b: u32,
                out: u32,
            ) {
                unsafe {
                    let av = buf.add(a as usize).read();
                    let bv = buf.add(b as usize).read();
                    let result = $crate::cirrus_core::ContextWithBitXor::bitxor(&mut *backend, av, bv)
                        .unwrap_or_else(|_| panic!(concat!("cirrus_rt_bitxor_", $tag, " failed")));
                    buf.add(out as usize).write(result);
                }
            }

            /// `buf[out] = backend.mux(buf[cond], buf[then], buf[r#else])`.
            ///
            /// # Safety
            ///
            /// `backend` must be valid for a mutable borrow; `buf` must be
            /// valid for reads at `cond`, `then`, `r#else` and a write at
            /// `out`.
            #[unsafe(export_name = concat!("cirrus_rt_mux_", $tag))]
            pub unsafe extern "C" fn mux<$($lt),*>(
                backend: *mut Backend<$($lt),*>,
                buf: *mut Wrapped<$($lt),*>,
                cond: u32,
                then: u32,
                r#else: u32,
                out: u32,
            ) {
                unsafe {
                    let cv = buf.add(cond as usize).read();
                    let tv = buf.add(then as usize).read();
                    let ev = buf.add(r#else as usize).read();
                    let result = $crate::cirrus_core::ContextWithMux::mux(&mut *backend, cv, tv, ev)
                        .unwrap_or_else(|_| panic!(concat!("cirrus_rt_mux_", $tag, " failed")));
                    buf.add(out as usize).write(result);
                }
            }
        }
    };
}

define_pinned_backend!(pub mod plaintext for [] () as "plaintext");

/// Run a whole [`Program`] directly against a generic backend's pinned
/// functions -- the reference "does this backend's pinned ABI itself
/// preserve semantics" check every backend-specific execution test compares
/// its own lowering's output against.
///
/// `Backend::Wrapped` must be `Copy`: buffer slots are read and written
/// freely (never dropped), matching every source type this project defines
/// (`bool`, a garbled-circuit `Label<N>`, an evaluator's `[u8; N]`).
pub fn execute<Backend>(
    backend: &mut Backend,
    program: &Program,
    inputs: &[Backend::Wrapped],
) -> Vec<Backend::Wrapped>
where
    Backend: cirrus_core::ContextWithBitAnd<bool>
        + cirrus_core::ContextWithBitOr<bool>
        + cirrus_core::ContextWithBitXor<bool>
        + cirrus_core::ContextWithCreate<bool>
        + cirrus_core::ContextWithMux<bool>,
    Backend::Wrapped: Copy,
{
    assert_eq!(
        inputs.len(),
        program.inputs.len(),
        "input count must match the recorded program's input slots"
    );
    let mut buf: Vec<Option<Backend::Wrapped>> = vec![None; program.ops.len()];
    for (&idx, &value) in program.inputs.iter().zip(inputs) {
        buf[idx.get()] = Some(value);
    }
    for (i, op) in program.ops.iter().enumerate() {
        if buf[i].is_some() {
            // Already supplied as a named input; see this crate's `plaintext`
            // module docs (and every backend's matching convention) for why
            // a recorded `Create` at an input slot is never re-run.
            continue;
        }
        let result = match *op {
            Op::Create(val) => cirrus_core::ContextWithCreate::create(backend, val)
                .unwrap_or_else(|_| panic!("cirrus_recompile_rt::execute: create failed")),
            Op::BitAnd(a, b) => cirrus_core::ContextWithBitAnd::bitand(
                backend,
                buf[a.get()].unwrap(),
                buf[b.get()].unwrap(),
            )
            .unwrap_or_else(|_| panic!("cirrus_recompile_rt::execute: bitand failed")),
            Op::BitOr(a, b) => cirrus_core::ContextWithBitOr::bitor(
                backend,
                buf[a.get()].unwrap(),
                buf[b.get()].unwrap(),
            )
            .unwrap_or_else(|_| panic!("cirrus_recompile_rt::execute: bitor failed")),
            Op::BitXor(a, b) => cirrus_core::ContextWithBitXor::bitxor(
                backend,
                buf[a.get()].unwrap(),
                buf[b.get()].unwrap(),
            )
            .unwrap_or_else(|_| panic!("cirrus_recompile_rt::execute: bitxor failed")),
            Op::Mux { cond, then, r#else } => cirrus_core::ContextWithMux::mux(
                backend,
                buf[cond.get()].unwrap(),
                buf[then.get()].unwrap(),
                buf[r#else.get()].unwrap(),
            )
            .unwrap_or_else(|_| panic!("cirrus_recompile_rt::execute: mux failed")),
        };
        buf[i] = Some(result);
    }
    program
        .outputs
        .iter()
        .map(|idx| buf[idx.get()].unwrap())
        .collect()
}

/// Run a [`PreparedProgram`] against a generic backend.
///
/// This is the reference executor for table-loop lowering.  It uses the same
/// pinned-operation semantics as [`execute`], but resolves each loop row to
/// the raw program's absolute scratch slots before calling the backend.
pub fn execute_prepared<Backend>(
    backend: &mut Backend,
    program: &PreparedProgram,
    inputs: &[Backend::Wrapped],
) -> Vec<Backend::Wrapped>
where
    Backend: cirrus_core::ContextWithBitAnd<bool>
        + cirrus_core::ContextWithBitOr<bool>
        + cirrus_core::ContextWithBitXor<bool>
        + cirrus_core::ContextWithCreate<bool>
        + cirrus_core::ContextWithMux<bool>,
    Backend::Wrapped: Copy,
{
    assert_eq!(
        inputs.len(),
        program.inputs.len(),
        "input count must match the recorded program's input slots"
    );
    let mut buf: Vec<Option<Backend::Wrapped>> = vec![None; program.slots];
    for (&idx, &value) in program.inputs.iter().zip(inputs) {
        buf[idx.get()] = Some(value);
    }
    for step in &program.steps {
        match step {
            PreparedStep::Flat(ops) => {
                for &op in ops {
                    execute_scheduled(backend, &mut buf, op);
                }
            }
            PreparedStep::Loop(loop_step) => {
                for iteration in 0..loop_step.iterations as usize {
                    for &op in &loop_step.ops {
                        execute_scheduled(backend, &mut buf, loop_step.scheduled_op(iteration, op));
                    }
                }
            }
        }
    }
    program
        .outputs
        .iter()
        .map(|idx| buf[idx.get()].unwrap())
        .collect()
}

fn execute_scheduled<Backend>(
    backend: &mut Backend,
    buf: &mut [Option<Backend::Wrapped>],
    scheduled: ScheduledOp,
) where
    Backend: cirrus_core::ContextWithBitAnd<bool>
        + cirrus_core::ContextWithBitOr<bool>
        + cirrus_core::ContextWithBitXor<bool>
        + cirrus_core::ContextWithCreate<bool>
        + cirrus_core::ContextWithMux<bool>,
    Backend::Wrapped: Copy,
{
    if buf[scheduled.out.get()].is_some() {
        return;
    }
    let result = match scheduled.op {
        Op::Create(val) => cirrus_core::ContextWithCreate::create(backend, val)
            .unwrap_or_else(|_| panic!("cirrus_recompile_rt::execute_prepared: create failed")),
        Op::BitAnd(a, b) => cirrus_core::ContextWithBitAnd::bitand(
            backend,
            buf[a.get()].unwrap(),
            buf[b.get()].unwrap(),
        )
        .unwrap_or_else(|_| panic!("cirrus_recompile_rt::execute_prepared: bitand failed")),
        Op::BitOr(a, b) => cirrus_core::ContextWithBitOr::bitor(
            backend,
            buf[a.get()].unwrap(),
            buf[b.get()].unwrap(),
        )
        .unwrap_or_else(|_| panic!("cirrus_recompile_rt::execute_prepared: bitor failed")),
        Op::BitXor(a, b) => cirrus_core::ContextWithBitXor::bitxor(
            backend,
            buf[a.get()].unwrap(),
            buf[b.get()].unwrap(),
        )
        .unwrap_or_else(|_| panic!("cirrus_recompile_rt::execute_prepared: bitxor failed")),
        Op::Mux { cond, then, r#else } => cirrus_core::ContextWithMux::mux(
            backend,
            buf[cond.get()].unwrap(),
            buf[then.get()].unwrap(),
            buf[r#else.get()].unwrap(),
        )
        .unwrap_or_else(|_| panic!("cirrus_recompile_rt::execute_prepared: mux failed")),
    };
    buf[scheduled.out.get()] = Some(result);
}

#[cfg(test)]
mod tests {
    use super::*;
    use cirrus_core::{ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate};
    use cirrus_recompile_core::Recorder;

    fn sample_program() -> (
        Program,
        cirrus_recompile_core::Idx,
        cirrus_recompile_core::Idx,
    ) {
        let mut recorder = Recorder::new();
        let a = recorder.create(true).unwrap();
        let b = recorder.create(false).unwrap();
        let and = ContextWithBitAnd::bitand(&mut recorder, a, b).unwrap();
        let or = ContextWithBitOr::bitor(&mut recorder, a, b).unwrap();
        let xor = ContextWithBitXor::bitxor(&mut recorder, a, b).unwrap();
        (recorder.finish(vec![a, b], vec![and, or, xor]), a, b)
    }

    #[test]
    fn generic_execute_matches_the_recorder_s_own_interpreter() {
        let (program, _, _) = sample_program();
        for &(x, y) in &[(false, false), (false, true), (true, false), (true, true)] {
            let expected = cirrus_recompile_core::interpret(&program, &[x, y]);
            let actual = execute(&mut (), &program, &[x, y]);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn plaintext_pinned_functions_match_generic_execute() {
        let (program, _, _) = sample_program();
        for &(x, y) in &[(false, false), (false, true), (true, false), (true, true)] {
            let expected = execute(&mut (), &program, &[x, y]);

            let mut backend = ();
            let mut buf = vec![false; program.ops.len()];
            for (&idx, &value) in program.inputs.iter().zip([x, y].iter()) {
                buf[idx.get()] = value;
            }
            for (i, op) in program.ops.iter().enumerate() {
                let out = i as u32;
                if program.inputs.iter().any(|idx| idx.get() == i) {
                    continue;
                }
                unsafe {
                    match *op {
                        Op::Create(v) => {
                            plaintext::create(&mut backend, buf.as_mut_ptr(), v as u8, out)
                        }
                        Op::BitAnd(a, b) => {
                            plaintext::bitand(&mut backend, buf.as_mut_ptr(), a.0, b.0, out)
                        }
                        Op::BitOr(a, b) => {
                            plaintext::bitor(&mut backend, buf.as_mut_ptr(), a.0, b.0, out)
                        }
                        Op::BitXor(a, b) => {
                            plaintext::bitxor(&mut backend, buf.as_mut_ptr(), a.0, b.0, out)
                        }
                        Op::Mux { cond, then, r#else } => plaintext::mux(
                            &mut backend,
                            buf.as_mut_ptr(),
                            cond.0,
                            then.0,
                            r#else.0,
                            out,
                        ),
                    }
                }
            }
            let actual: Vec<bool> = program.outputs.iter().map(|idx| buf[idx.get()]).collect();
            assert_eq!(actual, expected);
        }
    }
}
