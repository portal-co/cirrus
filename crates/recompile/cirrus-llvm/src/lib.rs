//! The LLVM recompile backend.
//!
//! [`CompiledProgram::compile`] lowers a `cirrus_recompile_core::Program`
//! into an LLVM module -- one function that, given a pointer to a backend
//! instance and a pointer to a `program.ops.len()`-element scratch buffer,
//! performs the recorded trace by calling out to that backend's pinned
//! functions, declared as external symbols -- and JIT-compiles it via
//! `inkwell`'s `ExecutionEngine`. The caller supplies each pinned function's
//! address ([`PinnedAddresses`], mirroring `cirrus-asm`'s identically-named
//! type); this crate does not depend on any particular
//! `cirrus-recompile-rt` backend, so the *same* emitted IR calls a native
//! `bool` context, a garbled-circuit garbler, or an evaluator depending only
//! on which addresses the caller wires in with `add_global_mapping`.

use inkwell::AddressSpace;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::{Linkage, Module};
use inkwell::values::FunctionValue;
use inkwell::OptimizationLevel;

use cirrus_recompile_core::{Op, Program};

/// The address of each pinned runtime function this backend calls, by exact
/// name -- matching whichever `cirrus-recompile-rt` backend module (e.g.
/// `plaintext`, `gc`, `eval`) the caller targets.
pub struct PinnedAddresses {
    /// `create(backend, buf, val, out)`.
    pub create: usize,
    /// `bitand(backend, buf, a, b, out)`.
    pub bitand: usize,
    /// `bitor(backend, buf, a, b, out)`.
    pub bitor: usize,
    /// `bitxor(backend, buf, a, b, out)`.
    pub bitxor: usize,
    /// `mux(backend, buf, cond, then, r#else, out)`.
    pub mux: usize,
}

/// A [`Program`] compiled to an LLVM module and JIT-loaded, ready to run.
///
/// Borrows an inkwell [`Context`] the caller owns, following inkwell's usual
/// ownership pattern (a `Context` outlives every `Module`/`Builder`/JIT
/// artifact built from it).
pub struct CompiledProgram<'ctx> {
    engine: ExecutionEngine<'ctx>,
    fn_name: String,
}

impl<'ctx> CompiledProgram<'ctx> {
    /// Build an LLVM module for `program`, declaring calls to `pinned`'s
    /// functions, and JIT-compile it.
    pub fn compile(context: &'ctx Context, program: &Program, fn_name: &str, pinned: &PinnedAddresses) -> Self {
        let module = context.create_module(fn_name);
        let builder = context.create_builder();
        let declared = emit(context, &module, &builder, program, fn_name);

        let engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("create LLVM JIT execution engine");

        // Wire the declared externs to the caller's chosen backend's real,
        // in-process function pointers.
        for (function, addr) in [
            (declared.create, pinned.create),
            (declared.bitand, pinned.bitand),
            (declared.bitor, pinned.bitor),
            (declared.bitxor, pinned.bitxor),
            (declared.mux, pinned.mux),
        ] {
            engine.add_global_mapping(&function, addr);
        }

        Self {
            engine,
            fn_name: fn_name.to_string(),
        }
    }

    /// Run the JIT-compiled artifact against an arbitrary backend instance
    /// and scratch buffer.
    ///
    /// # Safety
    ///
    /// `Backend`/`Wrapped` must be exactly the types `compile`'s
    /// [`PinnedAddresses`] were taken from; `backend` must be valid for
    /// whatever mutable access that backend's pinned functions need, and
    /// `buf` must have at least as many `Wrapped` elements as `program` has
    /// ops.
    pub unsafe fn run_raw<Backend, Wrapped>(&self, backend: *mut Backend, buf: *mut Wrapped) {
        // SAFETY: forwarded to the caller's own safety obligations above;
        // `fn_name` names the single such function `compile` built.
        unsafe {
            let function: inkwell::execution_engine::JitFunction<unsafe extern "C" fn(*mut Backend, *mut Wrapped)> =
                self.engine
                    .get_function(&self.fn_name)
                    .expect("resolve the JIT-compiled function");
            function.call(backend, buf);
        }
    }

    /// Run the JIT-compiled artifact against `inputs`, in `program.inputs`
    /// order, and read back `program.outputs`. Only valid for an artifact
    /// compiled against the native `bool` (`plaintext`) backend.
    pub fn run_plaintext(&self, program: &Program, inputs: &[bool]) -> Vec<bool> {
        assert_eq!(
            inputs.len(),
            program.inputs.len(),
            "input count must match the recorded program's input slots"
        );
        let mut buf = vec![false; program.ops.len()];
        for (&idx, &value) in program.inputs.iter().zip(inputs) {
            buf[idx.get()] = value;
        }
        // SAFETY: `()`/`bool` are the `plaintext` backend's `Backend`/
        // `Wrapped` types, and `buf` has `program.ops.len()` elements.
        unsafe { self.run_raw(&mut () as *mut (), buf.as_mut_ptr()) };
        program.outputs.iter().map(|idx| buf[idx.get()]).collect()
    }
}

struct PinnedFns<'ctx> {
    create: FunctionValue<'ctx>,
    bitand: FunctionValue<'ctx>,
    bitor: FunctionValue<'ctx>,
    bitxor: FunctionValue<'ctx>,
    mux: FunctionValue<'ctx>,
}

fn declare_pinned_functions<'ctx>(context: &'ctx Context, module: &Module<'ctx>) -> PinnedFns<'ctx> {
    let void_type = context.void_type();
    let i8_type = context.i8_type();
    let i32_type = context.i32_type();
    let ptr_type = context.ptr_type(AddressSpace::default());

    // Every pinned function takes the backend and the buffer as two
    // separate opaque pointers; LLVM's opaque-pointer model means this
    // declared signature is valid for whichever concrete `Backend`/
    // `Wrapped` layout the caller's chosen `PinnedAddresses` actually point
    // at -- the pointee type never appears in the IR.
    let create_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into(), i8_type.into(), i32_type.into()], false);
    let binop_type = void_type.fn_type(
        &[ptr_type.into(), ptr_type.into(), i32_type.into(), i32_type.into(), i32_type.into()],
        false,
    );
    let mux_type = void_type.fn_type(
        &[
            ptr_type.into(),
            ptr_type.into(),
            i32_type.into(),
            i32_type.into(),
            i32_type.into(),
            i32_type.into(),
        ],
        false,
    );

    PinnedFns {
        create: module.add_function("cirrus_rt_create", create_type, Some(Linkage::External)),
        bitand: module.add_function("cirrus_rt_bitand", binop_type, Some(Linkage::External)),
        bitor: module.add_function("cirrus_rt_bitor", binop_type, Some(Linkage::External)),
        bitxor: module.add_function("cirrus_rt_bitxor", binop_type, Some(Linkage::External)),
        mux: module.add_function("cirrus_rt_mux", mux_type, Some(Linkage::External)),
    }
}

fn emit<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    program: &Program,
    fn_name: &str,
) -> PinnedFns<'ctx> {
    let pinned = declare_pinned_functions(context, module);

    let void_type = context.void_type();
    let i8_type = context.i8_type();
    let i32_type = context.i32_type();
    let ptr_type = context.ptr_type(AddressSpace::default());

    let fn_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
    let function = module.add_function(fn_name, fn_type, None);
    let entry = context.append_basic_block(function, "entry");
    builder.position_at_end(entry);
    let backend = function
        .get_nth_param(0)
        .expect("generated function takes a backend pointer and a buffer pointer")
        .into_pointer_value();
    let buf = function
        .get_nth_param(1)
        .expect("generated function takes a backend pointer and a buffer pointer")
        .into_pointer_value();

    for (i, op) in program.ops.iter().enumerate() {
        if program.inputs.iter().any(|idx| idx.get() == i) {
            // Populated by the caller before `run_raw`; see
            // `generate_source`'s matching convention in the Rust backend
            // for why.
            continue;
        }
        let out = i32_type.const_int(i as u64, false);
        match *op {
            Op::Create(v) => {
                let val = i8_type.const_int(v as u64, false);
                builder
                    .build_call(pinned.create, &[backend.into(), buf.into(), val.into(), out.into()], "")
                    .expect("build call to cirrus_rt_create");
            }
            Op::BitAnd(a, b) => {
                let a = i32_type.const_int(a.0 as u64, false);
                let b = i32_type.const_int(b.0 as u64, false);
                builder
                    .build_call(pinned.bitand, &[backend.into(), buf.into(), a.into(), b.into(), out.into()], "")
                    .expect("build call to cirrus_rt_bitand");
            }
            Op::BitOr(a, b) => {
                let a = i32_type.const_int(a.0 as u64, false);
                let b = i32_type.const_int(b.0 as u64, false);
                builder
                    .build_call(pinned.bitor, &[backend.into(), buf.into(), a.into(), b.into(), out.into()], "")
                    .expect("build call to cirrus_rt_bitor");
            }
            Op::BitXor(a, b) => {
                let a = i32_type.const_int(a.0 as u64, false);
                let b = i32_type.const_int(b.0 as u64, false);
                builder
                    .build_call(pinned.bitxor, &[backend.into(), buf.into(), a.into(), b.into(), out.into()], "")
                    .expect("build call to cirrus_rt_bitxor");
            }
            Op::Mux { cond, then, r#else } => {
                let cond = i32_type.const_int(cond.0 as u64, false);
                let then = i32_type.const_int(then.0 as u64, false);
                let r#else = i32_type.const_int(r#else.0 as u64, false);
                builder
                    .build_call(
                        pinned.mux,
                        &[backend.into(), buf.into(), cond.into(), then.into(), r#else.into(), out.into()],
                        "",
                    )
                    .expect("build call to cirrus_rt_mux");
            }
        }
    }
    builder.build_return(None).expect("build return");
    pinned
}

#[cfg(test)]
mod tests {
    use super::*;
    use cirrus_core::{ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux};
    use cirrus_recompile_core::Recorder;

    fn plaintext_pinned() -> PinnedAddresses {
        PinnedAddresses {
            create: cirrus_recompile_rt::plaintext::create as *const () as usize,
            bitand: cirrus_recompile_rt::plaintext::bitand as *const () as usize,
            bitor: cirrus_recompile_rt::plaintext::bitor as *const () as usize,
            bitxor: cirrus_recompile_rt::plaintext::bitxor as *const () as usize,
            mux: cirrus_recompile_rt::plaintext::mux as *const () as usize,
        }
    }

    #[test]
    fn compiled_program_matches_the_reference_interpreter() {
        let mut recorder = Recorder::new();
        let a = recorder.create(false).unwrap();
        let b = recorder.create(false).unwrap();
        let and = ContextWithBitAnd::bitand(&mut recorder, a, b).unwrap();
        let or = ContextWithBitOr::bitor(&mut recorder, a, b).unwrap();
        let xor = ContextWithBitXor::bitxor(&mut recorder, a, b).unwrap();
        let mux = ContextWithMux::mux(&mut recorder, a, or, xor).unwrap();
        let program = recorder.finish(vec![a, b], vec![and, or, xor, mux]);

        let context = Context::create();
        let compiled = CompiledProgram::compile(&context, &program, "cirrus_llvm_test_fn", &plaintext_pinned());

        for &(x, y) in &[(false, false), (false, true), (true, false), (true, true)] {
            let expected = cirrus_recompile_rt::execute(&mut (), &program, &[x, y]);
            let actual = compiled.run_plaintext(&program, &[x, y]);
            assert_eq!(actual, expected, "mismatch for inputs ({x}, {y})");
        }
    }
}
