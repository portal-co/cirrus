//! The LLVM recompile backend.
//!
//! [`CompiledProgram::compile`] lowers a `cirrus_recompile_core::Program`
//! into an LLVM module -- one function that, given a pointer to a
//! `program.ops.len()`-byte scratch buffer, performs the recorded trace by
//! calling `cirrus-recompile-rt`'s pinned functions, declared as external
//! symbols with matching signatures -- and JIT-compiles it via `inkwell`'s
//! `ExecutionEngine`. The pinned functions are wired in with
//! `add_global_mapping` against `cirrus-recompile-rt`'s real, in-process
//! function pointers, so no separate linking step is needed the way the Rust
//! backend needs `rustc`.

use inkwell::AddressSpace;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::{Linkage, Module};
use inkwell::values::FunctionValue;
use inkwell::OptimizationLevel;

use cirrus_recompile_core::{Op, Program};

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
    /// Build an LLVM module for `program`, declaring calls to
    /// `cirrus-recompile-rt`'s pinned functions, and JIT-compile it.
    pub fn compile(context: &'ctx Context, program: &Program, fn_name: &str) -> Self {
        let module = context.create_module(fn_name);
        let builder = context.create_builder();
        emit(context, &module, &builder, program, fn_name);

        let engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("create LLVM JIT execution engine");

        // Wire the declared externs to `cirrus-recompile-rt`'s real, already
        // in-process function pointers -- the same functions the Rust and
        // assembly backends call by these same names.
        map_pinned_functions(&module, &engine);

        Self {
            engine,
            fn_name: fn_name.to_string(),
        }
    }

    /// Run the JIT-compiled artifact against `inputs`, in `program.inputs`
    /// order, and read back `program.outputs`.
    pub fn run(&self, program: &Program, inputs: &[bool]) -> Vec<bool> {
        assert_eq!(
            inputs.len(),
            program.inputs.len(),
            "input count must match the recorded program's input slots"
        );
        let mut buf = vec![0u8; program.ops.len()];
        for (&idx, &value) in program.inputs.iter().zip(inputs) {
            buf[idx.get()] = value as u8;
        }
        // SAFETY: `fn_name` names the single `void(i8*)` function `compile`
        // built, and `buf` has exactly the `program.ops.len()` bytes every
        // emitted pinned-function call assumes.
        unsafe {
            let function: inkwell::execution_engine::JitFunction<unsafe extern "C" fn(*mut u8)> =
                self.engine
                    .get_function(&self.fn_name)
                    .expect("resolve the JIT-compiled function");
            function.call(buf.as_mut_ptr());
        }
        program.outputs.iter().map(|idx| buf[idx.get()] != 0).collect()
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

    let create_type = void_type.fn_type(&[ptr_type.into(), i8_type.into(), i32_type.into()], false);
    let binop_type = void_type.fn_type(
        &[ptr_type.into(), i32_type.into(), i32_type.into(), i32_type.into()],
        false,
    );
    let mux_type = void_type.fn_type(
        &[
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

fn map_pinned_functions(module: &Module<'_>, engine: &ExecutionEngine<'_>) {
    let mappings: [(&str, usize); 5] = [
        ("cirrus_rt_create", cirrus_recompile_rt::cirrus_rt_create as *const () as usize),
        ("cirrus_rt_bitand", cirrus_recompile_rt::cirrus_rt_bitand as *const () as usize),
        ("cirrus_rt_bitor", cirrus_recompile_rt::cirrus_rt_bitor as *const () as usize),
        ("cirrus_rt_bitxor", cirrus_recompile_rt::cirrus_rt_bitxor as *const () as usize),
        ("cirrus_rt_mux", cirrus_recompile_rt::cirrus_rt_mux as *const () as usize),
    ];
    for (name, addr) in mappings {
        let function = module
            .get_function(name)
            .unwrap_or_else(|| panic!("module declares {name}"));
        engine.add_global_mapping(&function, addr);
    }
}

fn emit<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    program: &Program,
    fn_name: &str,
) {
    let pinned = declare_pinned_functions(context, module);

    let void_type = context.void_type();
    let i8_type = context.i8_type();
    let i32_type = context.i32_type();
    let ptr_type = context.ptr_type(AddressSpace::default());

    let fn_type = void_type.fn_type(&[ptr_type.into()], false);
    let function = module.add_function(fn_name, fn_type, None);
    let entry = context.append_basic_block(function, "entry");
    builder.position_at_end(entry);
    let buf = function
        .get_nth_param(0)
        .expect("generated function takes exactly one buffer-pointer argument")
        .into_pointer_value();

    for (i, op) in program.ops.iter().enumerate() {
        if program.inputs.iter().any(|idx| idx.get() == i) {
            // Populated by the caller before `run`; see `generate_source`'s
            // matching convention in the Rust backend for why.
            continue;
        }
        let out = i32_type.const_int(i as u64, false);
        match *op {
            Op::Create(v) => {
                let val = i8_type.const_int(v as u64, false);
                builder
                    .build_call(pinned.create, &[buf.into(), val.into(), out.into()], "")
                    .expect("build call to cirrus_rt_create");
            }
            Op::BitAnd(a, b) => {
                let a = i32_type.const_int(a.0 as u64, false);
                let b = i32_type.const_int(b.0 as u64, false);
                builder
                    .build_call(pinned.bitand, &[buf.into(), a.into(), b.into(), out.into()], "")
                    .expect("build call to cirrus_rt_bitand");
            }
            Op::BitOr(a, b) => {
                let a = i32_type.const_int(a.0 as u64, false);
                let b = i32_type.const_int(b.0 as u64, false);
                builder
                    .build_call(pinned.bitor, &[buf.into(), a.into(), b.into(), out.into()], "")
                    .expect("build call to cirrus_rt_bitor");
            }
            Op::BitXor(a, b) => {
                let a = i32_type.const_int(a.0 as u64, false);
                let b = i32_type.const_int(b.0 as u64, false);
                builder
                    .build_call(pinned.bitxor, &[buf.into(), a.into(), b.into(), out.into()], "")
                    .expect("build call to cirrus_rt_bitxor");
            }
            Op::Mux { cond, then, r#else } => {
                let cond = i32_type.const_int(cond.0 as u64, false);
                let then = i32_type.const_int(then.0 as u64, false);
                let r#else = i32_type.const_int(r#else.0 as u64, false);
                builder
                    .build_call(
                        pinned.mux,
                        &[buf.into(), cond.into(), then.into(), r#else.into(), out.into()],
                        "",
                    )
                    .expect("build call to cirrus_rt_mux");
            }
        }
    }
    builder.build_return(None).expect("build return");
}

#[cfg(test)]
mod tests {
    use super::*;
    use cirrus_core::{ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux};
    use cirrus_recompile_core::Recorder;

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
        let compiled = CompiledProgram::compile(&context, &program, "cirrus_llvm_test_fn");

        for &(x, y) in &[(false, false), (false, true), (true, false), (true, true)] {
            let expected = cirrus_recompile_rt::execute(&program, &[x, y]);
            let actual = compiled.run(&program, &[x, y]);
            assert_eq!(actual, expected, "mismatch for inputs ({x}, {y})");
        }
    }
}
