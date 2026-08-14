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

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::{Linkage, Module};
use inkwell::values::{FunctionValue, GlobalValue, IntValue};
use inkwell::AddressSpace;
use inkwell::{IntPredicate, OptimizationLevel};

use cirrus_recompile_core::{
    LoopOp, LoopSlot, OptimizationOptions, PreparedProgram, PreparedStep, Program, ScheduledOp,
};

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
    pub fn compile(
        context: &'ctx Context,
        program: &Program,
        fn_name: &str,
        pinned: &PinnedAddresses,
    ) -> Self {
        Self::compile_with_options(
            context,
            program,
            fn_name,
            pinned,
            &OptimizationOptions::default(),
        )
    }

    /// Build and JIT-compile `program` after preparing it with `options`.
    pub fn compile_with_options(
        context: &'ctx Context,
        program: &Program,
        fn_name: &str,
        pinned: &PinnedAddresses,
        options: &OptimizationOptions,
    ) -> Self {
        let prepared = program.prepare(options);
        Self::compile_prepared(context, &prepared, fn_name, pinned)
    }

    /// Build and JIT-compile a previously prepared program.
    pub fn compile_prepared(
        context: &'ctx Context,
        program: &PreparedProgram,
        fn_name: &str,
        pinned: &PinnedAddresses,
    ) -> Self {
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
            let function: inkwell::execution_engine::JitFunction<
                unsafe extern "C" fn(*mut Backend, *mut Wrapped),
            > = self
                .engine
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

    /// Run a prepared artifact against the native Boolean backend.
    pub fn run_prepared_plaintext(&self, program: &PreparedProgram, inputs: &[bool]) -> Vec<bool> {
        assert_eq!(
            inputs.len(),
            program.inputs.len(),
            "input count must match the recorded program's input slots"
        );
        let mut buf = vec![false; program.slots];
        for (&idx, &value) in program.inputs.iter().zip(inputs) {
            buf[idx.get()] = value;
        }
        // SAFETY: `()`/`bool` are the plaintext backend types and `buf` has
        // the prepared program's preserved raw slot width.
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

fn declare_pinned_functions<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
) -> PinnedFns<'ctx> {
    let void_type = context.void_type();
    let i8_type = context.i8_type();
    let i32_type = context.i32_type();
    let ptr_type = context.ptr_type(AddressSpace::default());

    // Every pinned function takes the backend and the buffer as two
    // separate opaque pointers; LLVM's opaque-pointer model means this
    // declared signature is valid for whichever concrete `Backend`/
    // `Wrapped` layout the caller's chosen `PinnedAddresses` actually point
    // at -- the pointee type never appears in the IR.
    let create_type = void_type.fn_type(
        &[
            ptr_type.into(),
            ptr_type.into(),
            i8_type.into(),
            i32_type.into(),
        ],
        false,
    );
    let binop_type = void_type.fn_type(
        &[
            ptr_type.into(),
            ptr_type.into(),
            i32_type.into(),
            i32_type.into(),
            i32_type.into(),
        ],
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
    program: &PreparedProgram,
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

    for (step_index, step) in program.steps.iter().enumerate() {
        match step {
            PreparedStep::Flat(ops) => {
                for &scheduled in ops {
                    if program.inputs.iter().any(|&input| input == scheduled.out) {
                        continue;
                    }
                    emit_scheduled(
                        &builder, &pinned, backend, buf, i8_type, i32_type, scheduled,
                    );
                }
            }
            PreparedStep::Loop(loop_step) => emit_table_loop(
                context, module, &builder, &pinned, function, backend, buf, i8_type, i32_type,
                loop_step, step_index,
            ),
        }
    }
    builder.build_return(None).expect("build return");
    pinned
}

fn emit_scheduled<'ctx>(
    builder: &Builder<'ctx>,
    pinned: &PinnedFns<'ctx>,
    backend: inkwell::values::PointerValue<'ctx>,
    buf: inkwell::values::PointerValue<'ctx>,
    i8_type: inkwell::types::IntType<'ctx>,
    i32_type: inkwell::types::IntType<'ctx>,
    scheduled: ScheduledOp,
) {
    let out = i32_type.const_int(scheduled.out.0 as u64, false);
    match scheduled.op {
        cirrus_recompile_core::Op::Create(value) => {
            let value = i8_type.const_int(value as u64, false);
            builder
                .build_call(
                    pinned.create,
                    &[backend.into(), buf.into(), value.into(), out.into()],
                    "",
                )
                .expect("build call to cirrus_rt_create");
        }
        cirrus_recompile_core::Op::BitAnd(a, b) => emit_binop(
            builder,
            pinned.bitand,
            backend,
            buf,
            i32_type.const_int(a.0 as u64, false),
            i32_type.const_int(b.0 as u64, false),
            out,
        ),
        cirrus_recompile_core::Op::BitOr(a, b) => emit_binop(
            builder,
            pinned.bitor,
            backend,
            buf,
            i32_type.const_int(a.0 as u64, false),
            i32_type.const_int(b.0 as u64, false),
            out,
        ),
        cirrus_recompile_core::Op::BitXor(a, b) => emit_binop(
            builder,
            pinned.bitxor,
            backend,
            buf,
            i32_type.const_int(a.0 as u64, false),
            i32_type.const_int(b.0 as u64, false),
            out,
        ),
        cirrus_recompile_core::Op::Mux { cond, then, r#else } => {
            builder
                .build_call(
                    pinned.mux,
                    &[
                        backend.into(),
                        buf.into(),
                        i32_type.const_int(cond.0 as u64, false).into(),
                        i32_type.const_int(then.0 as u64, false).into(),
                        i32_type.const_int(r#else.0 as u64, false).into(),
                        out.into(),
                    ],
                    "",
                )
                .expect("build call to cirrus_rt_mux");
        }
    }
}

fn emit_binop<'ctx>(
    builder: &Builder<'ctx>,
    function: FunctionValue<'ctx>,
    backend: inkwell::values::PointerValue<'ctx>,
    buf: inkwell::values::PointerValue<'ctx>,
    a: IntValue<'ctx>,
    b: IntValue<'ctx>,
    out: IntValue<'ctx>,
) {
    builder
        .build_call(
            function,
            &[backend.into(), buf.into(), a.into(), b.into(), out.into()],
            "",
        )
        .expect("build call to pinned binary operation");
}

#[allow(clippy::too_many_arguments)]
fn emit_table_loop<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    pinned: &PinnedFns<'ctx>,
    function: FunctionValue<'ctx>,
    backend: inkwell::values::PointerValue<'ctx>,
    buf: inkwell::values::PointerValue<'ctx>,
    i8_type: inkwell::types::IntType<'ctx>,
    i32_type: inkwell::types::IntType<'ctx>,
    loop_step: &cirrus_recompile_core::TableLoop,
    step_index: usize,
) {
    let values: Vec<IntValue<'ctx>> = loop_step
        .table
        .iter()
        .map(|&value| i32_type.const_int(value as u64, false))
        .collect();
    let table_type = i32_type.array_type(values.len() as u32);
    let table = module.add_global(table_type, None, &format!("cirrus_table_{step_index}"));
    table.set_initializer(&i32_type.const_array(&values));
    table.set_constant(true);
    table.set_linkage(Linkage::Private);

    let preheader = builder
        .get_insert_block()
        .expect("loop preheader is positioned after the previous step");
    let header = context.append_basic_block(function, "cirrus_loop_header");
    let body = context.append_basic_block(function, "cirrus_loop_body");
    let exit = context.append_basic_block(function, "cirrus_loop_exit");
    builder
        .build_unconditional_branch(header)
        .expect("branch to table-loop header");
    builder.position_at_end(header);
    let iteration = builder
        .build_phi(i32_type, "cirrus_iteration")
        .expect("build loop phi");
    iteration.add_incoming(&[(&i32_type.const_zero(), preheader)]);
    let active = builder
        .build_int_compare(
            IntPredicate::ULT,
            iteration.as_basic_value().into_int_value(),
            i32_type.const_int(loop_step.iterations as u64, false),
            "cirrus_loop_active",
        )
        .expect("compare table-loop iteration");
    builder
        .build_conditional_branch(active, body, exit)
        .expect("branch from table-loop header");
    builder.position_at_end(body);
    let row = builder
        .build_int_mul(
            iteration.as_basic_value().into_int_value(),
            i32_type.const_int(loop_step.fields_per_iteration as u64, false),
            "cirrus_row",
        )
        .expect("compute table-loop row");
    for &op in &loop_step.ops {
        emit_loop_op(
            builder, pinned, backend, buf, i8_type, i32_type, table, table_type, row, op,
        );
    }
    let next = builder
        .build_int_add(
            iteration.as_basic_value().into_int_value(),
            i32_type.const_int(1, false),
            "cirrus_next_iteration",
        )
        .expect("increment table-loop iteration");
    let body_end = builder
        .get_insert_block()
        .expect("loop body remains positioned");
    builder
        .build_unconditional_branch(header)
        .expect("backedge from table-loop body");
    iteration.add_incoming(&[(&next, body_end)]);
    builder.position_at_end(exit);
}

#[allow(clippy::too_many_arguments)]
fn emit_loop_op<'ctx>(
    builder: &Builder<'ctx>,
    pinned: &PinnedFns<'ctx>,
    backend: inkwell::values::PointerValue<'ctx>,
    buf: inkwell::values::PointerValue<'ctx>,
    i8_type: inkwell::types::IntType<'ctx>,
    i32_type: inkwell::types::IntType<'ctx>,
    table: GlobalValue<'ctx>,
    table_type: inkwell::types::ArrayType<'ctx>,
    row: IntValue<'ctx>,
    op: LoopOp,
) {
    let slot = |slot| resolve_loop_slot(builder, i32_type, table, table_type, row, slot);
    match op {
        LoopOp::Create { val, out } => {
            builder
                .build_call(
                    pinned.create,
                    &[
                        backend.into(),
                        buf.into(),
                        i8_type.const_int(val as u64, false).into(),
                        slot(out).into(),
                    ],
                    "",
                )
                .expect("build table-loop create");
        }
        LoopOp::BitAnd { a, b, out } => emit_binop(
            builder,
            pinned.bitand,
            backend,
            buf,
            slot(a),
            slot(b),
            slot(out),
        ),
        LoopOp::BitOr { a, b, out } => emit_binop(
            builder,
            pinned.bitor,
            backend,
            buf,
            slot(a),
            slot(b),
            slot(out),
        ),
        LoopOp::BitXor { a, b, out } => emit_binop(
            builder,
            pinned.bitxor,
            backend,
            buf,
            slot(a),
            slot(b),
            slot(out),
        ),
        LoopOp::Mux {
            cond,
            then,
            r#else,
            out,
        } => {
            builder
                .build_call(
                    pinned.mux,
                    &[
                        backend.into(),
                        buf.into(),
                        slot(cond).into(),
                        slot(then).into(),
                        slot(r#else).into(),
                        slot(out).into(),
                    ],
                    "",
                )
                .expect("build table-loop mux");
        }
    }
}

fn resolve_loop_slot<'ctx>(
    builder: &Builder<'ctx>,
    i32_type: inkwell::types::IntType<'ctx>,
    table: GlobalValue<'ctx>,
    table_type: inkwell::types::ArrayType<'ctx>,
    row: IntValue<'ctx>,
    slot: LoopSlot,
) -> IntValue<'ctx> {
    match slot {
        LoopSlot::Static(slot) => i32_type.const_int(slot.0 as u64, false),
        LoopSlot::Table(field) => {
            let index = builder
                .build_int_add(
                    row,
                    i32_type.const_int(field as u64, false),
                    "cirrus_table_index",
                )
                .expect("compute table-loop field index");
            let ptr = unsafe {
                builder.build_in_bounds_gep(
                    table_type,
                    table.as_pointer_value(),
                    &[i32_type.const_zero(), index],
                    "cirrus_table_slot",
                )
            }
            .expect("address table-loop field");
            builder
                .build_load(i32_type, ptr, "cirrus_table_value")
                .expect("load table-loop field")
                .into_int_value()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cirrus_core::{
        ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux,
    };
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
        let compiled = CompiledProgram::compile(
            &context,
            &program,
            "cirrus_llvm_test_fn",
            &plaintext_pinned(),
        );

        for &(x, y) in &[(false, false), (false, true), (true, false), (true, true)] {
            let expected = cirrus_recompile_rt::execute(&mut (), &program, &[x, y]);
            let actual = compiled.run_plaintext(&program, &[x, y]);
            assert_eq!(actual, expected, "mismatch for inputs ({x}, {y})");
        }
    }

    #[test]
    fn compiled_table_loop_matches_the_raw_oracle() {
        let mut recorder = Recorder::new();
        let inputs = (0..32)
            .map(|_| recorder.create(false).unwrap())
            .collect::<Vec<_>>();
        let outputs = (0..16)
            .map(|index| {
                recorder
                    .bitand(inputs[index * 2], inputs[index * 2 + 1])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let program = recorder.finish(inputs, outputs);
        let prepared = program.prepare(&OptimizationOptions::default());
        assert!(prepared.has_loops());

        let context = Context::create();
        let compiled = CompiledProgram::compile_prepared(
            &context,
            &prepared,
            "cirrus_llvm_table_loop_test_fn",
            &plaintext_pinned(),
        );
        let inputs = [
            true, false, true, true, false, false, true, true, true, false, true, true, false,
            true, false, false, true, true, false, true, true, true, false, false, true, false,
            true, true, false, true, true, true,
        ];
        assert_eq!(
            compiled.run_prepared_plaintext(&prepared, &inputs),
            cirrus_recompile_core::interpret(&program, &inputs),
        );
    }
}
