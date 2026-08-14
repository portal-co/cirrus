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
use inkwell::values::{FunctionValue, GlobalValue, IntValue};
use inkwell::{IntPredicate, OptimizationLevel};

use cirrus_recompile_core::{
    OptimizationOptions, PreparedLoop, PreparedOp, PreparedProgram, PreparedSlot, Program,
    Statement, StatementRange,
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

/// An error while adding a Cirrus companion function to an LLVM module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmitError {
    message: String,
}

impl EmitError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl core::fmt::Display for EmitError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.message.fmt(formatter)
    }
}

impl std::error::Error for EmitError {}

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
) -> Result<PinnedFns<'ctx>, EmitError> {
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

    Ok(PinnedFns {
        create: reusable_pinned_function(module, "cirrus_rt_create", create_type)?,
        bitand: reusable_pinned_function(module, "cirrus_rt_bitand", binop_type)?,
        bitor: reusable_pinned_function(module, "cirrus_rt_bitor", binop_type)?,
        bitxor: reusable_pinned_function(module, "cirrus_rt_bitxor", binop_type)?,
        mux: reusable_pinned_function(module, "cirrus_rt_mux", mux_type)?,
    })
}

fn reusable_pinned_function<'ctx>(
    module: &Module<'ctx>,
    name: &str,
    type_: inkwell::types::FunctionType<'ctx>,
) -> Result<FunctionValue<'ctx>, EmitError> {
    if let Some(function) = module.get_function(name) {
        if function.get_type() != type_ {
            return Err(EmitError::new(format!(
                "pinned runtime operation `{name}` has an incompatible ABI"
            )));
        }
        return Ok(function);
    }
    Ok(module.add_function(name, type_, Some(Linkage::External)))
}

/// Add a prepared Cirrus program as `fn_name` to an existing LLVM module.
///
/// The generated function has the portable pinned ABI
/// `void (ptr backend, ptr scratch_buffer)`.  The module receives (or reuses)
/// declarations for the five untagged pinned runtime operations.  This is the
/// in-place emission seam used by the LLVM pass; [`CompiledProgram`] retains
/// the owning-JIT convenience API above.
pub fn emit_prepared_into_module<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    program: &PreparedProgram,
    fn_name: &str,
) -> Result<FunctionValue<'ctx>, EmitError> {
    program
        .validate()
        .map_err(|error| EmitError::new(format!("invalid prepared program: {error:?}")))?;
    let expected = context.void_type().fn_type(
        &[
            context.ptr_type(AddressSpace::default()).into(),
            context.ptr_type(AddressSpace::default()).into(),
        ],
        false,
    );
    if let Some(function) = module.get_function(fn_name) {
        if function.count_basic_blocks() != 0 || function.get_type() != expected {
            return Err(EmitError::new(format!(
                "LLVM module already defines incompatible Cirrus companion `{fn_name}`"
            )));
        }
    }
    Ok(emit_inner(context, module, builder, program, fn_name)?.1)
}

fn emit<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    program: &PreparedProgram,
    fn_name: &str,
) -> PinnedFns<'ctx> {
    emit_inner(context, module, builder, program, fn_name)
        .expect("new JIT module must accept the pinned runtime ABI")
        .0
}

fn emit_inner<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    program: &PreparedProgram,
    fn_name: &str,
) -> Result<(PinnedFns<'ctx>, FunctionValue<'ctx>), EmitError> {
    let pinned = declare_pinned_functions(context, module)?;

    let void_type = context.void_type();
    let i8_type = context.i8_type();
    let i32_type = context.i32_type();
    let ptr_type = context.ptr_type(AddressSpace::default());

    let fn_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
    let function = module
        .get_function(fn_name)
        .unwrap_or_else(|| module.add_function(fn_name, fn_type, None));
    assert_eq!(
        function.get_type(),
        fn_type,
        "companion declaration type must match"
    );
    assert_eq!(
        function.count_basic_blocks(),
        0,
        "companion must not already have a body"
    );
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

    program
        .validate()
        .expect("prepared program must satisfy structural invariants");
    let mut loop_names = 0usize;
    emit_range(
        context,
        module,
        builder,
        &pinned,
        function,
        backend,
        buf,
        i8_type,
        i32_type,
        program,
        program.entry,
        &[],
        i32_type.const_zero(),
        &mut loop_names,
    );
    builder.build_return(None).expect("build return");
    Ok((pinned, function))
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

#[derive(Clone, Copy)]
struct ActiveTable<'ctx> {
    table: GlobalValue<'ctx>,
    table_type: inkwell::types::ArrayType<'ctx>,
    row: IntValue<'ctx>,
    fields_per_iteration: u32,
}

#[allow(clippy::too_many_arguments)]
fn emit_range<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    pinned: &PinnedFns<'ctx>,
    function: FunctionValue<'ctx>,
    backend: inkwell::values::PointerValue<'ctx>,
    buf: inkwell::values::PointerValue<'ctx>,
    i8_type: inkwell::types::IntType<'ctx>,
    i32_type: inkwell::types::IntType<'ctx>,
    program: &PreparedProgram,
    range: StatementRange,
    active: &[ActiveTable<'ctx>],
    invocation: IntValue<'ctx>,
    loop_names: &mut usize,
) {
    for statement in &program.statements[range.start as usize..range.end as usize] {
        match statement {
            Statement::Op(op) => {
                if writes_input(program, *op) {
                    continue;
                }
                emit_prepared_op(
                    builder, pinned, backend, buf, i8_type, i32_type, *op, active,
                );
            }
            Statement::Loop(loop_step) => emit_loop(
                context, module, builder, pinned, function, backend, buf, i8_type, i32_type,
                program, loop_step, active, invocation, loop_names,
            ),
        }
    }
}

fn writes_input(program: &PreparedProgram, op: PreparedOp) -> bool {
    let out = match op {
        PreparedOp::Create { out, .. }
        | PreparedOp::BitAnd { out, .. }
        | PreparedOp::BitOr { out, .. }
        | PreparedOp::BitXor { out, .. }
        | PreparedOp::Mux { out, .. } => out,
    };
    matches!(out, PreparedSlot::Static(slot) if program.inputs.contains(&slot))
}

#[allow(clippy::too_many_arguments)]
fn emit_loop<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    pinned: &PinnedFns<'ctx>,
    function: FunctionValue<'ctx>,
    backend: inkwell::values::PointerValue<'ctx>,
    buf: inkwell::values::PointerValue<'ctx>,
    i8_type: inkwell::types::IntType<'ctx>,
    i32_type: inkwell::types::IntType<'ctx>,
    program: &PreparedProgram,
    loop_step: &PreparedLoop,
    active: &[ActiveTable<'ctx>],
    invocation: IntValue<'ctx>,
    loop_names: &mut usize,
) {
    let loop_name = *loop_names;
    *loop_names += 1;
    let values: Vec<IntValue<'ctx>> = loop_step
        .table
        .iter()
        .map(|&value| i32_type.const_int(value as u64, false))
        .collect();
    let table_type = i32_type.array_type(values.len() as u32);
    let table = module.add_global(table_type, None, &format!("cirrus_table_{loop_name}"));
    table.set_initializer(&i32_type.const_array(&values));
    table.set_constant(true);
    table.set_linkage(Linkage::Private);

    let invocation_values = loop_step
        .invocations
        .iter()
        .flat_map(|descriptor| [descriptor.first_row, descriptor.iterations])
        .map(|value| i32_type.const_int(value as u64, false))
        .collect::<Vec<_>>();
    let invocation_type = i32_type.array_type(invocation_values.len() as u32);
    let invocations = module.add_global(
        invocation_type,
        None,
        &format!("cirrus_loop_invocations_{loop_name}"),
    );
    invocations.set_initializer(&i32_type.const_array(&invocation_values));
    invocations.set_constant(true);
    invocations.set_linkage(Linkage::Private);
    let invocation_offset = builder
        .build_int_mul(
            invocation,
            i32_type.const_int(2, false),
            "cirrus_invocation_offset",
        )
        .expect("compute loop invocation offset");
    let first_row = load_table_value(
        builder,
        i32_type,
        invocations,
        invocation_type,
        invocation_offset,
    );
    let count_index = builder
        .build_int_add(
            invocation_offset,
            i32_type.const_int(1, false),
            "cirrus_invocation_count_index",
        )
        .expect("compute loop invocation count index");
    let count = load_table_value(builder, i32_type, invocations, invocation_type, count_index);

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
    let loop_active = builder
        .build_int_compare(
            IntPredicate::ULT,
            iteration.as_basic_value().into_int_value(),
            count,
            "cirrus_loop_active",
        )
        .expect("compare table-loop iteration");
    builder
        .build_conditional_branch(loop_active, body, exit)
        .expect("branch from table-loop header");
    builder.position_at_end(body);
    let row = builder
        .build_int_mul(
            iteration.as_basic_value().into_int_value(),
            i32_type.const_int(1, false),
            "cirrus_row_offset",
        )
        .expect("compute loop iteration row offset");
    let row = builder
        .build_int_add(first_row, row, "cirrus_row")
        .expect("compute loop table row");
    let mut nested = active.to_vec();
    nested.push(ActiveTable {
        table,
        table_type,
        row,
        fields_per_iteration: loop_step.fields_per_iteration,
    });
    emit_range(
        context,
        module,
        builder,
        pinned,
        function,
        backend,
        buf,
        i8_type,
        i32_type,
        program,
        loop_step.body,
        &nested,
        row,
        loop_names,
    );
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
fn emit_prepared_op<'ctx>(
    builder: &Builder<'ctx>,
    pinned: &PinnedFns<'ctx>,
    backend: inkwell::values::PointerValue<'ctx>,
    buf: inkwell::values::PointerValue<'ctx>,
    i8_type: inkwell::types::IntType<'ctx>,
    i32_type: inkwell::types::IntType<'ctx>,
    op: PreparedOp,
    active: &[ActiveTable<'ctx>],
) {
    let slot = |slot| resolve_prepared_slot(builder, i32_type, slot, active);
    match op {
        PreparedOp::Create { value, out } => {
            builder
                .build_call(
                    pinned.create,
                    &[
                        backend.into(),
                        buf.into(),
                        i8_type.const_int(value as u64, false).into(),
                        slot(out).into(),
                    ],
                    "",
                )
                .expect("build table-loop create");
        }
        PreparedOp::BitAnd { a, b, out } => emit_binop(
            builder,
            pinned.bitand,
            backend,
            buf,
            slot(a),
            slot(b),
            slot(out),
        ),
        PreparedOp::BitOr { a, b, out } => emit_binop(
            builder,
            pinned.bitor,
            backend,
            buf,
            slot(a),
            slot(b),
            slot(out),
        ),
        PreparedOp::BitXor { a, b, out } => emit_binop(
            builder,
            pinned.bitxor,
            backend,
            buf,
            slot(a),
            slot(b),
            slot(out),
        ),
        PreparedOp::Mux {
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

fn resolve_prepared_slot<'ctx>(
    builder: &Builder<'ctx>,
    i32_type: inkwell::types::IntType<'ctx>,
    slot: PreparedSlot,
    active: &[ActiveTable<'ctx>],
) -> IntValue<'ctx> {
    match slot {
        PreparedSlot::Static(slot) => i32_type.const_int(slot.0 as u64, false),
        PreparedSlot::Table { depth, field } => {
            let active = active[active.len() - 1 - depth as usize];
            let index = builder
                .build_int_mul(
                    active.row,
                    i32_type.const_int(active.fields_per_iteration as u64, false),
                    "cirrus_table_row_offset",
                )
                .expect("compute table-loop row offset");
            let index = builder
                .build_int_add(
                    index,
                    i32_type.const_int(field as u64, false),
                    "cirrus_table_index",
                )
                .expect("compute table-loop field index");
            load_table_value(builder, i32_type, active.table, active.table_type, index)
        }
    }
}

fn load_table_value<'ctx>(
    builder: &Builder<'ctx>,
    i32_type: inkwell::types::IntType<'ctx>,
    table: GlobalValue<'ctx>,
    table_type: inkwell::types::ArrayType<'ctx>,
    index: IntValue<'ctx>,
) -> IntValue<'ctx> {
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
