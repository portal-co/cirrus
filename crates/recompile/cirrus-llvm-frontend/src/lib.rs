//! Cirrus's `Recorder`/`Program` adapter over the shared, backend-agnostic
//! LLVM frontend in `volar_llvm_import_core`.
//!
//! The frontend deliberately follows the ERT execution model: data may be
//! symbolic, but control flow and memory addresses must be concrete.  The
//! request is therefore the whole trust boundary; LLVM source carries no
//! Cirrus-specific annotations and every potentially secret scalar or byte is
//! named explicitly by the caller.
//!
//! [`lower`] accepts textual LLVM assembly or bitcode, executes one selected
//! direct-call-only function, and returns the ordinary
//! `cirrus_recompile_core::Program`.  Symbolic inputs occur in program order:
//! arguments first, then globals, with bytes and integer bits in little-endian
//! order.  Outputs occur in the order of [`LowerRequest::exports`].
//!
//! Everything backend-agnostic (the actual LLVM interpreter, value model,
//! and generic types below) lives in `volar_llvm_import_core`, shared with
//! `volar-ir`. This crate is the thin `Recorder`-specific layer: an
//! [`ExecutionBackend`] impl for `Recorder`, ergonomic type aliases that
//! restore the `Recorder`/`Idx` defaults callers are used to, and the
//! `Program`/`PreparedProgram`-producing entry points.

pub use volar_llvm_import_core::{
    ArgumentBinding, ExecutionBackend, ExecutionResult, Export, FrontendError, GlobalBinding,
    HostCall, HostCallContext, HostCallSignature, HostType, LoweringLimits, ModuleInput,
    RegionBinding, RegionByte, ScalarBinding, execute_module,
};

use cirrus_recompile_core::{
    CountdownPreparation, Idx, NoPreparation, OptimizationOptions, PreparedProgram, Program,
    Recorder,
};
use inkwell::context::Context;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::module::Module;

/// One binding for every entry-function parameter, in ABI order — defaults
/// its backend to [`RecorderBackend`] for ergonomic bare use (`LowerRequest<'a>`).
pub type LowerRequest<'a, Backend = RecorderBackend> =
    volar_llvm_import_core::LowerRequest<'a, Backend>;

/// A value visible to a custom [`HostCall`] adapter — defaults its wire type
/// to cirrus's own [`Idx`].
pub type HostValue<W = Idx> = volar_llvm_import_core::HostValue<W>;

/// A concrete or symbolic fixed-width integer passed to a [`HostCall`] —
/// defaults its wire type to cirrus's own [`Idx`].
pub type HostInteger<W = Idx> = volar_llvm_import_core::HostInteger<W>;

/// The declaration registry used by [`LowerRequest`] — defaults its backend
/// to [`RecorderBackend`] for ergonomic bare use (`HostCallRegistry::new()`).
pub type HostCallRegistry<Backend = RecorderBackend> =
    volar_llvm_import_core::HostCallRegistry<Backend>;

/// Newtype wrapping cirrus's [`Recorder`] so it can implement the shared
/// frontend's [`ExecutionBackend`] trait — Rust's orphan rules require a
/// local type here, since both `Recorder` (from `cirrus_recompile_core`) and
/// `ExecutionBackend` (from `volar_llvm_import_core`) are foreign to this
/// crate. Deref/DerefMut to the inner `Recorder` for its own inherent API
/// (e.g. `finish_with`).
#[derive(Default)]
pub struct RecorderBackend(pub Recorder);

impl std::ops::Deref for RecorderBackend {
    type Target = Recorder;
    fn deref(&self) -> &Recorder {
        &self.0
    }
}

impl std::ops::DerefMut for RecorderBackend {
    fn deref_mut(&mut self) -> &mut Recorder {
        &mut self.0
    }
}

impl ExecutionBackend for RecorderBackend {
    type Wire = <Recorder as cirrus_core::ContextWithValue<bool>>::Wrapped;
    type Error = <Recorder as cirrus_core::HasError>::Error;

    fn describe_error(error: Self::Error) -> String {
        error.to_string()
    }

    fn create(&mut self, val: bool) -> Result<Self::Wire, Self::Error> {
        cirrus_core::ContextWithCreate::create(&mut self.0, val)
    }

    fn bitand(&mut self, left: Self::Wire, right: Self::Wire) -> Result<Self::Wire, Self::Error> {
        cirrus_core::ContextWithBitAnd::bitand(&mut self.0, left, right)
    }

    fn bitor(&mut self, left: Self::Wire, right: Self::Wire) -> Result<Self::Wire, Self::Error> {
        cirrus_core::ContextWithBitOr::bitor(&mut self.0, left, right)
    }

    fn bitxor(&mut self, left: Self::Wire, right: Self::Wire) -> Result<Self::Wire, Self::Error> {
        cirrus_core::ContextWithBitXor::bitxor(&mut self.0, left, right)
    }

    fn mux(
        &mut self,
        cond: Self::Wire,
        then: Self::Wire,
        r#else: Self::Wire,
    ) -> Result<Self::Wire, Self::Error> {
        cirrus_core::ContextWithMux::mux(&mut self.0, cond, then, r#else)
    }
}

/// Parse and lower a textual LLVM module or bitcode buffer.
pub fn lower(
    context: &Context,
    input: ModuleInput<'_>,
    request: &LowerRequest<'_, RecorderBackend>,
) -> Result<Program, FrontendError> {
    let module = match input {
        ModuleInput::Assembly(source) => context
            .create_module_from_ir(MemoryBuffer::create_from_memory_range_copy(
                source.as_bytes(),
                "cirrus.ll",
            ))
            .map_err(|error| FrontendError::parse(error.to_string()))?,
        ModuleInput::Bitcode(bytes) => Module::parse_bitcode_from_buffer(
            &MemoryBuffer::create_from_memory_range_copy(bytes, "cirrus.bc"),
            context,
        )
        .map_err(|error| FrontendError::parse(error.to_string()))?,
    };
    lower_module(&module, request)
}

/// Parse and lower LLVM input directly to a reoptimizable prepared artifact.
///
/// This is opt-in: [`lower`] remains the portable raw-program path and does
/// not instantiate preparation work.  Canonical source loops are therefore
/// free to be represented by the shared table-loop optimizer, while a
/// frontend can retain more specific loop structure through
/// [`PreparedProgram::new`] in a future source adapter.
pub fn lower_prepared(
    context: &Context,
    input: ModuleInput<'_>,
    request: &LowerRequest<'_, RecorderBackend>,
) -> Result<PreparedProgram, FrontendError> {
    lower_prepared_with_options(context, input, request, &OptimizationOptions::default())
}

/// Like [`lower_prepared`], with explicit preparation options.
pub fn lower_prepared_with_options(
    context: &Context,
    input: ModuleInput<'_>,
    request: &LowerRequest<'_, RecorderBackend>,
    options: &OptimizationOptions,
) -> Result<PreparedProgram, FrontendError> {
    let module = match input {
        ModuleInput::Assembly(source) => context
            .create_module_from_ir(MemoryBuffer::create_from_memory_range_copy(
                source.as_bytes(),
                "cirrus.ll",
            ))
            .map_err(|error| FrontendError::parse(error.to_string()))?,
        ModuleInput::Bitcode(bytes) => Module::parse_bitcode_from_buffer(
            &MemoryBuffer::create_from_memory_range_copy(bytes, "cirrus.bc"),
            context,
        )
        .map_err(|error| FrontendError::parse(error.to_string()))?,
    };
    lower_module_prepared_with_options(&module, request, options)
}

/// Lower an already parsed LLVM module.
pub fn lower_module<'module, 'ctx>(
    module: &'module Module<'ctx>,
    request: &LowerRequest<'_, RecorderBackend>,
) -> Result<Program, FrontendError> {
    let mut recorder = RecorderBackend(Recorder::new());
    let execution = execute_module(module, request, &mut recorder)?;
    Ok(recorder.0.finish_with::<NoPreparation>(
        execution.inputs,
        execution.outputs,
        &OptimizationOptions::default(),
    ))
}

/// Lower an already parsed module to a prepared artifact using default
/// optimization options.
pub fn lower_module_prepared<'module, 'ctx>(
    module: &'module Module<'ctx>,
    request: &LowerRequest<'_, RecorderBackend>,
) -> Result<PreparedProgram, FrontendError> {
    lower_module_prepared_with_options(module, request, &OptimizationOptions::default())
}

/// Like [`lower_module_prepared`], with explicit preparation options.
pub fn lower_module_prepared_with_options<'module, 'ctx>(
    module: &'module Module<'ctx>,
    request: &LowerRequest<'_, RecorderBackend>,
    options: &OptimizationOptions,
) -> Result<PreparedProgram, FrontendError> {
    let mut recorder = RecorderBackend(Recorder::new());
    let execution = execute_module(module, request, &mut recorder)?;
    Ok(recorder.0.finish_with::<CountdownPreparation>(execution.inputs, execution.outputs, options))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cirrus_recompile_core::interpret;

    fn bits(value: u64, width: u32) -> Vec<bool> {
        (0..width).map(|bit| (value >> bit) & 1 != 0).collect()
    }

    fn request<'a>(arguments: &'a [ArgumentBinding], exports: &'a [Export]) -> LowerRequest<'a> {
        let hosts = Box::leak(Box::new(HostCallRegistry::new()));
        LowerRequest {
            entry: "kernel",
            arguments,
            globals: &[],
            exports,
            limits: LoweringLimits::default(),
            host_calls: hosts,
        }
    }

    #[test]
    fn folds_concrete_and_orders_symbolic_bits() {
        let context = Context::create();
        let arguments = [
            ArgumentBinding::Scalar(ScalarBinding::Symbolic),
            ArgumentBinding::Scalar(ScalarBinding::Concrete(3)),
        ];
        let exports = [Export::Return];
        let program = lower(
            &context,
            ModuleInput::Assembly(
                "define i8 @kernel(i8 %x, i8 %y) { entry: %z = add i8 %x, %y ret i8 %z }",
            ),
            &request(&arguments, &exports),
        )
        .unwrap();
        assert_eq!(program.inputs.len(), 8);
        assert_eq!(
            interpret(
                &program,
                &[true, false, false, false, false, false, false, false]
            ),
            vec![false, false, true, false, false, false, false, false]
        );
    }

    #[test]
    fn rejects_symbolic_control() {
        let context = Context::create();
        let arguments = [ArgumentBinding::Scalar(ScalarBinding::Symbolic)];
        let exports = [Export::Return];
        let error = lower(&context, ModuleInput::Assembly("define i8 @kernel(i1 %x) { entry: br i1 %x, label %a, label %b a: ret i8 1 b: ret i8 2 }"), &request(&arguments, &exports)).unwrap_err();
        assert!(error.to_string().contains("symbolic branch"));
    }

    #[test]
    fn executes_concrete_phi_loops_and_direct_calls() {
        let context = Context::create();
        let arguments = [ArgumentBinding::Scalar(ScalarBinding::Symbolic)];
        let exports = [Export::Return];
        let program = lower(
            &context,
            ModuleInput::Assembly(
                "define i8 @add_one(i8 %x) { entry: %y = add i8 %x, 1 ret i8 %y }\n\
                 define i8 @kernel(i8 %x) {\n\
                 entry: br label %loop\n\
                 loop: %i = phi i8 [ 0, %entry ], [ %next, %loop ]\n\
                 %next = add i8 %i, 1\n\
                 %done = icmp eq i8 %next, 3\n\
                 br i1 %done, label %exit, label %loop\n\
                 exit: %called = call i8 @add_one(i8 %x)\n\
                 %result = add i8 %called, %next\n\
                 ret i8 %result\n\
                 }",
            ),
            &request(&arguments, &exports),
        )
        .unwrap();
        let one = [true, false, false, false, false, false, false, false];
        assert_eq!(
            interpret(&program, &one),
            [true, false, true, false, false, false, false, false],
        );
    }

    #[test]
    fn lowers_global_layout_and_explicit_memory_exports() {
        let context = Context::create();
        let arguments = [ArgumentBinding::Region(RegionBinding {
            bytes: vec![RegionByte::Symbolic, RegionByte::Concrete(0)],
            writable: true,
        })];
        let exports = [
            Export::Return,
            Export::ArgumentMemory {
                argument: 0,
                offset: 0,
                len: 2,
            },
        ];
        let program = lower(
            &context,
            ModuleInput::Assembly(
                "@g = constant [4 x i8] c\"\\01\\02\\03\\04\"\n\
                 define i8 @kernel(ptr %p) {\n\
                 entry:\n\
                 %p1 = getelementptr i8, ptr %p, i64 1\n\
                 store i8 7, ptr %p1\n\
                 %g2 = getelementptr [4 x i8], ptr @g, i64 0, i64 2\n\
                 %a = load i8, ptr %p1\n\
                 %b = load i8, ptr %g2\n\
                 %sum = add i8 %a, %b\n\
                 ret i8 %sum\n\
                 }",
            ),
            &request(&arguments, &exports),
        )
        .unwrap();
        let input = [true, false, false, false, false, false, false, false];
        let output = interpret(&program, &input);
        // Return 7 + global[2] (3), then argument bytes: symbolic byte 1,
        // and the concrete stored byte 7.
        assert_eq!(
            output,
            [
                false, true, false, true, false, false, false, false, true, false, false, false,
                false, false, false, false, true, true, true, false, false, false, false, false
            ],
        );
    }

    #[test]
    fn parses_bitcode_too() {
        let context = Context::create();
        let module = context
            .create_module_from_ir(MemoryBuffer::create_from_memory_range_copy(
                b"define i8 @kernel(i8 %x) { entry: %y = xor i8 %x, 5 ret i8 %y }",
                "bitcode-source.ll",
            ))
            .unwrap();
        let bitcode = module.write_bitcode_to_memory();
        let arguments = [ArgumentBinding::Scalar(ScalarBinding::Concrete(6))];
        let exports = [Export::Return];
        let program = lower(
            &context,
            ModuleInput::Bitcode(bitcode.as_slice()),
            &request(&arguments, &exports),
        )
        .unwrap();
        assert_eq!(
            interpret(&program, &[]),
            [true, true, false, false, false, false, false, false]
        );
    }

    #[test]
    fn opt_in_prepared_lowering_keeps_raw_slots_and_reabstracts_countdown_work() {
        let context = Context::create();
        let arguments = [
            ArgumentBinding::Scalar(ScalarBinding::Symbolic),
            ArgumentBinding::Scalar(ScalarBinding::Symbolic),
        ];
        let exports = [Export::Return];
        let request = request(&arguments, &exports);
        let source = "define i8 @kernel(i8 %x, i8 %y) {\n\
            entry: br label %loop\n\
            loop:\n\
              %counter = phi i8 [ 4, %entry ], [ %next, %loop ]\n\
              %value = add i8 %x, %y\n\
              %next = add i8 %counter, -1\n\
              %done = icmp eq i8 %next, 0\n\
              br i1 %done, label %exit, label %loop\n\
            exit:\n\
              ret i8 %value\n\
          }";
        let raw = lower(&context, ModuleInput::Assembly(source), &request).unwrap();
        let prepared = lower_prepared(&context, ModuleInput::Assembly(source), &request).unwrap();

        assert_eq!(raw.len(), prepared.slots);
        assert_eq!(raw.inputs, prepared.inputs);
        assert_eq!(raw.outputs, prepared.outputs);
        assert!(prepared.has_loops());
        let mut input = bits(0xa5, 8);
        input.extend(bits(0x17, 8));
        assert_eq!(
            interpret(&raw, &input),
            cirrus_recompile_core::interpret_prepared(&prepared, &input)
        );
    }

    struct AddOneHost;

    impl HostCall<RecorderBackend> for AddOneHost {
        fn lower(
            &self,
            backend: &mut RecorderBackend,
            context: &HostCallContext<Idx>,
            arguments: &[HostValue<Idx>],
        ) -> Result<Option<HostValue<Idx>>, FrontendError> {
            let [HostValue::Integer(value)] = arguments else {
                return Err(FrontendError::request("test host expected one integer"));
            };
            let width = match value {
                HostInteger::Concrete { width, .. } | HostInteger::Symbolic { width, .. } => *width,
            };
            let one = context.concrete(width, 1)?;
            Ok(Some(HostValue::Integer(context.add(
                backend,
                value.clone(),
                one,
            )?)))
        }
    }

    fn add_one_via_backend<Backend: ExecutionBackend<Wire = bool>>(
        backend: &mut Backend,
        context: &HostCallContext<bool>,
        arguments: &[HostValue<bool>],
    ) -> Result<Option<HostValue<bool>>, FrontendError> {
        let [HostValue::Integer(value)] = arguments else {
            return Err(FrontendError::request("test host expected one integer"));
        };
        let width = match value {
            HostInteger::Concrete { width, .. } | HostInteger::Symbolic { width, .. } => *width,
        };
        let one = context.concrete(width, 1)?;
        Ok(Some(HostValue::Integer(context.add(
            backend,
            value.clone(),
            one,
        )?)))
    }

    struct DirectAddOneHost;

    impl HostCall<()> for DirectAddOneHost {
        fn lower(
            &self,
            backend: &mut (),
            context: &HostCallContext<bool>,
            arguments: &[HostValue<bool>],
        ) -> Result<Option<HostValue<bool>>, FrontendError> {
            add_one_via_backend(backend, context, arguments)
        }
    }

    /// A small direct-execution wrapper.  Its host adapter observes the
    /// wrapper itself, proving the LLVM runner tunnels the original concrete
    /// backend through host-call lowering rather than exposing only Recorder.
    #[derive(Default)]
    struct BackendWrapper {
        inner: (),
        host_calls: usize,
        operations: usize,
    }

    impl ExecutionBackend for BackendWrapper {
        type Wire = bool;
        type Error = std::convert::Infallible;

        fn describe_error(error: Self::Error) -> String {
            match error {}
        }

        fn create(&mut self, val: bool) -> Result<bool, Self::Error> {
            self.operations += 1;
            self.inner.create(val)
        }

        fn bitand(&mut self, left: bool, right: bool) -> Result<bool, Self::Error> {
            self.operations += 1;
            self.inner.bitand(left, right)
        }

        fn bitor(&mut self, left: bool, right: bool) -> Result<bool, Self::Error> {
            self.operations += 1;
            self.inner.bitor(left, right)
        }

        fn bitxor(&mut self, left: bool, right: bool) -> Result<bool, Self::Error> {
            self.operations += 1;
            self.inner.bitxor(left, right)
        }

        fn mux(&mut self, cond: bool, then: bool, r#else: bool) -> Result<bool, Self::Error> {
            self.operations += 1;
            self.inner.mux(cond, then, r#else)
        }
    }

    struct WrappedAddOneHost;

    impl HostCall<BackendWrapper> for WrappedAddOneHost {
        fn lower(
            &self,
            backend: &mut BackendWrapper,
            context: &HostCallContext<bool>,
            arguments: &[HostValue<bool>],
        ) -> Result<Option<HostValue<bool>>, FrontendError> {
            backend.host_calls += 1;
            add_one_via_backend(backend, context, arguments)
        }
    }

    #[test]
    fn invokes_custom_host_calls_without_symbolic_plaintext() {
        let context = Context::create();
        let arguments = [ArgumentBinding::Scalar(ScalarBinding::Symbolic)];
        let exports = [Export::Return];
        let mut hosts = HostCallRegistry::new();
        hosts.register(
            "host_add_one",
            HostCallSignature::new([HostType::Integer(8)], Some(HostType::Integer(8))),
            AddOneHost,
        );
        let request = LowerRequest {
            entry: "kernel",
            arguments: &arguments,
            globals: &[],
            exports: &exports,
            limits: LoweringLimits::default(),
            host_calls: &hosts,
        };
        let program = lower(
            &context,
            ModuleInput::Assembly("declare i8 @host_add_one(i8)\n define i8 @kernel(i8 %x) { entry: %y = call i8 @host_add_one(i8 %x) ret i8 %y }"),
            &request,
        )
        .unwrap();
        assert_eq!(
            interpret(
                &program,
                &[true, false, false, false, false, false, false, false]
            ),
            [false, true, false, false, false, false, false, false],
        );
    }

    #[test]
    fn executes_in_memory_ir_with_ir_keyed_direct_and_wrapped_hosts() {
        let context = Context::create();
        let module = context
            .create_module_from_ir(MemoryBuffer::create_from_memory_range_copy(
                b"declare i8 @host_add_one(i8)\n\
                  define i8 @kernel(i8 %x) {\n\
                  entry:\n\
                    %y = call i8 @host_add_one(i8 %x)\n\
                    ret i8 %y\n\
                  }",
                "in-memory-host.ll",
            ))
            .unwrap();
        let declaration = module.get_function("host_add_one").unwrap();
        let arguments = [ArgumentBinding::Scalar(ScalarBinding::Concrete(41))];
        let exports = [Export::Return];
        let signature = HostCallSignature::new([HostType::Integer(8)], Some(HostType::Integer(8)));

        let mut recorder_hosts = HostCallRegistry::<RecorderBackend>::new();
        recorder_hosts.register_ir(declaration, signature.clone(), AddOneHost);
        let recorder_request = LowerRequest {
            entry: "kernel",
            arguments: &arguments,
            globals: &[],
            exports: &exports,
            limits: LoweringLimits::default(),
            host_calls: &recorder_hosts,
        };
        let lowered = lower_module(&module, &recorder_request).unwrap();
        assert_eq!(interpret(&lowered, &[]), bits(42, 8));

        let mut direct_hosts = HostCallRegistry::<()>::new();
        direct_hosts.register_ir(declaration, signature.clone(), DirectAddOneHost);
        let direct_request = LowerRequest {
            entry: "kernel",
            arguments: &arguments,
            globals: &[],
            exports: &exports,
            limits: LoweringLimits::default(),
            host_calls: &direct_hosts,
        };
        let direct = execute_module(&module, &direct_request, &mut ()).unwrap();
        assert!(direct.inputs.is_empty());
        assert_eq!(direct.outputs, bits(42, 8));

        let mut wrapped_hosts = HostCallRegistry::<BackendWrapper>::new();
        wrapped_hosts.register_ir(declaration, signature, WrappedAddOneHost);
        let wrapped_request = LowerRequest {
            entry: "kernel",
            arguments: &arguments,
            globals: &[],
            exports: &exports,
            limits: LoweringLimits::default(),
            host_calls: &wrapped_hosts,
        };
        let mut backend = BackendWrapper::default();
        let wrapped = execute_module(&module, &wrapped_request, &mut backend).unwrap();
        assert_eq!(wrapped.outputs, bits(42, 8));
        assert_eq!(backend.host_calls, 1);
        // The runner constructed its two public Boolean constants through the
        // wrapper; the host adapter then directly observed that same wrapper.
        assert_eq!(backend.operations, 2);
    }

    #[test]
    fn keeps_argument_inputs_before_symbolic_global_inputs() {
        let context = Context::create();
        let arguments = [ArgumentBinding::Scalar(ScalarBinding::Symbolic)];
        let globals = [GlobalBinding {
            name: "g".into(),
            region: RegionBinding {
                bytes: vec![RegionByte::Symbolic],
                writable: false,
            },
        }];
        let exports = [Export::Return];
        let hosts = HostCallRegistry::new();
        let program = lower(
            &context,
            ModuleInput::Assembly(
                "@g = global i8 0\n define i8 @kernel(i8 %x) { entry: %y = load i8, ptr @g %z = xor i8 %x, %y ret i8 %z }",
            ),
            &LowerRequest {
                entry: "kernel",
                arguments: &arguments,
                globals: &globals,
                exports: &exports,
                limits: LoweringLimits::default(),
                host_calls: &hosts,
            },
        )
        .unwrap();
        assert_eq!(program.inputs, (2..18).map(Idx).collect::<Vec<_>>());
    }

    #[test]
    fn honors_aggregate_layout_for_32_and_64_bit_target_layouts() {
        for data_layout in [
            "e-m:e-p:32:32-i64:64-n32-S64",
            "e-m:e-i64:64-i128:128-n32:64-S128",
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
        ] {
            let context = Context::create();
            let module = format!(
                "target datalayout = \"{data_layout}\"\n\
                 @g = constant {{ i8, i32 }} {{ i8 0, i32 16909060 }}\n\
                 define i32 @kernel() {{ entry: %p = getelementptr {{ i8, i32 }}, ptr @g, i64 0, i32 1 %v = load i32, ptr %p ret i32 %v }}"
            );
            let program = lower(
                &context,
                ModuleInput::Assembly(&module),
                &request(&[], &[Export::Return]),
            )
            .unwrap();
            assert_eq!(interpret(&program, &[]), bits(0x0102_0304, 32));
        }
    }

    #[test]
    fn canonicalizes_reached_signed_division_overflow_to_zero() {
        let context = Context::create();
        let program = lower(
            &context,
            ModuleInput::Assembly(
                "define i64 @kernel() { entry: %v = sdiv i64 -9223372036854775808, -1 ret i64 %v }",
            ),
            &request(&[], &[Export::Return]),
        )
        .unwrap();
        assert_eq!(interpret(&program, &[]), bits(0, 64));

        let symbolic = lower(
            &context,
            ModuleInput::Assembly(
                "define i8 @kernel(i8 %x, i8 %y) { entry: %v = sdiv i8 %x, %y ret i8 %v }",
            ),
            &request(
                &[
                    ArgumentBinding::Scalar(ScalarBinding::Symbolic),
                    ArgumentBinding::Scalar(ScalarBinding::Symbolic),
                ],
                &[Export::Return],
            ),
        )
        .unwrap();
        let mut inputs = bits(0x80, 8);
        inputs.extend(bits(0xff, 8));
        assert_eq!(interpret(&symbolic, &inputs), bits(0, 8));
    }

    #[test]
    fn rejects_unsupported_types_and_enforces_instruction_limits() {
        let context = Context::create();
        let unsupported = lower(
            &context,
            ModuleInput::Assembly(
                "define float @kernel(float %x) { entry: %y = fadd float %x, 1.0 ret float %y }",
            ),
            &request(
                &[ArgumentBinding::Scalar(ScalarBinding::Concrete(0))],
                &[Export::Return],
            ),
        )
        .unwrap_err();
        assert!(unsupported.to_string().contains("unsupported"));

        let hosts = HostCallRegistry::new();
        let limits = LoweringLimits {
            max_instructions: 0,
            ..LoweringLimits::default()
        };
        let limited = lower(
            &context,
            ModuleInput::Assembly("define i8 @kernel() { entry: ret i8 0 }"),
            &LowerRequest {
                entry: "kernel",
                arguments: &[],
                globals: &[],
                exports: &[Export::Return],
                limits,
                host_calls: &hosts,
            },
        )
        .unwrap_err();
        assert!(limited.to_string().contains("instruction limit"));
    }
}
