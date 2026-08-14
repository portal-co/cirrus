//! A bounded LLVM 22 frontend for Cirrus Boolean-circuit programs.
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

use std::collections::{BTreeMap, HashMap};
use std::convert::Infallible;
use std::ffi::CStr;
use std::fmt;

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux,
    ContextWithValue, HasError,
};
use cirrus_ert_core::{BitOp as WordBitOp, add_bits_with};
use cirrus_recompile_core::{
    CountdownPreparation, Idx, NoPreparation, OptimizationOptions, PreparedProgram, Program,
    Recorder,
};
use inkwell::IntPredicate;
use inkwell::basic_block::BasicBlock;
use inkwell::context::Context;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::module::Module;
use inkwell::targets::TargetData;
use inkwell::types::{BasicMetadataTypeEnum, BasicTypeEnum};
use inkwell::values::{
    AnyValue, AsValueRef, BasicValueEnum, CallSiteValue, FunctionValue, InstructionOpcode,
    InstructionValue, Operand, PhiValue, ValueKind,
};

/// Boolean operation context used by the in-memory LLVM runner.
///
/// `Recorder` is the lowering adapter, while `()` and user-provided wrappers
/// can execute the same LLVM IR directly.  The runner keeps concrete shadows
/// itself; this backend only sees Boolean values or wires.
pub trait ExecutionBackend:
    HasError
    + ContextWithValue<bool, Wrapped = Self::Wire>
    + ContextWithCreate<bool>
    + ContextWithBitAnd<bool>
    + ContextWithBitOr<bool>
    + ContextWithBitXor<bool>
    + ContextWithMux<bool>
{
    /// The backend's Boolean wire or direct Boolean value.
    type Wire: Clone;
    /// Convert a backend failure into a contextual diagnostic string.
    fn describe_error(error: Self::Error) -> String;
}

impl<T> ExecutionBackend for T
where
    T: ContextWithCreate<bool>
        + ContextWithBitAnd<bool>
        + ContextWithBitOr<bool>
        + ContextWithBitXor<bool>
        + ContextWithMux<bool>,
    T::Wrapped: Clone,
    T::Error: fmt::Display,
{
    type Wire = T::Wrapped;

    fn describe_error(error: Self::Error) -> String {
        error.to_string()
    }
}

/// The LLVM container supplied to [`lower`].
#[derive(Clone, Copy, Debug)]
pub enum ModuleInput<'a> {
    /// Human-readable LLVM `.ll` assembly.
    Assembly(&'a str),
    /// LLVM bitcode.
    Bitcode(&'a [u8]),
}

/// Explicit data classification and output selection for one lowering run.
pub struct LowerRequest<'a, Backend = Recorder>
where
    Backend: ExecutionBackend,
{
    /// Name of the function to execute.
    pub entry: &'a str,
    /// One binding for every entry-function parameter, in ABI order.
    pub arguments: &'a [ArgumentBinding],
    /// Bindings for mutable globals which the selected kernel can access.
    pub globals: &'a [GlobalBinding],
    /// Requested result lanes and memory ranges, in output-bit order.
    pub exports: &'a [Export],
    /// Bounded-execution limits.
    pub limits: LoweringLimits,
    /// Declared-call adapters, keyed by their LLVM declaration name.
    pub host_calls: &'a HostCallRegistry<Backend>,
}

/// Bind one entry-function parameter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArgumentBinding {
    /// An integer parameter.
    Scalar(ScalarBinding),
    /// A pointer parameter naming a concrete-address byte region.
    Region(RegionBinding),
}

/// Bind an integer parameter as concrete or wholly symbolic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarBinding {
    /// A public value used for concrete control and constant folding.
    Concrete(u64),
    /// A secret integer. Its low bit precedes its high bit in `Program::inputs`.
    Symbolic,
}

/// A concrete-address byte region used by a pointer argument or mutable global.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegionBinding {
    /// Initial byte classifications in increasing address order.
    pub bytes: Vec<RegionByte>,
    /// Whether LLVM stores into this region are allowed.
    pub writable: bool,
}

/// One initial byte in an explicitly bound region.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegionByte {
    /// A public initial byte.
    Concrete(u8),
    /// A secret initial byte, ordered low bit then high bit.
    Symbolic,
}

/// Bind a mutable LLVM global by its unmangled LLVM name.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalBinding {
    /// The global's name without the leading `@`.
    pub name: String,
    /// Its request-owned storage.
    pub region: RegionBinding,
}

/// An explicit output range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Export {
    /// The selected function's scalar integer return, low bit first.
    Return,
    /// Bytes from a request-owned pointer argument.
    ArgumentMemory {
        /// Entry parameter index.
        argument: usize,
        /// First byte in the region.
        offset: usize,
        /// Number of bytes to export.
        len: usize,
    },
    /// Bytes from a named mutable or immutable LLVM global.
    GlobalMemory {
        /// Global name without `@`.
        name: String,
        /// First byte in the global.
        offset: usize,
        /// Number of bytes to export.
        len: usize,
    },
}

/// Limits which make lowering a bounded optimized-kernel execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoweringLimits {
    /// Maximum dynamic LLVM instruction visits across direct calls.
    pub max_instructions: usize,
    /// Maximum direct calls, including intrinsic adapters.
    pub max_calls: usize,
    /// Maximum private `alloca` bytes.
    pub max_alloca_bytes: usize,
}

impl Default for LoweringLimits {
    fn default() -> Self {
        Self {
            max_instructions: 100_000,
            max_calls: 10_000,
            max_alloca_bytes: 1 << 20,
        }
    }
}

/// A value visible to a custom [`HostCall`] adapter.
///
/// A symbolic value contains only circuit slot identities; it never exposes a
/// secret plaintext bit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HostValue<W = Idx> {
    /// An LLVM integer value.
    Integer(HostInteger<W>),
    /// A concrete-address pointer into a request, global, or stack region.
    Pointer { region: usize, offset: usize },
}

/// A concrete or symbolic fixed-width integer passed to a [`HostCall`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HostInteger<W = Idx> {
    /// A public fixed-width value.
    Concrete { width: u32, value: u64 },
    /// A secret fixed-width value, low bit first.
    Symbolic { width: u32, bits: Vec<W> },
}

/// One LLVM type permitted in a [`HostCallSignature`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostType {
    /// An integer with this exact bit width.
    Integer(u32),
    /// An opaque LLVM pointer. Its address remains concrete.
    Pointer,
}

/// The exact non-varargs ABI an adapter accepts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostCallSignature {
    /// Parameter types in declaration order.
    pub parameters: Vec<HostType>,
    /// `None` for `void`, otherwise the exact result type.
    pub result: Option<HostType>,
}

impl HostCallSignature {
    /// Declare an adapter's accepted LLVM parameter and result types.
    pub fn new(parameters: impl Into<Vec<HostType>>, result: Option<HostType>) -> Self {
        Self {
            parameters: parameters.into(),
            result,
        }
    }
}

/// A user-defined lowering for an LLVM declaration.
pub trait HostCall<Backend = Recorder>: Send + Sync
where
    Backend: ExecutionBackend,
{
    /// Lower a validated, direct call without observing symbolic plaintext.
    ///
    /// Return `None` for a `void` call and `Some` for an integer or pointer
    /// result which matches the LLVM declaration's result type.
    fn lower(
        &self,
        backend: &mut Backend,
        context: &HostCallContext<Backend::Wire>,
        arguments: &[HostValue<Backend::Wire>],
    ) -> Result<Option<HostValue<Backend::Wire>>, FrontendError>;
}

/// Concrete constants and Boolean-word helpers available to a [`HostCall`].
///
/// The [`HostCall::lower`] backend parameter is deliberately separate from
/// this context, just like ERT's `ecall`/`svc` handler receives its concrete
/// generic context directly.  This lets an adapter call backend-specific
/// operations or inspect wrapper state without exposing symbolic plaintext.
pub struct HostCallContext<W> {
    zero: W,
    one: W,
}

impl<W: Clone> HostCallContext<W> {
    /// Construct a public integer.
    pub fn concrete(&self, width: u32, value: u64) -> Result<HostInteger<W>, FrontendError> {
        check_width(width)?;
        Ok(HostInteger::Concrete {
            width,
            value: value & mask(width),
        })
    }

    /// Form a bitwise XOR with constant-side simplification.
    pub fn xor<Backend: ExecutionBackend<Wire = W>>(
        &self,
        backend: &mut Backend,
        left: HostInteger<W>,
        right: HostInteger<W>,
    ) -> Result<HostInteger<W>, FrontendError> {
        self.with_engine(backend, |engine| {
            engine.bitwise(int_from_host(left)?, int_from_host(right)?, BitOp::Xor)
        })
    }

    /// Form a bitwise AND with constant-side simplification.
    pub fn and<Backend: ExecutionBackend<Wire = W>>(
        &self,
        backend: &mut Backend,
        left: HostInteger<W>,
        right: HostInteger<W>,
    ) -> Result<HostInteger<W>, FrontendError> {
        self.with_engine(backend, |engine| {
            engine.bitwise(int_from_host(left)?, int_from_host(right)?, BitOp::And)
        })
    }

    /// Form a fixed-width modular sum with constant-side simplification.
    pub fn add<Backend: ExecutionBackend<Wire = W>>(
        &self,
        backend: &mut Backend,
        left: HostInteger<W>,
        right: HostInteger<W>,
    ) -> Result<HostInteger<W>, FrontendError> {
        self.with_engine(backend, |engine| {
            Ok(engine.add(int_from_host(left)?, int_from_host(right)?))
        })
    }

    /// Select `then_value` when `condition` is set, otherwise `else_value`.
    pub fn select<Backend: ExecutionBackend<Wire = W>>(
        &self,
        backend: &mut Backend,
        condition: HostInteger<W>,
        then_value: HostInteger<W>,
        else_value: HostInteger<W>,
    ) -> Result<HostInteger<W>, FrontendError> {
        let condition = int_from_host(condition)?;
        if condition.width != 1 {
            return Err(FrontendError::request("host select condition must be i1"));
        }
        self.with_engine(backend, |engine| {
            engine.select(
                condition,
                int_from_host(then_value)?,
                int_from_host(else_value)?,
            )
        })
    }

    fn with_engine<Backend: ExecutionBackend<Wire = W>>(
        &self,
        backend: &mut Backend,
        run: impl FnOnce(&mut Engine<'_, Backend>) -> Result<SymInt<W>, FrontendError>,
    ) -> Result<HostInteger<W>, FrontendError> {
        let mut engine = Engine::from_constants(backend, self.zero.clone(), self.one.clone());
        let value = run(&mut engine)?;
        engine.check_error()?;
        Ok(engine.host_int(value))
    }
}

/// The declaration registry used by [`LowerRequest`].
///
/// Registrations made with [`HostCallRegistry::register_ir`] take precedence
/// over the name fallback. This lets a caller bind host behavior to a specific
/// in-memory declaration without depending on its textual name.
pub struct HostCallRegistry<Backend = Recorder>
where
    Backend: ExecutionBackend,
{
    calls: BTreeMap<String, RegisteredHostCall<Backend>>,
    ir_calls: HashMap<usize, RegisteredHostCall<Backend>>,
}

struct RegisteredHostCall<Backend>
where
    Backend: ExecutionBackend,
{
    signature: HostCallSignature,
    call: Box<dyn HostCall<Backend>>,
}

impl<Backend: ExecutionBackend> Default for HostCallRegistry<Backend> {
    fn default() -> Self {
        Self {
            calls: BTreeMap::new(),
            ir_calls: HashMap::new(),
        }
    }
}

impl<Backend: ExecutionBackend> HostCallRegistry<Backend> {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a signature-validated adapter. Re-registering a name replaces
    /// the old adapter.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        signature: HostCallSignature,
        call: impl HostCall<Backend> + 'static,
    ) {
        self.calls.insert(
            name.into(),
            RegisteredHostCall {
                signature,
                call: Box::new(call),
            },
        );
    }

    /// Register an adapter for this in-memory LLVM declaration.  It is keyed
    /// by IR identity, so distinct declarations with the same textual name in
    /// separate modules can safely use different adapters.
    pub fn register_ir<'ctx>(
        &mut self,
        function: FunctionValue<'ctx>,
        signature: HostCallSignature,
        call: impl HostCall<Backend> + 'static,
    ) {
        self.ir_calls.insert(
            function.as_value_ref() as usize,
            RegisteredHostCall {
                signature,
                call: Box::new(call),
            },
        );
    }

    fn get<'ctx>(&self, function: FunctionValue<'ctx>) -> Option<&RegisteredHostCall<Backend>> {
        self.ir_calls
            .get(&(function.as_value_ref() as usize))
            .or_else(|| self.calls.get(&cstr(function.get_name())))
    }
}

/// A contextual parsing, request, or execution diagnostic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontendError {
    message: String,
}

impl FrontendError {
    fn parse(message: impl Into<String>) -> Self {
        Self {
            message: format!("LLVM parse error: {}", message.into()),
        }
    }

    fn request(message: impl Into<String>) -> Self {
        Self {
            message: format!("invalid LLVM lowering request: {}", message.into()),
        }
    }

    fn unsupported(message: impl Into<String>) -> Self {
        Self {
            message: format!("unsupported LLVM operation: {}", message.into()),
        }
    }

    fn execution(message: impl Into<String>) -> Self {
        Self {
            message: format!("LLVM concrete-control execution failed: {}", message.into()),
        }
    }

    fn backend(error: impl fmt::Display) -> Self {
        Self {
            message: format!("LLVM backend operation failed: {error}"),
        }
    }
}

impl fmt::Display for FrontendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(formatter)
    }
}

impl std::error::Error for FrontendError {}

/// Parse and lower a textual LLVM module or bitcode buffer.
pub fn lower(
    context: &Context,
    input: ModuleInput<'_>,
    request: &LowerRequest<'_, Recorder>,
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
    request: &LowerRequest<'_, Recorder>,
) -> Result<PreparedProgram, FrontendError> {
    lower_prepared_with_options(context, input, request, &OptimizationOptions::default())
}

/// Like [`lower_prepared`], with explicit preparation options.
pub fn lower_prepared_with_options(
    context: &Context,
    input: ModuleInput<'_>,
    request: &LowerRequest<'_, Recorder>,
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
    request: &LowerRequest<'_, Recorder>,
) -> Result<Program, FrontendError> {
    let mut recorder = Recorder::new();
    let execution = execute_module(module, request, &mut recorder)?;
    Ok(recorder.finish_with::<NoPreparation>(
        execution.inputs,
        execution.outputs,
        &OptimizationOptions::default(),
    ))
}

/// Lower an already parsed module to a prepared artifact using default
/// optimization options.
pub fn lower_module_prepared<'module, 'ctx>(
    module: &'module Module<'ctx>,
    request: &LowerRequest<'_, Recorder>,
) -> Result<PreparedProgram, FrontendError> {
    lower_module_prepared_with_options(module, request, &OptimizationOptions::default())
}

/// Like [`lower_module_prepared`], with explicit preparation options.
pub fn lower_module_prepared_with_options<'module, 'ctx>(
    module: &'module Module<'ctx>,
    request: &LowerRequest<'_, Recorder>,
    options: &OptimizationOptions,
) -> Result<PreparedProgram, FrontendError> {
    let mut recorder = Recorder::new();
    let execution = execute_module(module, request, &mut recorder)?;
    Ok(recorder.finish_with::<CountdownPreparation>(execution.inputs, execution.outputs, options))
}

/// Execute an already parsed, in-memory LLVM module through any Boolean
/// backend.  This is the frontend's primary seam; [`lower_module`] is merely
/// the [`Recorder`] adapter which turns the resulting wires into a raw
/// [`Program`].
pub fn execute_module<'module, 'ctx, Backend: ExecutionBackend>(
    module: &'module Module<'ctx>,
    request: &LowerRequest<'_, Backend>,
    backend: &mut Backend,
) -> Result<ExecutionResult<Backend::Wire>, FrontendError> {
    let entry = module.get_function(request.entry).ok_or_else(|| {
        FrontendError::request(format!("entry function `{}` does not exist", request.entry))
    })?;
    if entry.count_basic_blocks() == 0 {
        return Err(FrontendError::request("entry function is a declaration"));
    }
    let data_layout =
        TargetData::create(module.get_data_layout().as_str().to_string_lossy().as_ref());
    let mut lowerer = Lowerer::new(module, data_layout, request, backend)?;
    let arguments = lowerer.bind_entry(entry)?;
    // Bind entry inputs before global regions so the raw Program's input order
    // is stable and follows the public request order: arguments, then globals.
    lowerer.seed_globals()?;
    lowerer.returned = lowerer.execute_function(entry, arguments)?;
    lowerer.finish()
}

/// Wires supplied and produced by [`execute_module`] in request order.
pub struct ExecutionResult<W> {
    /// Symbolic scalar and region input wires in deterministic request order.
    pub inputs: Vec<W>,
    /// Explicitly requested return and memory wires in export order.
    pub outputs: Vec<W>,
}

#[derive(Clone, Debug)]
enum Value<W = Idx> {
    Integer(SymInt<W>),
    Pointer(Pointer),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Pointer {
    region: usize,
    offset: usize,
}

#[derive(Clone, Debug)]
struct Region<W = Idx> {
    bytes: Vec<SymInt<W>>,
    writable: bool,
    label: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SymInt<W = Idx> {
    width: u32,
    value: SymRepr<W>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum SymRepr<W = Idx> {
    Concrete(u64),
    Wires(Vec<W>),
}

struct Lowerer<'module, 'ctx, 'request, 'backend, Backend: ExecutionBackend> {
    module: &'module Module<'ctx>,
    target_data: TargetData,
    request: &'request LowerRequest<'request, Backend>,
    engine: Engine<'backend, Backend>,
    regions: Vec<Region<Backend::Wire>>,
    globals: HashMap<String, usize>,
    argument_regions: Vec<Option<usize>>,
    visits: usize,
    calls: usize,
    alloca_bytes: usize,
    call_stack: Vec<String>,
    returned: Option<Value<Backend::Wire>>,
}

impl<'module, 'ctx, 'request, 'backend, Backend: ExecutionBackend>
    Lowerer<'module, 'ctx, 'request, 'backend, Backend>
{
    fn new(
        module: &'module Module<'ctx>,
        target_data: TargetData,
        request: &'request LowerRequest<'request, Backend>,
        backend: &'backend mut Backend,
    ) -> Result<Self, FrontendError> {
        Ok(Self {
            module,
            target_data,
            request,
            engine: Engine::new(backend)?,
            regions: Vec::new(),
            globals: HashMap::new(),
            argument_regions: Vec::new(),
            visits: 0,
            calls: 0,
            alloca_bytes: 0,
            call_stack: Vec::new(),
            returned: None,
        })
    }

    fn seed_globals(&mut self) -> Result<(), FrontendError> {
        for global in self.module.get_globals() {
            let name = cstr(global.get_name());
            if global.is_constant() {
                let initializer = global.get_initializer().ok_or_else(|| {
                    FrontendError::unsupported(format!(
                        "immutable global @{name} has no initializer"
                    ))
                })?;
                let bytes = self.constant_bytes(initializer)?;
                self.add_region(bytes, false, format!("global @{name}"));
                self.globals.insert(name, self.regions.len() - 1);
            } else if let Some(binding) = self
                .request
                .globals
                .iter()
                .find(|binding| binding.name == name)
            {
                let region = self.request_region(&binding.region, format!("global @{name}"))?;
                self.add_region(region, binding.region.writable, format!("global @{name}"));
                self.globals.insert(name, self.regions.len() - 1);
            }
        }
        Ok(())
    }

    fn bind_entry(
        &mut self,
        function: FunctionValue<'ctx>,
    ) -> Result<Vec<Value<Backend::Wire>>, FrontendError> {
        let parameters = function.get_params();
        if parameters.len() != self.request.arguments.len() {
            return Err(FrontendError::request(format!(
                "entry `{}` has {} parameters but {} bindings were supplied",
                cstr(function.get_name()),
                parameters.len(),
                self.request.arguments.len()
            )));
        }
        let mut args = Vec::with_capacity(parameters.len());
        self.argument_regions = vec![None; parameters.len()];
        for (index, (parameter, binding)) in
            parameters.iter().zip(self.request.arguments).enumerate()
        {
            match (parameter.get_type(), binding) {
                (BasicTypeEnum::IntType(integer), ArgumentBinding::Scalar(binding)) => {
                    let width = integer.get_bit_width();
                    self.engine.check_width(width)?;
                    args.push(Value::Integer(match binding {
                        ScalarBinding::Concrete(value) => SymInt::concrete(width, *value),
                        ScalarBinding::Symbolic => self.engine.symbolic(width),
                    }));
                }
                (BasicTypeEnum::PointerType(_), ArgumentBinding::Region(binding)) => {
                    let bytes = self.request_region(binding, format!("argument {index}"))?;
                    self.add_region(bytes, binding.writable, format!("argument {index}"));
                    let region = self.regions.len() - 1;
                    self.argument_regions[index] = Some(region);
                    args.push(Value::Pointer(Pointer { region, offset: 0 }));
                }
                (BasicTypeEnum::IntType(_), ArgumentBinding::Region(_)) => {
                    return Err(FrontendError::request(format!(
                        "argument {index} is integer but has a region binding"
                    )));
                }
                (BasicTypeEnum::PointerType(_), ArgumentBinding::Scalar(_)) => {
                    return Err(FrontendError::request(format!(
                        "argument {index} is pointer but has a scalar binding"
                    )));
                }
                (_, _) => {
                    return Err(FrontendError::unsupported(format!(
                        "argument {index} has unsupported LLVM type"
                    )));
                }
            }
        }
        Ok(args)
    }

    fn request_region(
        &mut self,
        binding: &RegionBinding,
        label: String,
    ) -> Result<Vec<SymInt<Backend::Wire>>, FrontendError> {
        binding
            .bytes
            .iter()
            .map(|byte| match byte {
                RegionByte::Concrete(value) => Ok::<SymInt<Backend::Wire>, FrontendError>(
                    SymInt::concrete(8, u64::from(*value)),
                ),
                RegionByte::Symbolic => {
                    Ok::<SymInt<Backend::Wire>, FrontendError>(self.engine.symbolic(8))
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| FrontendError::request(format!("{label}: {error}")))
    }

    fn add_region(&mut self, bytes: Vec<SymInt<Backend::Wire>>, writable: bool, label: String) {
        self.regions.push(Region {
            bytes,
            writable,
            label,
        });
    }

    fn execute_function(
        &mut self,
        function: FunctionValue<'ctx>,
        arguments: Vec<Value<Backend::Wire>>,
    ) -> Result<Option<Value<Backend::Wire>>, FrontendError> {
        let name = cstr(function.get_name());
        if self.call_stack.iter().any(|active| active == &name) {
            return Err(FrontendError::unsupported(format!(
                "recursive call to `{name}`"
            )));
        }
        if function.count_basic_blocks() == 0 {
            return self.execute_declaration(function, arguments);
        }
        if function.count_params() as usize != arguments.len() {
            return Err(FrontendError::execution(format!(
                "call to `{name}` has an invalid argument count"
            )));
        }
        self.call_stack.push(name);
        let mut values = HashMap::new();
        for (parameter, value) in function.get_params().into_iter().zip(arguments) {
            values.insert(parameter, value);
        }
        let mut block = function
            .get_first_basic_block()
            .ok_or_else(|| FrontendError::execution("defined function has no entry block"))?;
        let mut predecessor = None;
        let result = 'run: loop {
            self.apply_phis(block, predecessor, &mut values)?;
            let mut next = None;
            for instruction in block.get_instructions() {
                if instruction.get_opcode() == InstructionOpcode::Phi {
                    continue;
                }
                self.visits = self
                    .visits
                    .checked_add(1)
                    .ok_or_else(|| FrontendError::execution("instruction counter overflow"))?;
                if self.visits > self.request.limits.max_instructions {
                    return Err(FrontendError::execution(format!(
                        "instruction limit {} exceeded",
                        self.request.limits.max_instructions
                    )));
                }
                match self.execute_instruction(instruction, &mut values)? {
                    Flow::Continue => {}
                    Flow::Branch(target) => {
                        next = Some(target);
                        break;
                    }
                    Flow::Return(value) => break 'run Ok(value),
                }
            }
            let target = next.ok_or_else(|| {
                FrontendError::execution(format!(
                    "block `{}` has no terminator",
                    cstr(block.get_name())
                ))
            })?;
            predecessor = Some(block);
            block = target;
        };
        self.call_stack.pop();
        result
    }

    fn apply_phis(
        &mut self,
        block: BasicBlock<'ctx>,
        predecessor: Option<BasicBlock<'ctx>>,
        values: &mut HashMap<BasicValueEnum<'ctx>, Value<Backend::Wire>>,
    ) -> Result<(), FrontendError> {
        let mut updates = Vec::new();
        for instruction in block.get_instructions() {
            if instruction.get_opcode() != InstructionOpcode::Phi {
                break;
            }
            let phi = PhiValue::try_from(instruction)
                .map_err(|_| FrontendError::execution("malformed phi instruction"))?;
            let incoming = match predecessor {
                Some(predecessor) => phi
                    .get_incomings()
                    .find(|(_, incoming_block)| *incoming_block == predecessor)
                    .map(|(value, _)| value)
                    .ok_or_else(|| {
                        FrontendError::execution(format!(
                            "phi in `{}` has no predecessor input",
                            cstr(block.get_name())
                        ))
                    })?,
                None => {
                    if phi.count_incoming() != 1 {
                        return Err(FrontendError::execution(format!(
                            "entry phi in `{}` is ambiguous",
                            cstr(block.get_name())
                        )));
                    }
                    phi.get_incoming(0).expect("counted phi input").0
                }
            };
            updates.push((phi.as_basic_value(), self.value(incoming, values)?));
        }
        for (key, value) in updates {
            values.insert(key, value);
        }
        Ok(())
    }

    fn execute_instruction(
        &mut self,
        instruction: InstructionValue<'ctx>,
        values: &mut HashMap<BasicValueEnum<'ctx>, Value<Backend::Wire>>,
    ) -> Result<Flow<'ctx, Backend::Wire>, FrontendError> {
        let opcode = instruction.get_opcode();
        let operands = |this: &mut Self| -> Result<Vec<Value<Backend::Wire>>, FrontendError> {
            (0..instruction.get_num_operands())
                .map(|index| match instruction.get_operand(index) {
                    Some(Operand::Value(value)) => this.value(value, values),
                    Some(Operand::Block(_)) => Err(FrontendError::execution(
                        "basic block used where a value was required",
                    )),
                    None => Err(FrontendError::execution(
                        "instruction has a missing operand",
                    )),
                })
                .collect()
        };
        match opcode {
            InstructionOpcode::Add
            | InstructionOpcode::Sub
            | InstructionOpcode::Mul
            | InstructionOpcode::UDiv
            | InstructionOpcode::SDiv
            | InstructionOpcode::URem
            | InstructionOpcode::SRem
            | InstructionOpcode::And
            | InstructionOpcode::Or
            | InstructionOpcode::Xor
            | InstructionOpcode::Shl
            | InstructionOpcode::LShr
            | InstructionOpcode::AShr => {
                let operands = operands(self)?;
                let [Value::Integer(left), Value::Integer(right)] = operands.as_slice() else {
                    return Err(FrontendError::unsupported(format!(
                        "{opcode:?} requires integer operands"
                    )));
                };
                let value = match opcode {
                    InstructionOpcode::Add => self.engine.add(left.clone(), right.clone()),
                    InstructionOpcode::Sub => self.engine.sub(left.clone(), right.clone()),
                    InstructionOpcode::Mul => self.engine.mul(left.clone(), right.clone())?,
                    InstructionOpcode::UDiv => {
                        self.engine
                            .divrem(left.clone(), right.clone(), false, false)?
                            .0
                    }
                    InstructionOpcode::SDiv => {
                        self.engine
                            .divrem(left.clone(), right.clone(), true, false)?
                            .0
                    }
                    InstructionOpcode::URem => {
                        self.engine
                            .divrem(left.clone(), right.clone(), false, true)?
                            .1
                    }
                    InstructionOpcode::SRem => {
                        self.engine
                            .divrem(left.clone(), right.clone(), true, true)?
                            .1
                    }
                    InstructionOpcode::And => {
                        self.engine
                            .bitwise(left.clone(), right.clone(), BitOp::And)?
                    }
                    InstructionOpcode::Or => {
                        self.engine
                            .bitwise(left.clone(), right.clone(), BitOp::Or)?
                    }
                    InstructionOpcode::Xor => {
                        self.engine
                            .bitwise(left.clone(), right.clone(), BitOp::Xor)?
                    }
                    InstructionOpcode::Shl => {
                        self.engine
                            .shift(left.clone(), right.clone(), ShiftOp::Left)?
                    }
                    InstructionOpcode::LShr => {
                        self.engine
                            .shift(left.clone(), right.clone(), ShiftOp::LogicalRight)?
                    }
                    InstructionOpcode::AShr => {
                        self.engine
                            .shift(left.clone(), right.clone(), ShiftOp::ArithmeticRight)?
                    }
                    _ => unreachable!(),
                };
                self.assign_integer(instruction, value, values)?;
                Ok(Flow::Continue)
            }
            InstructionOpcode::ICmp => {
                let operands = operands(self)?;
                let [Value::Integer(left), Value::Integer(right)] = operands.as_slice() else {
                    return Err(FrontendError::unsupported("icmp requires integer operands"));
                };
                let predicate = instruction
                    .get_icmp_predicate()
                    .ok_or_else(|| FrontendError::execution("icmp has no predicate"))?;
                let result = self
                    .engine
                    .compare(left.clone(), right.clone(), predicate)?;
                self.assign_integer(instruction, result, values)?;
                Ok(Flow::Continue)
            }
            InstructionOpcode::Select => {
                let operands = operands(self)?;
                let [
                    Value::Integer(condition),
                    Value::Integer(then_value),
                    Value::Integer(else_value),
                ] = operands.as_slice()
                else {
                    return Err(FrontendError::unsupported(
                        "select supports only integer values",
                    ));
                };
                let result = self.engine.select(
                    condition.clone(),
                    then_value.clone(),
                    else_value.clone(),
                )?;
                self.assign_integer(instruction, result, values)?;
                Ok(Flow::Continue)
            }
            InstructionOpcode::Trunc | InstructionOpcode::ZExt | InstructionOpcode::SExt => {
                let operands = operands(self)?;
                let [Value::Integer(value)] = operands.as_slice() else {
                    return Err(FrontendError::unsupported(format!(
                        "{opcode:?} supports only integer values"
                    )));
                };
                let width = integer_width(instruction)?;
                let result = match opcode {
                    InstructionOpcode::Trunc => self.engine.trunc(value.clone(), width)?,
                    InstructionOpcode::ZExt => self.engine.zext(value.clone(), width)?,
                    InstructionOpcode::SExt => self.engine.sext(value.clone(), width)?,
                    _ => unreachable!(),
                };
                self.assign_integer(instruction, result, values)?;
                Ok(Flow::Continue)
            }
            InstructionOpcode::Alloca => {
                let count = match instruction.get_operand(0).and_then(Operand::value) {
                    Some(value) => {
                        self.integer(value, values)?
                            .as_concrete()
                            .ok_or_else(|| FrontendError::execution("alloca count is symbolic"))?
                            .value
                    }
                    None => 1,
                };
                let type_ = instruction
                    .get_allocated_type()
                    .map_err(|_| FrontendError::execution("malformed alloca"))?;
                let len = usize::try_from(self.target_data.get_abi_size(&type_))
                    .ok()
                    .and_then(|size| size.checked_mul(count as usize))
                    .ok_or_else(|| FrontendError::execution("alloca size overflow"))?;
                self.alloca_bytes = self
                    .alloca_bytes
                    .checked_add(len)
                    .ok_or_else(|| FrontendError::execution("alloca byte counter overflow"))?;
                if self.alloca_bytes > self.request.limits.max_alloca_bytes {
                    return Err(FrontendError::execution(format!(
                        "alloca limit {} exceeded",
                        self.request.limits.max_alloca_bytes
                    )));
                }
                self.add_region(vec![SymInt::concrete(8, 0); len], true, "alloca".into());
                self.assign_pointer(
                    instruction,
                    Pointer {
                        region: self.regions.len() - 1,
                        offset: 0,
                    },
                    values,
                )?;
                Ok(Flow::Continue)
            }
            InstructionOpcode::GetElementPtr => {
                let pointer = match instruction.get_operand(0).and_then(Operand::value) {
                    Some(value) => self.pointer(value, values)?,
                    None => return Err(FrontendError::execution("GEP has no pointer operand")),
                };
                let source = instruction
                    .get_gep_source_element_type()
                    .map_err(|_| FrontendError::execution("malformed GEP"))?;
                let indices = (1..instruction.get_num_operands())
                    .map(|index| {
                        instruction
                            .get_operand(index)
                            .and_then(Operand::value)
                            .ok_or_else(|| FrontendError::execution("GEP index missing"))
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .map(|value| {
                        self.integer(value, values)?
                            .signed_concrete()
                            .ok_or_else(|| FrontendError::execution("symbolic address in GEP"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let offset = self.gep_offset(source, &indices)?;
                let offset = pointer
                    .offset
                    .checked_add_signed(offset)
                    .ok_or_else(|| FrontendError::execution("GEP address overflow"))?;
                self.check_address(
                    Pointer {
                        region: pointer.region,
                        offset,
                    },
                    0,
                )?;
                self.assign_pointer(
                    instruction,
                    Pointer {
                        region: pointer.region,
                        offset,
                    },
                    values,
                )?;
                Ok(Flow::Continue)
            }
            InstructionOpcode::Load => {
                if instruction.get_volatile().unwrap_or(false) {
                    return Err(FrontendError::unsupported("volatile load"));
                }
                let pointer = match instruction.get_operand(0).and_then(Operand::value) {
                    Some(value) => self.pointer(value, values)?,
                    None => return Err(FrontendError::execution("load has no pointer operand")),
                };
                let width = integer_width(instruction)?;
                let result = self.load_integer(pointer, width)?;
                self.assign_integer(instruction, result, values)?;
                Ok(Flow::Continue)
            }
            InstructionOpcode::Store => {
                if instruction.get_volatile().unwrap_or(false) {
                    return Err(FrontendError::unsupported("volatile store"));
                }
                let value = instruction
                    .get_operand(0)
                    .and_then(Operand::value)
                    .ok_or_else(|| FrontendError::execution("store value missing"))?;
                let pointer = instruction
                    .get_operand(1)
                    .and_then(Operand::value)
                    .ok_or_else(|| FrontendError::execution("store pointer missing"))?;
                let pointer = self.pointer(pointer, values)?;
                let value = self.integer(value, values)?;
                self.store_integer(pointer, value)?;
                Ok(Flow::Continue)
            }
            InstructionOpcode::Br => self.branch(instruction, values),
            InstructionOpcode::Switch => self.switch(instruction, values),
            InstructionOpcode::Return => {
                let value = instruction
                    .get_operand(0)
                    .and_then(Operand::value)
                    .map(|value| self.value(value, values))
                    .transpose()?;
                Ok(Flow::Return(value))
            }
            InstructionOpcode::Call => self.call(instruction, values),
            InstructionOpcode::Freeze => {
                let operand = instruction
                    .get_operand(0)
                    .and_then(Operand::value)
                    .ok_or_else(|| FrontendError::execution("freeze value missing"))?;
                match self.value(operand, values)? {
                    Value::Integer(value) => self.assign_integer(instruction, value, values)?,
                    Value::Pointer(value) => self.assign_pointer(instruction, value, values)?,
                }
                Ok(Flow::Continue)
            }
            InstructionOpcode::BitCast | InstructionOpcode::AddrSpaceCast => {
                let operand = instruction
                    .get_operand(0)
                    .and_then(Operand::value)
                    .ok_or_else(|| FrontendError::execution("cast value missing"))?;
                let pointer = self.pointer(operand, values)?;
                self.assign_pointer(instruction, pointer, values)?;
                Ok(Flow::Continue)
            }
            InstructionOpcode::PtrToInt
            | InstructionOpcode::IntToPtr
            | InstructionOpcode::PtrToAddr => Err(FrontendError::unsupported(format!(
                "{opcode:?} pointer/integer cast"
            ))),
            InstructionOpcode::FAdd
            | InstructionOpcode::FSub
            | InstructionOpcode::FMul
            | InstructionOpcode::FDiv
            | InstructionOpcode::FRem
            | InstructionOpcode::FNeg
            | InstructionOpcode::FCmp
            | InstructionOpcode::FPToSI
            | InstructionOpcode::FPToUI
            | InstructionOpcode::SIToFP
            | InstructionOpcode::UIToFP
            | InstructionOpcode::FPTrunc
            | InstructionOpcode::FPExt => {
                Err(FrontendError::unsupported("floating-point operation"))
            }
            InstructionOpcode::AtomicCmpXchg
            | InstructionOpcode::AtomicRMW
            | InstructionOpcode::Fence => Err(FrontendError::unsupported("atomic operation")),
            InstructionOpcode::ExtractElement
            | InstructionOpcode::InsertElement
            | InstructionOpcode::ShuffleVector => {
                Err(FrontendError::unsupported("vector operation"))
            }
            InstructionOpcode::Unreachable => {
                Ok(Flow::Return(Some(Value::Integer(SymInt::concrete(1, 0)))))
            }
            _ => Err(FrontendError::unsupported(format!("{opcode:?}"))),
        }
    }

    fn branch(
        &mut self,
        instruction: InstructionValue<'ctx>,
        values: &HashMap<BasicValueEnum<'ctx>, Value<Backend::Wire>>,
    ) -> Result<Flow<'ctx, Backend::Wire>, FrontendError> {
        match instruction.get_num_operands() {
            1 => Ok(Flow::Branch(
                instruction
                    .get_operand(0)
                    .and_then(Operand::block)
                    .ok_or_else(|| FrontendError::execution("branch target missing"))?,
            )),
            3 => {
                let condition = instruction
                    .get_operand(0)
                    .and_then(Operand::value)
                    .ok_or_else(|| FrontendError::execution("branch condition missing"))?;
                let condition = self
                    .integer(condition, values)?
                    .as_concrete()
                    .ok_or_else(|| FrontendError::execution("symbolic branch condition"))?
                    .value;
                // LLVM's low-level operand list stores the false successor
                // before the true successor (the textual syntax does the
                // opposite), so retain the C API's actual order here.
                let target_index = if condition & 1 == 1 { 2 } else { 1 };
                Ok(Flow::Branch(
                    instruction
                        .get_operand(target_index)
                        .and_then(Operand::block)
                        .ok_or_else(|| FrontendError::execution("branch target missing"))?,
                ))
            }
            _ => Err(FrontendError::execution("malformed branch")),
        }
    }

    fn switch(
        &mut self,
        instruction: InstructionValue<'ctx>,
        values: &HashMap<BasicValueEnum<'ctx>, Value<Backend::Wire>>,
    ) -> Result<Flow<'ctx, Backend::Wire>, FrontendError> {
        let condition = instruction
            .get_operand(0)
            .and_then(Operand::value)
            .ok_or_else(|| FrontendError::execution("switch condition missing"))?;
        let condition = self
            .integer(condition, values)?
            .as_concrete()
            .ok_or_else(|| FrontendError::execution("symbolic switch condition"))?
            .value;
        let default = instruction
            .get_operand(1)
            .and_then(Operand::block)
            .ok_or_else(|| FrontendError::execution("switch default missing"))?;
        let mut target = default;
        let mut index = 2;
        while index + 1 < instruction.get_num_operands() {
            let case = instruction
                .get_operand(index)
                .and_then(Operand::value)
                .ok_or_else(|| FrontendError::execution("switch case missing"))?;
            if self
                .integer(case, values)?
                .as_concrete()
                .is_some_and(|case| case.value == condition)
            {
                target = instruction
                    .get_operand(index + 1)
                    .and_then(Operand::block)
                    .ok_or_else(|| FrontendError::execution("switch target missing"))?;
                break;
            }
            index += 2;
        }
        Ok(Flow::Branch(target))
    }

    fn call(
        &mut self,
        instruction: InstructionValue<'ctx>,
        values: &mut HashMap<BasicValueEnum<'ctx>, Value<Backend::Wire>>,
    ) -> Result<Flow<'ctx, Backend::Wire>, FrontendError> {
        self.calls = self
            .calls
            .checked_add(1)
            .ok_or_else(|| FrontendError::execution("call counter overflow"))?;
        if self.calls > self.request.limits.max_calls {
            return Err(FrontendError::execution(format!(
                "call limit {} exceeded",
                self.request.limits.max_calls
            )));
        }
        let site = CallSiteValue::try_from(instruction)
            .map_err(|_| FrontendError::execution("malformed call"))?;
        let function = site
            .get_called_fn_value()
            .ok_or_else(|| FrontendError::unsupported("indirect call"))?;
        if function.get_type().is_var_arg() {
            return Err(FrontendError::unsupported("varargs call"));
        }
        let arguments = (0..site.count_arguments())
            .map(|index| {
                instruction
                    .get_operand(index)
                    .and_then(Operand::value)
                    .ok_or_else(|| FrontendError::execution("call argument missing"))
                    .and_then(|value| self.value(value, values))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let result = self.execute_function(function, arguments)?;
        match site.try_as_basic_value() {
            ValueKind::Instruction(_) => {
                if result.is_some() {
                    return Err(FrontendError::execution(
                        "void call adapter returned a value",
                    ));
                }
            }
            ValueKind::Basic(key) => {
                let result = result
                    .ok_or_else(|| FrontendError::execution("non-void call returned no value"))?;
                values.insert(key, result);
            }
        }
        Ok(Flow::Continue)
    }

    fn execute_declaration(
        &mut self,
        function: FunctionValue<'ctx>,
        arguments: Vec<Value<Backend::Wire>>,
    ) -> Result<Option<Value<Backend::Wire>>, FrontendError> {
        let name = cstr(function.get_name());
        if name.starts_with("llvm.lifetime.") || name.starts_with("llvm.dbg.") {
            return Ok(None);
        }
        if name.starts_with("llvm.memcpy.") || name.starts_with("llvm.memmove.") {
            if arguments.len() < 3 {
                return Err(FrontendError::execution(format!(
                    "{name} has too few arguments"
                )));
            }
            let (Value::Pointer(destination), Value::Pointer(source), Value::Integer(len)) =
                (&arguments[0], &arguments[1], &arguments[2])
            else {
                return Err(FrontendError::unsupported(format!("{name} argument types")));
            };
            let len = len
                .as_concrete()
                .ok_or_else(|| FrontendError::execution("symbolic memcpy length"))?
                .value as usize;
            let source = self.read_bytes(*source, len)?;
            self.write_bytes(*destination, source)?;
            return Ok(None);
        }
        if name.starts_with("llvm.memset.") {
            if arguments.len() < 3 {
                return Err(FrontendError::execution(format!(
                    "{name} has too few arguments"
                )));
            }
            let (Value::Pointer(destination), Value::Integer(value), Value::Integer(len)) =
                (&arguments[0], &arguments[1], &arguments[2])
            else {
                return Err(FrontendError::unsupported(format!("{name} argument types")));
            };
            let len = len
                .as_concrete()
                .ok_or_else(|| FrontendError::execution("symbolic memset length"))?
                .value as usize;
            let byte = self.engine.trunc(value.clone(), 8)?;
            self.write_bytes(*destination, vec![byte; len])?;
            return Ok(None);
        }
        let adapter = self.request.host_calls.get(function).ok_or_else(|| {
            FrontendError::unsupported(format!("declaration `{name}` has no HostCall adapter"))
        })?;
        self.validate_host_signature(function, &adapter.signature)?;
        let host_arguments = arguments
            .iter()
            .cloned()
            .map(|value| self.to_host(value))
            .collect::<Result<Vec<_>, _>>()?;
        let context = HostCallContext {
            zero: self.engine.zero.clone(),
            one: self.engine.one.clone(),
        };
        let result = adapter
            .call
            .lower(self.engine.backend, &context, &host_arguments)?;
        self.from_host_result(function, result)
    }

    fn validate_host_signature(
        &self,
        function: FunctionValue<'ctx>,
        signature: &HostCallSignature,
    ) -> Result<(), FrontendError> {
        let function_type = function.get_type();
        let parameters = function_type.get_param_types();
        if parameters.len() != signature.parameters.len() {
            return Err(FrontendError::request(format!(
                "HostCall `{}` expects {} parameters but LLVM declares {}",
                cstr(function.get_name()),
                signature.parameters.len(),
                parameters.len()
            )));
        }
        for (index, (actual, expected)) in parameters.iter().zip(&signature.parameters).enumerate()
        {
            if !matches_host_parameter_type(*actual, *expected) {
                return Err(FrontendError::request(format!(
                    "HostCall `{}` parameter {index} does not match its registered signature",
                    cstr(function.get_name())
                )));
            }
        }
        let matches_result = match (function_type.get_return_type(), signature.result) {
            (None, None) => true,
            (Some(actual), Some(expected)) => matches_host_type(actual, expected),
            _ => false,
        };
        if !matches_result {
            return Err(FrontendError::request(format!(
                "HostCall `{}` result does not match its registered signature",
                cstr(function.get_name())
            )));
        }
        Ok(())
    }

    fn to_host(
        &self,
        value: Value<Backend::Wire>,
    ) -> Result<HostValue<Backend::Wire>, FrontendError> {
        Ok(match value {
            Value::Integer(value) => HostValue::Integer(match value.value {
                SymRepr::Concrete(bits) => HostInteger::Concrete {
                    width: value.width,
                    value: bits,
                },
                SymRepr::Wires(bits) => HostInteger::Symbolic {
                    width: value.width,
                    bits,
                },
            }),
            Value::Pointer(pointer) => HostValue::Pointer {
                region: pointer.region,
                offset: pointer.offset,
            },
        })
    }

    fn from_host_result(
        &mut self,
        function: FunctionValue<'ctx>,
        result: Option<HostValue<Backend::Wire>>,
    ) -> Result<Option<Value<Backend::Wire>>, FrontendError> {
        match (function.get_type().get_return_type(), result) {
            (None, None) => Ok(None),
            (Some(BasicTypeEnum::IntType(int)), Some(HostValue::Integer(value))) => {
                let value = int_from_host(value)?;
                if value.width != int.get_bit_width() {
                    return Err(FrontendError::execution(
                        "HostCall integer result width does not match declaration",
                    ));
                }
                Ok(Some(Value::Integer(value)))
            }
            (Some(BasicTypeEnum::PointerType(_)), Some(HostValue::Pointer { region, offset })) => {
                self.check_address(Pointer { region, offset }, 0)?;
                Ok(Some(Value::Pointer(Pointer { region, offset })))
            }
            (None, Some(_)) => Err(FrontendError::execution(
                "HostCall returned a value for void declaration",
            )),
            (Some(_), None) => Err(FrontendError::execution(
                "HostCall omitted a non-void result",
            )),
            _ => Err(FrontendError::unsupported(
                "HostCall result has unsupported LLVM type",
            )),
        }
    }

    fn value(
        &mut self,
        value: BasicValueEnum<'ctx>,
        values: &HashMap<BasicValueEnum<'ctx>, Value<Backend::Wire>>,
    ) -> Result<Value<Backend::Wire>, FrontendError> {
        if let Some(value) = values.get(&value) {
            return Ok(value.clone());
        }
        if let BasicValueEnum::IntValue(value) = value {
            self.engine.check_width(value.get_type().get_bit_width())?;
            if value.is_undef() || value.is_poison() {
                return Ok(Value::Integer(SymInt::concrete(
                    value.get_type().get_bit_width(),
                    0,
                )));
            }
            if let Some(constant) = value.get_zero_extended_constant() {
                return Ok(Value::Integer(SymInt::concrete(
                    value.get_type().get_bit_width(),
                    constant,
                )));
            }
        }
        if let BasicValueEnum::PointerValue(pointer) = value {
            let name = cstr(pointer.get_name());
            if let Some(&region) = self.globals.get(&name) {
                return Ok(Value::Pointer(Pointer { region, offset: 0 }));
            }
            if pointer.is_undef() || pointer.is_poison() {
                return Err(FrontendError::execution(
                    "undefined pointer value was reached",
                ));
            }
        }
        Err(FrontendError::unsupported(format!(
            "unbound value `{}`",
            cstr(value.get_name())
        )))
    }

    fn integer(
        &mut self,
        value: BasicValueEnum<'ctx>,
        values: &HashMap<BasicValueEnum<'ctx>, Value<Backend::Wire>>,
    ) -> Result<SymInt<Backend::Wire>, FrontendError> {
        match self.value(value, values)? {
            Value::Integer(value) => Ok(value),
            Value::Pointer(_) => Err(FrontendError::unsupported("pointer used as integer")),
        }
    }

    fn pointer(
        &mut self,
        value: BasicValueEnum<'ctx>,
        values: &HashMap<BasicValueEnum<'ctx>, Value<Backend::Wire>>,
    ) -> Result<Pointer, FrontendError> {
        match self.value(value, values)? {
            Value::Pointer(value) => Ok(value),
            Value::Integer(_) => Err(FrontendError::unsupported("integer used as pointer")),
        }
    }

    fn assign_integer(
        &mut self,
        instruction: InstructionValue<'ctx>,
        value: SymInt<Backend::Wire>,
        values: &mut HashMap<BasicValueEnum<'ctx>, Value<Backend::Wire>>,
    ) -> Result<(), FrontendError> {
        let width = integer_width(instruction)?;
        if value.width != width {
            return Err(FrontendError::execution(
                "integer instruction result width mismatch",
            ));
        }
        values.insert(basic_instruction(instruction)?, Value::Integer(value));
        Ok(())
    }

    fn assign_pointer(
        &mut self,
        instruction: InstructionValue<'ctx>,
        value: Pointer,
        values: &mut HashMap<BasicValueEnum<'ctx>, Value<Backend::Wire>>,
    ) -> Result<(), FrontendError> {
        values.insert(basic_instruction(instruction)?, Value::Pointer(value));
        Ok(())
    }

    fn gep_offset(
        &mut self,
        source: BasicTypeEnum<'ctx>,
        indices: &[i64],
    ) -> Result<isize, FrontendError> {
        if indices.is_empty() {
            return Ok(0);
        }
        let mut offset =
            i128::from(indices[0]) * i128::from(self.target_data.get_abi_size(&source));
        let mut current = source;
        for &index in &indices[1..] {
            if current.is_array_type() {
                let array = current.into_array_type();
                let element = array.get_element_type();
                offset += i128::from(index) * i128::from(self.target_data.get_abi_size(&element));
                current = element;
            } else if current.is_struct_type() {
                if index < 0 {
                    return Err(FrontendError::execution("negative struct GEP field"));
                }
                let structure = current.into_struct_type();
                let field = u32::try_from(index)
                    .map_err(|_| FrontendError::execution("struct GEP field overflow"))?;
                offset += i128::from(
                    self.target_data
                        .offset_of_element(&structure, field)
                        .ok_or_else(|| FrontendError::execution("struct GEP field out of range"))?,
                );
                current = *structure
                    .get_field_types()
                    .get(field as usize)
                    .ok_or_else(|| FrontendError::execution("struct GEP field type missing"))?;
            } else {
                return Err(FrontendError::unsupported(
                    "GEP through a non-aggregate type",
                ));
            }
        }
        isize::try_from(offset).map_err(|_| FrontendError::execution("GEP offset overflow"))
    }

    fn check_address(&self, pointer: Pointer, len: usize) -> Result<(), FrontendError> {
        let region = self
            .regions
            .get(pointer.region)
            .ok_or_else(|| FrontendError::execution("pointer references no region"))?;
        if pointer
            .offset
            .checked_add(len)
            .is_none_or(|end| end > region.bytes.len())
        {
            return Err(FrontendError::execution(format!(
                "out-of-bounds access to {}",
                region.label
            )));
        }
        Ok(())
    }

    fn read_bytes(
        &self,
        pointer: Pointer,
        len: usize,
    ) -> Result<Vec<SymInt<Backend::Wire>>, FrontendError> {
        self.check_address(pointer, len)?;
        Ok(self.regions[pointer.region].bytes[pointer.offset..pointer.offset + len].to_vec())
    }

    fn write_bytes(
        &mut self,
        pointer: Pointer,
        bytes: Vec<SymInt<Backend::Wire>>,
    ) -> Result<(), FrontendError> {
        self.check_address(pointer, bytes.len())?;
        let region = &mut self.regions[pointer.region];
        if !region.writable {
            return Err(FrontendError::execution(format!(
                "store to immutable {}",
                region.label
            )));
        }
        region.bytes[pointer.offset..pointer.offset + bytes.len()].clone_from_slice(&bytes);
        Ok(())
    }

    fn load_integer(
        &mut self,
        pointer: Pointer,
        width: u32,
    ) -> Result<SymInt<Backend::Wire>, FrontendError> {
        self.engine.check_width(width)?;
        let bytes = self.read_bytes(pointer, (width / 8).max(1) as usize)?;
        let mut output = SymInt::concrete(width, 0);
        for (index, byte) in bytes.into_iter().enumerate() {
            let widened = if width < 8 {
                self.engine.trunc(byte, width)?
            } else {
                self.engine.zext(byte, width)?
            };
            let shift = SymInt::concrete(width, (index * 8) as u64);
            let shifted = self.engine.shift(widened, shift, ShiftOp::Left)?;
            output = self.engine.bitwise(output, shifted, BitOp::Or)?;
        }
        Ok(output)
    }

    fn store_integer(
        &mut self,
        pointer: Pointer,
        value: SymInt<Backend::Wire>,
    ) -> Result<(), FrontendError> {
        self.engine.check_width(value.width)?;
        let len = (value.width / 8).max(1) as usize;
        let mut bytes = Vec::with_capacity(len);
        for index in 0..len {
            let shifted = self.engine.shift(
                value.clone(),
                SymInt::concrete(value.width, (index * 8) as u64),
                ShiftOp::LogicalRight,
            )?;
            bytes.push(if value.width < 8 {
                self.engine.zext(shifted, 8)?
            } else {
                self.engine.trunc(shifted, 8)?
            });
        }
        self.write_bytes(pointer, bytes)
    }

    fn constant_bytes(
        &mut self,
        value: BasicValueEnum<'ctx>,
    ) -> Result<Vec<SymInt<Backend::Wire>>, FrontendError> {
        let type_ = value.get_type();
        let len = usize::try_from(self.target_data.get_abi_size(&type_))
            .map_err(|_| FrontendError::unsupported("global initializer too large"))?;
        let mut bytes = vec![SymInt::concrete(8, 0); len];
        self.write_constant_bytes(value, 0, &mut bytes)?;
        Ok(bytes)
    }

    fn write_constant_bytes(
        &mut self,
        value: BasicValueEnum<'ctx>,
        offset: usize,
        bytes: &mut [SymInt<Backend::Wire>],
    ) -> Result<(), FrontendError> {
        match value {
            BasicValueEnum::IntValue(value) => {
                let width = value.get_type().get_bit_width();
                self.engine.check_width(width)?;
                let integer = if value.is_undef() || value.is_poison() {
                    0
                } else {
                    value.get_zero_extended_constant().ok_or_else(|| {
                        FrontendError::unsupported("non-integer global initializer")
                    })?
                };
                let len = (width / 8).max(1) as usize;
                for index in 0..len {
                    bytes[offset + index] = SymInt::concrete(8, integer >> (index * 8));
                }
            }
            BasicValueEnum::ArrayValue(array) => {
                let type_ = array.get_type();
                let element = type_.get_element_type();
                let stride = self.target_data.get_abi_size(&element) as usize;
                if let Some(string) = array.as_const_string() {
                    if !element.is_int_type() || element.into_int_type().get_bit_width() != 8 {
                        return Err(FrontendError::unsupported(
                            "non-byte LLVM string initializer",
                        ));
                    }
                    let source = string;
                    if source.len() > type_.len() as usize {
                        return Err(FrontendError::unsupported(
                            "LLVM string initializer is larger than its array",
                        ));
                    }
                    for (index, byte) in source.iter().copied().enumerate() {
                        bytes[offset + index] = SymInt::concrete(8, u64::from(byte));
                    }
                    return Ok(());
                }
                for index in 0..type_.len() {
                    // LLVM represents a constant aggregate's elements as
                    // operands, even though inkwell exposes no high-level
                    // iterator for ArrayValue.
                    let raw = unsafe {
                        inkwell::llvm_sys::core::LLVMGetOperand(array.as_value_ref(), index)
                    };
                    if raw.is_null() {
                        return Err(FrontendError::unsupported(
                            "array initializer element is not available as an LLVM operand",
                        ));
                    }
                    let element_value = unsafe { BasicValueEnum::new(raw) };
                    self.write_constant_bytes(
                        element_value,
                        offset + index as usize * stride,
                        bytes,
                    )?;
                }
            }
            BasicValueEnum::StructValue(structure) => {
                let type_ = structure.get_type();
                for (index, _) in type_.get_field_types().iter().enumerate() {
                    let field_offset = self
                        .target_data
                        .offset_of_element(&type_, index as u32)
                        .ok_or_else(|| {
                            FrontendError::unsupported("struct initializer field offset missing")
                        })? as usize;
                    self.write_constant_bytes(
                        structure.get_field_at_index(index as u32).ok_or_else(|| {
                            FrontendError::unsupported("struct initializer field missing")
                        })?,
                        offset + field_offset,
                        bytes,
                    )?;
                }
            }
            _ => {
                return Err(FrontendError::unsupported(
                    "aggregate global initializer type",
                ));
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<ExecutionResult<Backend::Wire>, FrontendError> {
        let return_value = self.returned.take();
        let mut outputs = Vec::new();
        for export in self.request.exports {
            match export {
                Export::Return => {
                    let value = return_value.clone().ok_or_else(|| {
                        FrontendError::request("return export requested from a void entry function")
                    })?;
                    let Value::Integer(value) = value else {
                        return Err(FrontendError::unsupported("pointer return export"));
                    };
                    outputs.extend(self.engine.materialize(value)?);
                }
                Export::ArgumentMemory {
                    argument,
                    offset,
                    len,
                } => {
                    let region = self
                        .argument_regions
                        .get(*argument)
                        .and_then(|region| *region)
                        .ok_or_else(|| {
                            FrontendError::request(format!(
                                "argument {argument} is not a pointer region"
                            ))
                        })?;
                    for byte in self.read_bytes(
                        Pointer {
                            region,
                            offset: *offset,
                        },
                        *len,
                    )? {
                        outputs.extend(self.engine.materialize(byte)?);
                    }
                }
                Export::GlobalMemory { name, offset, len } => {
                    let region = *self.globals.get(name).ok_or_else(|| {
                        FrontendError::request(format!("global @{name} is not bound or immutable"))
                    })?;
                    for byte in self.read_bytes(
                        Pointer {
                            region,
                            offset: *offset,
                        },
                        *len,
                    )? {
                        outputs.extend(self.engine.materialize(byte)?);
                    }
                }
            }
        }
        self.engine.finish(outputs)
    }
}

enum Flow<'ctx, W = Idx> {
    Continue,
    Branch(BasicBlock<'ctx>),
    Return(Option<Value<W>>),
}

struct Engine<'backend, Backend: ExecutionBackend> {
    backend: &'backend mut Backend,
    inputs: Vec<Backend::Wire>,
    zero: Backend::Wire,
    one: Backend::Wire,
    error: Option<FrontendError>,
}

#[derive(Clone, Copy)]
enum BitOp {
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy)]
enum ShiftOp {
    Left,
    LogicalRight,
    ArithmeticRight,
}

impl<'backend, Backend: ExecutionBackend> Engine<'backend, Backend> {
    fn new(backend: &'backend mut Backend) -> Result<Self, FrontendError> {
        let zero = backend
            .create(false)
            .map_err(|error| FrontendError::backend(Backend::describe_error(error)))?;
        let one = backend
            .create(true)
            .map_err(|error| FrontendError::backend(Backend::describe_error(error)))?;
        Ok(Self::from_constants(backend, zero, one))
    }

    fn from_constants(
        backend: &'backend mut Backend,
        zero: Backend::Wire,
        one: Backend::Wire,
    ) -> Self {
        Self {
            backend,
            inputs: Vec::new(),
            zero,
            one,
            error: None,
        }
    }

    fn finish(
        mut self,
        outputs: Vec<Backend::Wire>,
    ) -> Result<ExecutionResult<Backend::Wire>, FrontendError> {
        self.check_error()?;
        Ok(ExecutionResult {
            inputs: self.inputs,
            outputs,
        })
    }

    fn check_error(&mut self) -> Result<(), FrontendError> {
        match self.error.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn record_backend_error(&mut self, error: Backend::Error) {
        if self.error.is_none() {
            self.error = Some(FrontendError::backend(Backend::describe_error(error)));
        }
    }

    fn check_width(&self, width: u32) -> Result<(), FrontendError> {
        check_width(width)
    }

    fn symbolic(&mut self, width: u32) -> SymInt<Backend::Wire> {
        let mut wires = Vec::with_capacity(width as usize);
        for _ in 0..width {
            match self.backend.create(false) {
                Ok(input) => {
                    self.inputs.push(input.clone());
                    wires.push(input);
                }
                Err(error) => {
                    self.record_backend_error(error);
                    wires.push(self.zero.clone());
                }
            }
        }
        SymInt {
            width,
            value: SymRepr::Wires(wires),
        }
    }

    fn host_int(&self, value: SymInt<Backend::Wire>) -> HostInteger<Backend::Wire> {
        match value.value {
            SymRepr::Concrete(bits) => HostInteger::Concrete {
                width: value.width,
                value: bits,
            },
            SymRepr::Wires(bits) => HostInteger::Symbolic {
                width: value.width,
                bits,
            },
        }
    }

    fn materialize(
        &mut self,
        value: SymInt<Backend::Wire>,
    ) -> Result<Vec<Backend::Wire>, FrontendError> {
        self.check_width(value.width)?;
        Ok(match value.value {
            SymRepr::Concrete(bits) => (0..value.width)
                .map(|bit| {
                    if (bits >> bit) & 1 == 0 {
                        self.zero.clone()
                    } else {
                        self.one.clone()
                    }
                })
                .collect(),
            SymRepr::Wires(bits) => bits,
        })
    }

    fn bit(&self, value: &SymInt<Backend::Wire>, index: usize) -> IdxOrConst<Backend::Wire> {
        match &value.value {
            SymRepr::Concrete(value) => IdxOrConst::Constant((value >> index) & 1 == 1),
            SymRepr::Wires(bits) => IdxOrConst::Wire(bits[index].clone()),
        }
    }

    fn from_bits(&self, width: u32, bits: Vec<IdxOrConst<Backend::Wire>>) -> SymInt<Backend::Wire> {
        if bits
            .iter()
            .all(|bit| matches!(bit, IdxOrConst::Constant(_)))
        {
            let value = bits.iter().enumerate().fold(0u64, |value, (index, bit)| {
                value | (u64::from(matches!(bit, IdxOrConst::Constant(true))) << index)
            });
            SymInt::concrete(width, value)
        } else {
            SymInt {
                width,
                value: SymRepr::Wires(
                    bits.into_iter()
                        .map(|bit| match bit {
                            IdxOrConst::Wire(wire) => wire,
                            IdxOrConst::Constant(false) => self.zero.clone(),
                            IdxOrConst::Constant(true) => self.one.clone(),
                        })
                        .collect(),
                ),
            }
        }
    }

    fn gate(
        &mut self,
        op: BitOp,
        left: IdxOrConst<Backend::Wire>,
        right: IdxOrConst<Backend::Wire>,
    ) -> IdxOrConst<Backend::Wire> {
        use IdxOrConst::{Constant as C, Wire as W};
        match (op, left, right) {
            (_, C(left), C(right)) => C(match op {
                BitOp::And => left & right,
                BitOp::Or => left | right,
                BitOp::Xor => left ^ right,
            }),
            (BitOp::And, C(false), _) | (BitOp::And, _, C(false)) => C(false),
            (BitOp::And, C(true), right) | (BitOp::And, right, C(true)) => right,
            (BitOp::Or, C(true), _) | (BitOp::Or, _, C(true)) => C(true),
            (BitOp::Or, C(false), right) | (BitOp::Or, right, C(false)) => right,
            (BitOp::Xor, C(false), right) | (BitOp::Xor, right, C(false)) => right,
            (BitOp::Xor, C(true), W(right)) | (BitOp::Xor, W(right), C(true)) => {
                W(self.backend_gate(BitOp::Xor, right, self.one.clone()))
            }
            (BitOp::And, W(left), W(right)) => W(self.backend_gate(BitOp::And, left, right)),
            (BitOp::Or, W(left), W(right)) => W(self.backend_gate(BitOp::Or, left, right)),
            (BitOp::Xor, W(left), W(right)) => W(self.backend_gate(BitOp::Xor, left, right)),
        }
    }

    fn backend_gate(
        &mut self,
        operation: BitOp,
        left: Backend::Wire,
        right: Backend::Wire,
    ) -> Backend::Wire {
        let result = match operation {
            BitOp::And => self.backend.bitand(left, right),
            BitOp::Or => self.backend.bitor(left, right),
            BitOp::Xor => self.backend.bitxor(left, right),
        };
        match result {
            Ok(value) => value,
            Err(error) => {
                self.record_backend_error(error);
                self.zero.clone()
            }
        }
    }

    fn mux_bit(
        &mut self,
        condition: IdxOrConst<Backend::Wire>,
        then_value: IdxOrConst<Backend::Wire>,
        else_value: IdxOrConst<Backend::Wire>,
    ) -> IdxOrConst<Backend::Wire> {
        match (condition, then_value, else_value) {
            (IdxOrConst::Constant(true), then_value, _) => then_value,
            (IdxOrConst::Constant(false), _, else_value) => else_value,
            (_, IdxOrConst::Constant(left), IdxOrConst::Constant(right)) if left == right => {
                IdxOrConst::Constant(left)
            }
            (IdxOrConst::Wire(condition), then_value, else_value) => {
                let then_value = match then_value {
                    IdxOrConst::Wire(value) => value,
                    IdxOrConst::Constant(false) => self.zero.clone(),
                    IdxOrConst::Constant(true) => self.one.clone(),
                };
                let else_value = match else_value {
                    IdxOrConst::Wire(value) => value,
                    IdxOrConst::Constant(false) => self.zero.clone(),
                    IdxOrConst::Constant(true) => self.one.clone(),
                };
                let result = self.backend.mux(condition, then_value, else_value);
                match result {
                    Ok(value) => IdxOrConst::Wire(value),
                    Err(error) => {
                        self.record_backend_error(error);
                        IdxOrConst::Wire(self.zero.clone())
                    }
                }
            }
        }
    }

    fn bitwise(
        &mut self,
        left: SymInt<Backend::Wire>,
        right: SymInt<Backend::Wire>,
        op: BitOp,
    ) -> Result<SymInt<Backend::Wire>, FrontendError> {
        self.same_width(&left, &right)?;
        if let (Some(left), Some(right)) = (left.as_concrete(), right.as_concrete()) {
            return Ok(SymInt::concrete(
                left.width,
                match op {
                    BitOp::And => left.value & right.value,
                    BitOp::Or => left.value | right.value,
                    BitOp::Xor => left.value ^ right.value,
                },
            ));
        }
        let bits = (0..left.width as usize)
            .map(|index| self.gate(op, self.bit(&left, index), self.bit(&right, index)))
            .collect();
        Ok(self.from_bits(left.width, bits))
    }

    fn not(&mut self, value: SymInt<Backend::Wire>) -> SymInt<Backend::Wire> {
        let width = value.width;
        self.bitwise(value, SymInt::concrete(width, mask(width)), BitOp::Xor)
            .expect("same width")
    }

    fn add(
        &mut self,
        left: SymInt<Backend::Wire>,
        right: SymInt<Backend::Wire>,
    ) -> SymInt<Backend::Wire> {
        self.same_width(&left, &right)
            .expect("LLVM operation widths match");
        if let (Some(left), Some(right)) = (left.as_concrete(), right.as_concrete()) {
            return SymInt::concrete(left.width, left.value.wrapping_add(right.value));
        }
        let output = match left.width {
            1 => self.add_fixed::<1>(&left, &right).to_vec(),
            8 => self.add_fixed::<8>(&left, &right).to_vec(),
            16 => self.add_fixed::<16>(&left, &right).to_vec(),
            32 => self.add_fixed::<32>(&left, &right).to_vec(),
            64 => self.add_fixed::<64>(&left, &right).to_vec(),
            _ => unreachable!("integer width was validated before lowering"),
        };
        self.from_bits(left.width, output)
    }

    fn add_fixed<const N: usize>(
        &mut self,
        left: &SymInt<Backend::Wire>,
        right: &SymInt<Backend::Wire>,
    ) -> [IdxOrConst<Backend::Wire>; N] {
        let left_bits = std::array::from_fn(|index| self.bit(left, index));
        let right_bits = std::array::from_fn(|index| self.bit(right, index));
        add_bits_with(
            &left_bits,
            &right_bits,
            IdxOrConst::Constant(false),
            |operation, left, right| {
                let operation = match operation {
                    WordBitOp::And => BitOp::And,
                    WordBitOp::Or => BitOp::Or,
                    WordBitOp::Xor => BitOp::Xor,
                };
                Ok::<IdxOrConst<Backend::Wire>, Infallible>(self.gate(operation, left, right))
            },
        )
        .expect("LLVM circuit recorder gates are infallible")
    }

    fn sub(
        &mut self,
        left: SymInt<Backend::Wire>,
        right: SymInt<Backend::Wire>,
    ) -> SymInt<Backend::Wire> {
        let width = left.width;
        let inverted = self.not(right);
        let adjusted = self.add(inverted, SymInt::concrete(width, 1));
        self.add(left, adjusted)
    }

    fn mul(
        &mut self,
        left: SymInt<Backend::Wire>,
        right: SymInt<Backend::Wire>,
    ) -> Result<SymInt<Backend::Wire>, FrontendError> {
        self.same_width(&left, &right)?;
        if let (Some(left), Some(right)) = (left.as_concrete(), right.as_concrete()) {
            return Ok(SymInt::concrete(
                left.width,
                left.value.wrapping_mul(right.value),
            ));
        }
        let mut output = SymInt::concrete(left.width, 0);
        for index in 0..left.width as usize {
            let bit = self.bit(&right, index);
            let mut product_bits = Vec::with_capacity(left.width as usize);
            for bit_index in 0..left.width as usize {
                let product_bit = if bit_index < index {
                    IdxOrConst::Constant(false)
                } else {
                    self.gate(BitOp::And, self.bit(&left, bit_index - index), bit.clone())
                };
                product_bits.push(product_bit);
            }
            let product = self.from_bits(left.width, product_bits);
            output = self.add(output, product);
        }
        Ok(output)
    }

    fn compare(
        &mut self,
        left: SymInt<Backend::Wire>,
        right: SymInt<Backend::Wire>,
        predicate: IntPredicate,
    ) -> Result<SymInt<Backend::Wire>, FrontendError> {
        self.same_width(&left, &right)?;
        if let (Some(left), Some(right)) = (left.as_concrete(), right.as_concrete()) {
            let signed_left = signed(left.value, left.width);
            let signed_right = signed(right.value, right.width);
            let value = match predicate {
                IntPredicate::EQ => left.value == right.value,
                IntPredicate::NE => left.value != right.value,
                IntPredicate::UGT => left.value > right.value,
                IntPredicate::UGE => left.value >= right.value,
                IntPredicate::ULT => left.value < right.value,
                IntPredicate::ULE => left.value <= right.value,
                IntPredicate::SGT => signed_left > signed_right,
                IntPredicate::SGE => signed_left >= signed_right,
                IntPredicate::SLT => signed_left < signed_right,
                IntPredicate::SLE => signed_left <= signed_right,
            };
            return Ok(SymInt::concrete(1, u64::from(value)));
        }
        let equal = self.equal(left.clone(), right.clone())?;
        let unsigned_lt = self.unsigned_lt(left.clone(), right.clone())?;
        let unsigned_gt = self.unsigned_lt(right.clone(), left.clone())?;
        let sign_left = self.bit(&left, left.width as usize - 1);
        let sign_right = self.bit(&right, right.width as usize - 1);
        let signs_differ = self.gate(BitOp::Xor, sign_left.clone(), sign_right.clone());
        let unsigned_lt_bit = self.bit(&unsigned_lt, 0);
        let unsigned_gt_bit = self.bit(&unsigned_gt, 0);
        let signed_lt = self.mux_bit(signs_differ.clone(), sign_left, unsigned_lt_bit.clone());
        let signed_gt = self.mux_bit(signs_differ, sign_right, unsigned_gt_bit.clone());
        let equal_bit = self.bit(&equal, 0);
        let bit = match predicate {
            IntPredicate::EQ => equal_bit,
            IntPredicate::NE => {
                self.gate(BitOp::Xor, equal_bit.clone(), IdxOrConst::Constant(true))
            }
            IntPredicate::ULT => unsigned_lt_bit,
            IntPredicate::UGT => unsigned_gt_bit,
            IntPredicate::ULE => self.gate(BitOp::Or, unsigned_lt_bit, equal_bit.clone()),
            IntPredicate::UGE => self.gate(BitOp::Or, unsigned_gt_bit, equal_bit.clone()),
            IntPredicate::SLT => signed_lt,
            IntPredicate::SGT => signed_gt,
            IntPredicate::SLE => self.gate(BitOp::Or, signed_lt, equal_bit.clone()),
            IntPredicate::SGE => self.gate(BitOp::Or, signed_gt, equal_bit),
        };
        Ok(self.from_bits(1, vec![bit]))
    }

    fn equal(
        &mut self,
        left: SymInt<Backend::Wire>,
        right: SymInt<Backend::Wire>,
    ) -> Result<SymInt<Backend::Wire>, FrontendError> {
        self.same_width(&left, &right)?;
        let mut result = IdxOrConst::Constant(true);
        for index in 0..left.width as usize {
            let unequal = self.gate(BitOp::Xor, self.bit(&left, index), self.bit(&right, index));
            let matches = self.gate(BitOp::Xor, unequal, IdxOrConst::Constant(true));
            result = self.gate(BitOp::And, result, matches);
        }
        Ok(self.from_bits(1, vec![result]))
    }

    fn unsigned_lt(
        &mut self,
        left: SymInt<Backend::Wire>,
        right: SymInt<Backend::Wire>,
    ) -> Result<SymInt<Backend::Wire>, FrontendError> {
        self.same_width(&left, &right)?;
        let mut less = IdxOrConst::Constant(false);
        let mut equal = IdxOrConst::Constant(true);
        for index in (0..left.width as usize).rev() {
            let left_bit = self.bit(&left, index);
            let right_bit = self.bit(&right, index);
            let not_left = self.gate(BitOp::Xor, left_bit.clone(), IdxOrConst::Constant(true));
            let this_less = self.gate(BitOp::And, not_left, right_bit.clone());
            let less_here = self.gate(BitOp::And, equal.clone(), this_less);
            less = self.gate(BitOp::Or, less, less_here);
            let unequal = self.gate(BitOp::Xor, left_bit, right_bit);
            let same = self.gate(BitOp::Xor, unequal, IdxOrConst::Constant(true));
            equal = self.gate(BitOp::And, equal, same);
        }
        Ok(self.from_bits(1, vec![less]))
    }

    fn select(
        &mut self,
        condition: SymInt<Backend::Wire>,
        then_value: SymInt<Backend::Wire>,
        else_value: SymInt<Backend::Wire>,
    ) -> Result<SymInt<Backend::Wire>, FrontendError> {
        if condition.width != 1 {
            return Err(FrontendError::unsupported("select condition is not i1"));
        }
        self.same_width(&then_value, &else_value)?;
        if let Some(condition) = condition.as_concrete() {
            return Ok(if condition.value & 1 == 1 {
                then_value
            } else {
                else_value
            });
        }
        let condition = self.bit(&condition, 0);
        let mut bits = Vec::with_capacity(then_value.width as usize);
        for index in 0..then_value.width as usize {
            bits.push(self.mux_bit(
                condition.clone(),
                self.bit(&then_value, index),
                self.bit(&else_value, index),
            ));
        }
        Ok(self.from_bits(then_value.width, bits))
    }

    fn trunc(
        &mut self,
        value: SymInt<Backend::Wire>,
        width: u32,
    ) -> Result<SymInt<Backend::Wire>, FrontendError> {
        self.check_width(width)?;
        if width > value.width {
            return Err(FrontendError::unsupported("integer truncation widens"));
        }
        match value.value {
            SymRepr::Concrete(value) => Ok(SymInt::concrete(width, value)),
            SymRepr::Wires(bits) => Ok(SymInt {
                width,
                value: SymRepr::Wires(bits[..width as usize].to_vec()),
            }),
        }
    }

    fn zext(
        &mut self,
        value: SymInt<Backend::Wire>,
        width: u32,
    ) -> Result<SymInt<Backend::Wire>, FrontendError> {
        self.check_width(width)?;
        if width < value.width {
            return Err(FrontendError::unsupported("zero extension narrows"));
        }
        if let Some(value) = value.as_concrete() {
            return Ok(SymInt::concrete(width, value.value));
        }
        Ok(self.from_bits(
            width,
            (0..width as usize)
                .map(|index| {
                    if index < value.width as usize {
                        self.bit(&value, index)
                    } else {
                        IdxOrConst::Constant(false)
                    }
                })
                .collect(),
        ))
    }

    fn sext(
        &mut self,
        value: SymInt<Backend::Wire>,
        width: u32,
    ) -> Result<SymInt<Backend::Wire>, FrontendError> {
        self.check_width(width)?;
        if width < value.width {
            return Err(FrontendError::unsupported("sign extension narrows"));
        }
        if let Some(value) = value.as_concrete() {
            return Ok(SymInt::concrete(
                width,
                signed(value.value, value.width) as u64,
            ));
        }
        let sign = self.bit(&value, value.width as usize - 1);
        Ok(self.from_bits(
            width,
            (0..width as usize)
                .map(|index| {
                    if index < value.width as usize {
                        self.bit(&value, index)
                    } else {
                        sign.clone()
                    }
                })
                .collect(),
        ))
    }

    fn shift(
        &mut self,
        value: SymInt<Backend::Wire>,
        amount: SymInt<Backend::Wire>,
        op: ShiftOp,
    ) -> Result<SymInt<Backend::Wire>, FrontendError> {
        self.check_width(value.width)?;
        if let Some(amount) = amount.as_concrete() {
            return Ok(self.shift_constant(value, amount.value as usize, op));
        }
        let mut output = value.clone();
        let stages = (value.width as usize).ilog2() as usize + 1;
        for stage in 0..stages {
            let distance = 1usize << stage;
            if distance >= value.width as usize {
                break;
            }
            let candidate = self.shift_constant(output.clone(), distance, op);
            let bit = if stage < amount.width as usize {
                self.from_bits(1, vec![self.bit(&amount, stage)])
            } else {
                SymInt::concrete(1, 0)
            };
            output = self.select(bit, candidate, output)?;
        }
        let in_range = self.compare(
            amount,
            SymInt::concrete(value.width, value.width as u64),
            IntPredicate::ULT,
        )?;
        // LLVM makes every shift whose count is at least the operand width
        // poison, including arithmetic right shift.  The selected Cirrus
        // policy canonicalizes such reached poison to zero.
        self.select(in_range, output, SymInt::concrete(value.width, 0))
    }

    fn shift_constant(
        &mut self,
        value: SymInt<Backend::Wire>,
        amount: usize,
        op: ShiftOp,
    ) -> SymInt<Backend::Wire> {
        if amount >= value.width as usize {
            // The LLVM operation is poison, regardless of shift direction.
            return SymInt::concrete(value.width, 0);
        }
        if let Some(value) = value.as_concrete() {
            let result = match op {
                ShiftOp::Left => value.value << amount,
                ShiftOp::LogicalRight => value.value >> amount,
                ShiftOp::ArithmeticRight => (signed(value.value, value.width) >> amount) as u64,
            };
            return SymInt::concrete(value.width, result);
        }
        self.from_bits(
            value.width,
            (0..value.width as usize)
                .map(|index| match op {
                    ShiftOp::Left => {
                        if index < amount {
                            IdxOrConst::Constant(false)
                        } else {
                            self.bit(&value, index - amount)
                        }
                    }
                    ShiftOp::LogicalRight => {
                        if index + amount >= value.width as usize {
                            IdxOrConst::Constant(false)
                        } else {
                            self.bit(&value, index + amount)
                        }
                    }
                    ShiftOp::ArithmeticRight => {
                        if index + amount >= value.width as usize {
                            self.bit(&value, value.width as usize - 1)
                        } else {
                            self.bit(&value, index + amount)
                        }
                    }
                })
                .collect(),
        )
    }

    fn divrem(
        &mut self,
        left: SymInt<Backend::Wire>,
        right: SymInt<Backend::Wire>,
        signed_division: bool,
        _remainder: bool,
    ) -> Result<(SymInt<Backend::Wire>, SymInt<Backend::Wire>), FrontendError> {
        self.same_width(&left, &right)?;
        if let (Some(left), Some(right)) = (left.as_concrete(), right.as_concrete()) {
            if right.value == 0 {
                return Ok((
                    SymInt::concrete(left.width, 0),
                    SymInt::concrete(left.width, 0),
                ));
            }
            let (quotient, remainder) = if signed_division {
                let width = left.width;
                let left = signed(left.value, left.width);
                let right = signed(right.value, right.width);
                // LLVM defines this signed overflow as poison.  Cirrus's
                // selected deterministic poison policy represents it as zero
                // instead of relying on Rust's debug-overflow behavior.
                let minimum = if width == 64 {
                    i64::MIN
                } else {
                    -(1i64 << (width - 1))
                };
                if left == minimum && right == -1 {
                    (0, 0)
                } else {
                    ((left / right) as u64, (left % right) as u64)
                }
            } else {
                (left.value / right.value, left.value % right.value)
            };
            return Ok((
                SymInt::concrete(left.width, quotient),
                SymInt::concrete(left.width, remainder),
            ));
        }
        let width = left.width;
        let original_left = left.clone();
        let original_right = right.clone();
        let sign_left = self.from_bits(1, vec![self.bit(&left, width as usize - 1)]);
        let sign_right = self.from_bits(1, vec![self.bit(&right, width as usize - 1)]);
        let left_abs = if signed_division {
            let negated = self.sub(SymInt::concrete(width, 0), left.clone());
            self.select(sign_left.clone(), negated, left.clone())?
        } else {
            left
        };
        let right_abs = if signed_division {
            let negated = self.sub(SymInt::concrete(width, 0), right.clone());
            self.select(sign_right.clone(), negated, right.clone())?
        } else {
            right
        };
        let mut quotient_bits = vec![IdxOrConst::Constant(false); width as usize];
        let mut remainder = SymInt::concrete(width, 0);
        for index in (0..width as usize).rev() {
            remainder = self.shift_constant(remainder, 1, ShiftOp::Left);
            let mut bits: Vec<IdxOrConst<Backend::Wire>> = match remainder.value {
                SymRepr::Concrete(value) => (0..width)
                    .map(|bit| IdxOrConst::Constant((value >> bit) & 1 == 1))
                    .collect(),
                SymRepr::Wires(bits) => bits.into_iter().map(IdxOrConst::Wire).collect(),
            };
            bits[0] = self.bit(&left_abs, index);
            remainder = self.from_bits(width, bits);
            let ge = self.compare(remainder.clone(), right_abs.clone(), IntPredicate::UGE)?;
            let subtracted = self.sub(remainder.clone(), right_abs.clone());
            remainder = self.select(ge.clone(), subtracted, remainder)?;
            quotient_bits[index] = self.bit(&ge, 0);
        }
        let quotient = self.from_bits(width, quotient_bits);
        let (quotient, remainder) = if signed_division {
            let signs_differ = self.bitwise(sign_left.clone(), sign_right, BitOp::Xor)?;
            let negated_quotient = self.sub(SymInt::concrete(width, 0), quotient.clone());
            let quotient = self.select(signs_differ, negated_quotient, quotient)?;
            let negated_remainder = self.sub(SymInt::concrete(width, 0), remainder.clone());
            let remainder = self.select(sign_left, negated_remainder, remainder)?;
            (quotient, remainder)
        } else {
            (quotient, remainder)
        };
        let zero = SymInt::concrete(width, 0);
        let divide_by_zero =
            self.compare(original_right.clone(), zero.clone(), IntPredicate::EQ)?;
        let invalid = if signed_division {
            let minimum = SymInt::concrete(width, 1u64 << (width - 1));
            let minus_one = SymInt::concrete(width, mask(width));
            let left_is_minimum = self.compare(original_left, minimum, IntPredicate::EQ)?;
            let right_is_minus_one = self.compare(original_right, minus_one, IntPredicate::EQ)?;
            let signed_overflow = self.bitwise(left_is_minimum, right_is_minus_one, BitOp::And)?;
            self.bitwise(divide_by_zero, signed_overflow, BitOp::Or)?
        } else {
            divide_by_zero
        };
        Ok((
            self.select(invalid.clone(), zero.clone(), quotient)?,
            self.select(invalid, zero, remainder)?,
        ))
    }

    fn same_width(
        &self,
        left: &SymInt<Backend::Wire>,
        right: &SymInt<Backend::Wire>,
    ) -> Result<(), FrontendError> {
        if left.width == right.width {
            Ok(())
        } else {
            Err(FrontendError::execution("integer operand widths differ"))
        }
    }
}

#[derive(Clone)]
enum IdxOrConst<W = Idx> {
    Wire(W),
    Constant(bool),
}

impl<W> SymInt<W> {
    fn concrete(width: u32, value: u64) -> Self {
        Self {
            width,
            value: SymRepr::Concrete(value & mask(width)),
        }
    }
    fn as_concrete(&self) -> Option<ConcreteInt> {
        match self.value {
            SymRepr::Concrete(value) => Some(ConcreteInt {
                width: self.width,
                value,
            }),
            SymRepr::Wires(_) => None,
        }
    }
    fn signed_concrete(&self) -> Option<i64> {
        self.as_concrete()
            .map(|value| signed(value.value, value.width))
    }
}

#[derive(Clone, Copy)]
struct ConcreteInt {
    width: u32,
    value: u64,
}

fn int_from_host<W>(value: HostInteger<W>) -> Result<SymInt<W>, FrontendError> {
    let value = match value {
        HostInteger::Concrete { width, value } => SymInt::concrete(width, value),
        HostInteger::Symbolic { width, bits } => {
            if bits.len() != width as usize {
                return Err(FrontendError::request(
                    "host symbolic integer bit count does not match its width",
                ));
            }
            SymInt {
                width,
                value: SymRepr::Wires(bits),
            }
        }
    };
    if matches!(value.width, 1 | 8 | 16 | 32 | 64) {
        Ok(value)
    } else {
        Err(FrontendError::unsupported(format!(
            "host integer width i{}",
            value.width
        )))
    }
}

fn basic_instruction(
    instruction: InstructionValue<'_>,
) -> Result<BasicValueEnum<'_>, FrontendError> {
    // All callers have already established that the instruction is non-void.
    // LLVM's C API represents its result with the instruction value itself.
    Ok(unsafe { BasicValueEnum::new(instruction.as_value_ref()) })
}

fn integer_width(instruction: InstructionValue<'_>) -> Result<u32, FrontendError> {
    let value = basic_instruction(instruction)?;
    match value {
        BasicValueEnum::IntValue(value) => Ok(value.get_type().get_bit_width()),
        _ => Err(FrontendError::unsupported("non-integer instruction result")),
    }
}

fn cstr(value: &CStr) -> String {
    value.to_string_lossy().into_owned()
}

fn matches_host_type(actual: BasicTypeEnum<'_>, expected: HostType) -> bool {
    match (actual, expected) {
        (BasicTypeEnum::IntType(actual), HostType::Integer(width)) => {
            actual.get_bit_width() == width
        }
        (BasicTypeEnum::PointerType(_), HostType::Pointer) => true,
        _ => false,
    }
}

fn matches_host_parameter_type(actual: BasicMetadataTypeEnum<'_>, expected: HostType) -> bool {
    match (actual, expected) {
        (BasicMetadataTypeEnum::IntType(actual), HostType::Integer(width)) => {
            actual.get_bit_width() == width
        }
        (BasicMetadataTypeEnum::PointerType(_), HostType::Pointer) => true,
        _ => false,
    }
}

const fn mask(width: u32) -> u64 {
    if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

fn signed(value: u64, width: u32) -> i64 {
    if width == 64 {
        value as i64
    } else {
        let shift = 64 - width;
        ((value << shift) as i64) >> shift
    }
}

fn check_width(width: u32) -> Result<(), FrontendError> {
    if matches!(width, 1 | 8 | 16 | 32 | 64) {
        Ok(())
    } else {
        Err(FrontendError::unsupported(format!(
            "integer width i{width}"
        )))
    }
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

    impl HostCall<Recorder> for AddOneHost {
        fn lower(
            &self,
            backend: &mut Recorder,
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

    impl HasError for BackendWrapper {
        type Error = Infallible;
    }

    impl ContextWithValue<bool> for BackendWrapper {
        type Wrapped = bool;
    }

    impl ContextWithCreate<bool> for BackendWrapper {
        fn create(&mut self, value: bool) -> Result<bool, Self::Error> {
            self.operations += 1;
            self.inner.create(value)
        }
    }

    impl ContextWithBitAnd<bool> for BackendWrapper {
        fn bitand(&mut self, left: bool, right: bool) -> Result<bool, Self::Error> {
            self.operations += 1;
            self.inner.bitand(left, right)
        }

        fn bitand_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Self::Error> {
            self.operations += 1;
            self.inner.bitand_assign(left, right)
        }
    }

    impl ContextWithBitOr<bool> for BackendWrapper {
        fn bitor(&mut self, left: bool, right: bool) -> Result<bool, Self::Error> {
            self.operations += 1;
            self.inner.bitor(left, right)
        }

        fn bitor_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Self::Error> {
            self.operations += 1;
            self.inner.bitor_assign(left, right)
        }
    }

    impl ContextWithBitXor<bool> for BackendWrapper {
        fn bitxor(&mut self, left: bool, right: bool) -> Result<bool, Self::Error> {
            self.operations += 1;
            self.inner.bitxor(left, right)
        }

        fn bitxor_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Self::Error> {
            self.operations += 1;
            self.inner.bitxor_assign(left, right)
        }
    }

    impl ContextWithMux<bool> for BackendWrapper {
        fn mux(
            &mut self,
            condition: bool,
            then_value: bool,
            else_value: bool,
        ) -> Result<bool, Self::Error> {
            self.operations += 1;
            self.inner.mux(condition, then_value, else_value)
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

        let mut recorder_hosts = HostCallRegistry::<Recorder>::new();
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
