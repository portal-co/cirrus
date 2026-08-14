//! LLVM 22 pass-plugin implementation for descriptor-backed Cirrus kernels.
//!
//! The C++ shim owns LLVM's New-PM registration. This crate deliberately owns
//! all policy: it decodes only immutable LLVM constants, creates the existing
//! frontend request, and emits a prepared companion into the borrowed module.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CString, c_char};
use std::mem::ManuallyDrop;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use cirrus_llvm::emit_prepared_into_module;
use cirrus_llvm_frontend::{
    ArgumentBinding, Export, GlobalBinding, HostCallRegistry, LowerRequest, LoweringLimits,
    RegionBinding, RegionByte, ScalarBinding, lower_module_prepared,
};
use cirrus_llvm_pass_api::{
    ABI_VERSION_V1, ARGUMENT_CONCRETE, ARGUMENT_REGION, ARGUMENT_SYMBOLIC, BYTE_CONCRETE,
    BYTE_SYMBOLIC, EXPORT_ARGUMENT_MEMORY, EXPORT_GLOBAL_MEMORY, EXPORT_RETURN, SELECT_EXACT,
    SELECT_PREFIX,
};
use inkwell::AddressSpace;
use inkwell::context::Context;
use inkwell::llvm_sys::LLVMTypeKind;
use inkwell::llvm_sys::core::{
    LLVMConstIntGetZExtValue, LLVMGetArgOperand, LLVMGetArrayLength2, LLVMGetAsString,
    LLVMGetElementType, LLVMGetInitializer, LLVMGetModuleContext, LLVMGetNumOperands,
    LLVMGetOperand, LLVMGetTypeKind, LLVMInstructionEraseFromParent, LLVMIsAConstantAggregateZero,
    LLVMIsAConstantDataSequential, LLVMIsAConstantExpr, LLVMIsAConstantInt, LLVMIsAFunction,
    LLVMIsAGlobalVariable, LLVMTypeOf,
};
use inkwell::llvm_sys::prelude::LLVMModuleRef;
use inkwell::module::{Linkage, Module};
use inkwell::types::{AnyTypeEnum, BasicMetadataTypeEnum, BasicType, BasicTypeEnum, StructType};
use inkwell::values::{AsValueRef, CallSiteValue, FunctionValue, InstructionOpcode, PointerValue};

unsafe extern "C" {
    fn cirrus_llvm_pass_link_anchor();
}

const MARKER: &str = "__cirrus_entry";
const CONFIG: &str = "__cirrus_module_config";
const DEFAULT_OUTPUT_PREFIX: &str = "__cirrus_";

#[derive(Clone, Debug, Eq, PartialEq)]
struct Plan {
    arguments: Vec<ArgumentBinding>,
    globals: Vec<GlobalBinding>,
    exports: Vec<Export>,
    limits: LoweringLimits,
}

/// Rust currently serializes even `repr(C)` private statics as compact LLVM
/// byte aggregates. This view restores their C ABI field addresses without
/// dereferencing host memory; pointer cells remain borrowed LLVM constants.
#[derive(Clone, Debug)]
struct CompactMemory {
    bytes: Vec<u8>,
    pointers: BTreeMap<usize, inkwell::llvm_sys::prelude::LLVMValueRef>,
}

#[derive(Clone, Debug)]
struct Candidate<'ctx> {
    target: FunctionValue<'ctx>,
    plan: Plan,
    output_prefix: String,
}

#[derive(Clone, Debug)]
struct ModuleConfig<'ctx> {
    source_prefix: String,
    output_prefix: String,
    selectors: Vec<Selector<'ctx>>,
}

#[derive(Clone, Debug)]
struct Selector<'ctx> {
    target: FunctionValue<'ctx>,
    plan: Plan,
    flags: u32,
}

#[derive(Clone, Debug)]
struct PassError(String);

impl PassError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl core::fmt::Display for PassError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// C++ calls this with an LLVM-owned module. `0` means no candidates, `1`
/// means the module changed, and `-1` returns an allocated diagnostic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cirrus_llvm_pass_run(
    raw_module: LLVMModuleRef,
    error: *mut *mut c_char,
) -> i32 {
    // SAFETY: this empty C++ function deliberately anchors the pass-plugin
    // archive member containing llvmGetPassPluginInfo in the final cdylib.
    unsafe { cirrus_llvm_pass_link_anchor() };
    if !error.is_null() {
        // SAFETY: caller supplies a valid output slot.
        unsafe { *error = ptr::null_mut() };
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the C++ pass manager owns both references for the duration
        // of this call. ManuallyDrop below prevents inkwell from disposing
        // either of them.
        unsafe { run_borrowed_module(raw_module) }
    }));
    match result {
        Ok(Ok(changed)) => i32::from(changed),
        Ok(Err(reason)) => store_error(error, reason.to_string()),
        Err(_) => store_error(
            error,
            "Cirrus pass panicked; no LLVM state was retained".into(),
        ),
    }
}

/// Release a diagnostic allocated by [`cirrus_llvm_pass_run`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cirrus_llvm_pass_free_error(error: *mut c_char) {
    if !error.is_null() {
        // SAFETY: this function only receives pointers made by CString::into_raw.
        unsafe { drop(CString::from_raw(error)) };
    }
}

fn store_error(out: *mut *mut c_char, message: String) -> i32 {
    if !out.is_null() {
        let message = CString::new(message)
            .unwrap_or_else(|_| CString::new("Cirrus pass diagnostic contained NUL").unwrap());
        // SAFETY: caller supplied a valid output slot.
        unsafe { *out = message.into_raw() };
    }
    -1
}

unsafe fn run_borrowed_module(raw_module: LLVMModuleRef) -> Result<bool, PassError> {
    if raw_module.is_null() {
        return Err(PassError::new("received null LLVM module"));
    }
    // SAFETY: raw_module is borrowed from LLVM's pass manager.
    let context = ManuallyDrop::new(unsafe { Context::new(LLVMGetModuleContext(raw_module)) });
    // SAFETY: raw_module stays valid and must not be disposed by this crate.
    let module = ManuallyDrop::new(unsafe { Module::new(raw_module) });
    lower_module_with_pass(&context, &module)
}

fn lower_module_with_pass<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
) -> Result<bool, PassError> {
    let config = decode_module_config(module)?;
    let mut candidates = BTreeMap::<String, Candidate<'ctx>>::new();
    let output_prefix = config
        .as_ref()
        .map_or(DEFAULT_OUTPUT_PREFIX, |config| &config.output_prefix);

    if let Some(config) = &config {
        for selector in &config.selectors {
            let name = value_name(selector.target.as_value_ref())?;
            let selected = selector.flags & SELECT_EXACT != 0
                || (selector.flags & SELECT_PREFIX != 0 && name.starts_with(&config.source_prefix));
            if selected {
                insert_candidate(
                    &mut candidates,
                    name,
                    Candidate {
                        target: selector.target,
                        plan: selector.plan.clone(),
                        output_prefix: config.output_prefix.clone(),
                    },
                )?;
            }
        }
    }

    let marker_calls = find_marker_calls(module)?;
    for (call, _, target, plan) in &marker_calls {
        let name = value_name(target.as_value_ref())?;
        insert_candidate(
            &mut candidates,
            name,
            Candidate {
                target: *target,
                plan: plan.clone(),
                output_prefix: output_prefix.to_owned(),
            },
        )?;
        let _ = call;
    }

    if candidates.is_empty() {
        return Ok(false);
    }

    let marker_helpers = marker_calls
        .iter()
        .filter_map(|(_, owner, _, _)| marker_helper(*owner).then_some(*owner))
        .map(|owner| (owner.as_value_ref() as usize, owner))
        .collect::<BTreeMap<_, _>>();
    let marker_retention_globals = marker_retention_globals(module, &marker_helpers);
    let builder = context.create_builder();
    let mut changed = false;
    for (name, candidate) in &candidates {
        if candidate.target.count_basic_blocks() == 0 {
            return Err(PassError::new(format!(
                "selected target `{name}` is still an LLVM declaration; Full LTO must resolve it, and ThinLTO descriptors must be co-located with their definitions"
            )));
        }
        let output_name = format!("{}{name}", candidate.output_prefix);
        if companion_is_already_emitted(module, &output_name) {
            continue;
        }
        if module
            .get_function(&output_name)
            .is_some_and(|function| function.count_basic_blocks() != 0)
            || module
                .get_global(&format!("{output_name}_abi"))
                .is_some_and(|global| global.get_initializer().is_some())
            || module
                .get_global(&format!("{output_name}_inputs"))
                .is_some_and(|global| global.get_initializer().is_some())
            || module
                .get_global(&format!("{output_name}_outputs"))
                .is_some_and(|global| global.get_initializer().is_some())
        {
            return Err(PassError::new(format!(
                "generated companion `{output_name}` collides with an existing module symbol"
            )));
        }
        let hosts = HostCallRegistry::new();
        let request = LowerRequest {
            entry: name,
            arguments: &candidate.plan.arguments,
            globals: &candidate.plan.globals,
            exports: &candidate.plan.exports,
            limits: candidate.plan.limits,
            host_calls: &hosts,
        };
        let prepared = lower_module_prepared(module, &request)
            .map_err(|error| PassError::new(format!("target `{name}`: {error}")))?;
        emit_prepared_into_module(context, module, &builder, &prepared, &output_name)
            .map_err(|error| PassError::new(format!("target `{name}`: {error}")))?;
        emit_program_abi(context, module, &output_name, &prepared)?;
        changed = true;
    }

    let mut discarded = marker_helpers.keys().copied().collect::<BTreeSet<_>>();
    discarded.extend(marker_retention_globals.keys().copied());
    if config.is_some() {
        let global = module
            .get_global(CONFIG)
            .expect("decoded configuration remains in the borrowed module");
        discarded.insert(global.as_value_ref() as usize);
    }
    discard_llvm_used_entries(context, module, &discarded)?;

    for (call, _, _, _) in marker_calls {
        // SAFETY: calls were discovered before the module was mutated and no
        // generated code references them.
        unsafe { LLVMInstructionEraseFromParent(call) };
        changed = true;
    }
    for retention in marker_retention_globals.into_values() {
        // SAFETY: it was an llvm.used-only marker holder, removed from that
        // list above, and the helper it referenced has no runtime role.
        unsafe { retention.delete() };
    }
    for helper in marker_helpers.into_values() {
        // SAFETY: discarded llvm.used entries were removed above and the
        // helper consists exclusively of pseudo-intrinsic marker calls.
        unsafe { helper.delete() };
    }
    if let Some(marker) = module.get_function(MARKER) {
        // SAFETY: every direct use was erased above, and a valid marker is a
        // declaration only (validated while scanning it).
        unsafe { marker.delete() };
        changed = true;
    }
    if config.is_some() {
        // SAFETY: configuration is descriptor metadata only; no generated
        // companion references it and llvm.used no longer retains it.
        unsafe {
            module
                .get_global(CONFIG)
                .expect("decoded configuration remains in the borrowed module")
                .delete();
        }
        changed = true;
    }
    Ok(changed)
}

fn companion_is_already_emitted(module: &Module<'_>, output_name: &str) -> bool {
    module
        .get_function(output_name)
        .is_some_and(|function| function.count_basic_blocks() != 0)
        && module
            .get_global(&format!("{output_name}_abi"))
            .is_some_and(|global| global.get_initializer().is_some())
        && module
            .get_global(&format!("{output_name}_inputs"))
            .is_some_and(|global| global.get_initializer().is_some())
        && module
            .get_global(&format!("{output_name}_outputs"))
            .is_some_and(|global| global.get_initializer().is_some())
}

fn discard_llvm_used_entries<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    discarded: &BTreeSet<usize>,
) -> Result<(), PassError> {
    let Some(used) = module.get_global("llvm.used") else {
        return Ok(());
    };
    let initializer = unsafe { LLVMGetInitializer(used.as_value_ref()) };
    if initializer.is_null() {
        return Err(PassError::new("llvm.used must have a constant initializer"));
    }
    let mut retained = Vec::<PointerValue<'_>>::new();
    for index in 0..unsafe { LLVMGetNumOperands(initializer) } as u32 {
        let entry = unsafe { LLVMGetOperand(initializer, index) };
        let is_discarded = strip_pointer(entry, "llvm.used entry")
            .ok()
            .is_some_and(|value| discarded.contains(&(value as usize)));
        if !is_discarded {
            // SAFETY: llvm.used is an array of pointer constants by LLVM's
            // special-global contract.
            retained.push(unsafe { PointerValue::new(entry) });
        }
    }
    // SAFETY: replacing this special global is valid after retaining all
    // unrelated entries, and no generated operation refers to it.
    unsafe { used.delete() };
    if retained.is_empty() {
        return Ok(());
    }
    let pointer = context.ptr_type(AddressSpace::default());
    let replacement =
        module.add_global(pointer.array_type(retained.len() as u32), None, "llvm.used");
    replacement.set_linkage(Linkage::Appending);
    replacement.set_section(Some("llvm.metadata"));
    replacement.set_initializer(&pointer.const_array(&retained));
    Ok(())
}

fn insert_candidate<'ctx>(
    candidates: &mut BTreeMap<String, Candidate<'ctx>>,
    name: String,
    candidate: Candidate<'ctx>,
) -> Result<(), PassError> {
    if let Some(previous) = candidates.get(&name) {
        if previous.plan != candidate.plan || previous.output_prefix != candidate.output_prefix {
            return Err(PassError::new(format!(
                "conflicting Cirrus descriptors select `{name}`"
            )));
        }
        return Ok(());
    }
    candidates.insert(name, candidate);
    Ok(())
}

fn decode_module_config<'ctx>(
    module: &Module<'ctx>,
) -> Result<Option<ModuleConfig<'ctx>>, PassError> {
    let Some(global) = module.get_global(CONFIG) else {
        return Ok(None);
    };
    if !global.is_constant() {
        return Err(PassError::new("module config must be a constant global"));
    }
    let fields = aggregate_fields(global.as_value_ref(), "module config")?;
    if fields.len() != 7 {
        return decode_compact_module_config(global.as_value_ref()).map(Some);
    }
    require_fields(&fields, 7, "module config")?;
    require_version(&fields, "module config")?;
    require_zero(fields[1], "module config reserved")?;
    require_zero(fields[6], "module config reserved2")?;
    let source_prefix = c_string_from_pointer(fields[2], "module config source_prefix")?;
    let output_prefix = c_string_from_pointer(fields[3], "module config output_prefix")?;
    if output_prefix.is_empty() {
        return Err(PassError::new(
            "module config output_prefix must not be empty",
        ));
    }
    let selectors = decode_array_from_pointer(
        fields[4],
        int_u32(fields[5], "module config selectors_len")?,
        decode_selector,
    )?;
    Ok(Some(ModuleConfig {
        source_prefix,
        output_prefix,
        selectors,
    }))
}

fn decode_compact_module_config<'ctx>(
    raw: inkwell::llvm_sys::prelude::LLVMValueRef,
) -> Result<ModuleConfig<'ctx>, PassError> {
    let config = compact_memory(raw)?;
    if compact_u32(&config, 0, "module config version")? != ABI_VERSION_V1 {
        return Err(PassError::new("module config has unsupported ABI version"));
    }
    compact_zero(&config, 4, 4, "module config reserved")?;
    compact_zero(&config, 36, 4, "module config reserved2")?;
    let output_prefix = c_string_from_pointer(
        compact_pointer(&config, 16, "module config output_prefix")?,
        "module config output_prefix",
    )?;
    if output_prefix.is_empty() {
        return Err(PassError::new(
            "module config output_prefix must not be empty",
        ));
    }
    let selector_count = compact_u32(&config, 32, "module config selectors_len")?;
    let selectors = decode_compact_array_field(
        &config,
        24,
        selector_count,
        24,
        "module config selectors",
        decode_compact_selector,
    )?;
    Ok(ModuleConfig {
        source_prefix: c_string_from_pointer(
            compact_pointer(&config, 8, "module config source_prefix")?,
            "module config source_prefix",
        )?,
        output_prefix,
        selectors,
    })
}

fn decode_selector<'ctx>(
    raw: inkwell::llvm_sys::prelude::LLVMValueRef,
) -> Result<Selector<'ctx>, PassError> {
    let fields = aggregate_fields(raw, "selector")?;
    require_fields(&fields, 4, "selector")?;
    require_zero(fields[3], "selector reserved")?;
    let target = function_from_pointer(fields[0], "selector target")?;
    let descriptor = global_from_pointer(fields[1], "selector descriptor")?;
    let flags = int_u32(fields[2], "selector flags")?;
    if flags == 0 || flags & !(SELECT_EXACT | SELECT_PREFIX) != 0 {
        return Err(PassError::new(format!("invalid selector flags {flags}")));
    }
    Ok(Selector {
        target,
        plan: decode_plan(descriptor.as_value_ref())?,
        flags,
    })
}

fn decode_compact_selector<'ctx>(memory: CompactMemory) -> Result<Selector<'ctx>, PassError> {
    compact_zero(&memory, 20, 4, "selector reserved")?;
    let flags = compact_u32(&memory, 16, "selector flags")?;
    if flags == 0 || flags & !(SELECT_EXACT | SELECT_PREFIX) != 0 {
        return Err(PassError::new(format!("invalid selector flags {flags}")));
    }
    let target = function_from_pointer(
        compact_pointer(&memory, 0, "selector target")?,
        "selector target",
    )?;
    let descriptor = global_from_pointer(
        compact_pointer(&memory, 8, "selector descriptor")?,
        "selector descriptor",
    )?;
    Ok(Selector {
        target,
        plan: decode_plan(descriptor.as_value_ref())?,
        flags,
    })
}

fn find_marker_calls<'ctx>(
    module: &Module<'ctx>,
) -> Result<
    Vec<(
        inkwell::llvm_sys::prelude::LLVMValueRef,
        FunctionValue<'ctx>,
        FunctionValue<'ctx>,
        Plan,
    )>,
    PassError,
> {
    let mut calls = Vec::new();
    for function in module.get_functions() {
        for block in function.get_basic_blocks() {
            let mut instruction = block.get_first_instruction();
            while let Some(current) = instruction {
                instruction = current.get_next_instruction();
                if current.get_opcode() != InstructionOpcode::Call {
                    continue;
                }
                let Ok(call) = CallSiteValue::try_from(current) else {
                    continue;
                };
                let Some(callee) = call.get_called_fn_value() else {
                    continue;
                };
                if value_name(callee.as_value_ref())? != MARKER {
                    continue;
                }
                validate_marker_declaration(callee)?;
                if call.count_arguments() != 2 {
                    return Err(PassError::new(
                        "__cirrus_entry must take exactly (ptr target, ptr descriptor)",
                    ));
                }
                // SAFETY: count_arguments above proves these argument slots exist.
                let target = unsafe { LLVMGetArgOperand(call.as_value_ref(), 0) };
                // SAFETY: count_arguments above proves these argument slots exist.
                let descriptor = unsafe { LLVMGetArgOperand(call.as_value_ref(), 1) };
                let target = function_from_pointer(target, "__cirrus_entry target")?;
                let descriptor = global_from_pointer(descriptor, "__cirrus_entry descriptor")?;
                calls.push((
                    current.as_value_ref(),
                    function,
                    target,
                    decode_plan(descriptor.as_value_ref())?,
                ));
            }
        }
    }
    Ok(calls)
}

fn validate_marker_declaration(marker: FunctionValue<'_>) -> Result<(), PassError> {
    if marker.count_basic_blocks() != 0 || marker.get_linkage() != Linkage::External {
        return Err(PassError::new(
            "__cirrus_entry must be an external pseudo-intrinsic declaration",
        ));
    }
    let type_ = marker.get_type();
    if type_.is_var_arg()
        || type_.get_return_type().is_some()
        || type_.get_param_types().len() != 2
        || !type_
            .get_param_types()
            .iter()
            .all(|type_| matches!(type_, BasicMetadataTypeEnum::PointerType(_)))
    {
        return Err(PassError::new(
            "__cirrus_entry must have type void (ptr target, ptr descriptor)",
        ));
    }
    Ok(())
}

fn marker_helper(function: FunctionValue<'_>) -> bool {
    let blocks = function.get_basic_blocks();
    if blocks.len() != 1 {
        return false;
    }
    let mut saw_marker = false;
    let mut instruction = blocks[0].get_first_instruction();
    while let Some(current) = instruction {
        instruction = current.get_next_instruction();
        match current.get_opcode() {
            InstructionOpcode::Return => return saw_marker && instruction.is_none(),
            InstructionOpcode::Call => {
                let Ok(call) = CallSiteValue::try_from(current) else {
                    return false;
                };
                let Some(callee) = call.get_called_fn_value() else {
                    return false;
                };
                if value_name(callee.as_value_ref()).ok().as_deref() != Some(MARKER) {
                    return false;
                }
                saw_marker = true;
            }
            _ => return false,
        }
    }
    false
}

fn marker_retention_globals<'ctx>(
    module: &Module<'ctx>,
    helpers: &BTreeMap<usize, FunctionValue<'ctx>>,
) -> BTreeMap<usize, inkwell::values::GlobalValue<'ctx>> {
    let Some(used) = module.get_global("llvm.used") else {
        return BTreeMap::new();
    };
    let initializer = unsafe { LLVMGetInitializer(used.as_value_ref()) };
    if initializer.is_null() {
        return BTreeMap::new();
    }
    let mut retained = BTreeMap::new();
    for index in 0..unsafe { LLVMGetNumOperands(initializer) } as u32 {
        let entry = unsafe { LLVMGetOperand(initializer, index) };
        let Ok(global) = global_from_pointer(entry, "llvm.used marker entry") else {
            continue;
        };
        let initializer = unsafe { LLVMGetInitializer(global.as_value_ref()) };
        if initializer.is_null() {
            continue;
        }
        let points_to_marker = strip_pointer(initializer, "llvm.used marker initializer")
            .ok()
            .is_some_and(|value| helpers.contains_key(&(value as usize)));
        if points_to_marker {
            retained.insert(global.as_value_ref() as usize, global);
        }
    }
    retained
}

fn decode_plan(raw: inkwell::llvm_sys::prelude::LLVMValueRef) -> Result<Plan, PassError> {
    // SAFETY: every public caller obtains this from global_from_pointer.
    let descriptor = unsafe { inkwell::values::GlobalValue::new(raw) };
    if !descriptor.is_constant() {
        return Err(PassError::new("entry descriptor must be a constant global"));
    }
    let fields = aggregate_fields(raw, "entry descriptor")?;
    if fields.len() != 10 {
        return decode_compact_plan(raw);
    }
    require_fields(&fields, 10, "entry descriptor")?;
    require_version(&fields, "entry descriptor")?;
    require_zero(fields[1], "entry descriptor reserved")?;
    require_zero(fields[8], "entry descriptor reserved2")?;
    let arguments = decode_array_from_pointer(
        fields[2],
        int_u32(fields[3], "arguments_len")?,
        decode_argument,
    )?;
    let globals =
        decode_array_from_pointer(fields[5], int_u32(fields[4], "globals_len")?, decode_global)?;
    let exports =
        decode_array_from_pointer(fields[6], int_u32(fields[7], "exports_len")?, decode_export)?;
    let limits = decode_limits(fields[9])?;
    Ok(Plan {
        arguments,
        globals,
        exports,
        limits,
    })
}

fn decode_argument(
    raw: inkwell::llvm_sys::prelude::LLVMValueRef,
) -> Result<ArgumentBinding, PassError> {
    let fields = aggregate_fields(raw, "argument")?;
    require_fields(&fields, 6, "argument")?;
    require_zero(fields[5], "argument reserved")?;
    let kind = int_u32(fields[0], "argument kind")?;
    match kind {
        ARGUMENT_SYMBOLIC => Ok(ArgumentBinding::Scalar(ScalarBinding::Symbolic)),
        ARGUMENT_CONCRETE => Ok(ArgumentBinding::Scalar(ScalarBinding::Concrete(int_u64(
            fields[2],
            "argument value",
        )?))),
        ARGUMENT_REGION => Ok(ArgumentBinding::Region(RegionBinding {
            bytes: decode_array_from_pointer(
                fields[3],
                int_u32(fields[4], "argument bytes_len")?,
                decode_region_byte,
            )?,
            writable: int_u32(fields[1], "argument writable")? != 0,
        })),
        _ => Err(PassError::new(format!("unknown argument kind {kind}"))),
    }
}

fn decode_global(
    raw: inkwell::llvm_sys::prelude::LLVMValueRef,
) -> Result<GlobalBinding, PassError> {
    let fields = aggregate_fields(raw, "global binding")?;
    require_fields(&fields, 6, "global binding")?;
    require_zero(fields[2], "global binding reserved")?;
    require_zero(fields[5], "global binding reserved2")?;
    Ok(GlobalBinding {
        name: c_string_from_pointer(fields[0], "global binding name")?,
        region: RegionBinding {
            writable: int_u32(fields[1], "global writable")? != 0,
            bytes: decode_array_from_pointer(
                fields[3],
                int_u32(fields[4], "global bytes_len")?,
                decode_region_byte,
            )?,
        },
    })
}

fn decode_region_byte(
    raw: inkwell::llvm_sys::prelude::LLVMValueRef,
) -> Result<RegionByte, PassError> {
    let fields = aggregate_fields(raw, "region byte")?;
    require_fields(&fields, 3, "region byte")?;
    for reserved in aggregate_fields(fields[2], "region byte reserved")? {
        require_zero(reserved, "region byte reserved")?;
    }
    match int_u32(fields[0], "region byte kind")? {
        value if value == BYTE_CONCRETE as u32 => Ok(RegionByte::Concrete(int_u32(
            fields[1],
            "region byte value",
        )? as u8)),
        value if value == BYTE_SYMBOLIC as u32 => Ok(RegionByte::Symbolic),
        value => Err(PassError::new(format!("unknown region byte kind {value}"))),
    }
}

fn decode_export(raw: inkwell::llvm_sys::prelude::LLVMValueRef) -> Result<Export, PassError> {
    let fields = aggregate_fields(raw, "export")?;
    require_fields(&fields, 5, "export")?;
    let kind = int_u32(fields[0], "export kind")?;
    match kind {
        EXPORT_RETURN => Ok(Export::Return),
        EXPORT_ARGUMENT_MEMORY => Ok(Export::ArgumentMemory {
            argument: int_u32(fields[1], "export argument")? as usize,
            offset: int_u32(fields[3], "export offset")? as usize,
            len: int_u32(fields[4], "export len")? as usize,
        }),
        EXPORT_GLOBAL_MEMORY => Ok(Export::GlobalMemory {
            name: c_string_from_pointer(fields[2], "export global name")?,
            offset: int_u32(fields[3], "export offset")? as usize,
            len: int_u32(fields[4], "export len")? as usize,
        }),
        _ => Err(PassError::new(format!("unknown export kind {kind}"))),
    }
}

fn decode_limits(
    raw: inkwell::llvm_sys::prelude::LLVMValueRef,
) -> Result<LoweringLimits, PassError> {
    let fields = aggregate_fields(raw, "lowering limits")?;
    require_fields(&fields, 3, "lowering limits")?;
    Ok(LoweringLimits {
        max_instructions: usize_from_u64(
            int_u64(fields[0], "max_instructions")?,
            "max_instructions",
        )?,
        max_calls: usize_from_u64(int_u64(fields[1], "max_calls")?, "max_calls")?,
        max_alloca_bytes: usize_from_u64(
            int_u64(fields[2], "max_alloca_bytes")?,
            "max_alloca_bytes",
        )?,
    })
}

// A compact descriptor is a Rust `repr(C)` static which LLVM has represented
// as byte runs interspersed with relocatable pointers. All supported Rust LTO
// targets here are 64-bit; C and other source languages retain ordinary LLVM
// aggregates and use the decoder above.
const COMPACT_POINTER_BYTES: usize = 8;

fn decode_compact_plan(raw: inkwell::llvm_sys::prelude::LLVMValueRef) -> Result<Plan, PassError> {
    let descriptor = compact_memory(raw)?;
    let version = compact_u32(&descriptor, 0, "entry descriptor version")?;
    if version != ABI_VERSION_V1 {
        return Err(PassError::new(
            "entry descriptor has unsupported ABI version",
        ));
    }
    compact_zero(&descriptor, 4, 4, "entry descriptor reserved")?;
    let arguments_len = compact_u32(&descriptor, 16, "arguments_len")?;
    let globals_len = compact_u32(&descriptor, 20, "globals_len")?;
    let exports_len = compact_u32(&descriptor, 40, "exports_len")?;
    compact_zero(&descriptor, 44, 4, "entry descriptor reserved2")?;
    let arguments = decode_compact_array_field(
        &descriptor,
        8,
        arguments_len,
        32,
        "arguments",
        decode_compact_argument,
    )?;
    let globals = decode_compact_array_field(
        &descriptor,
        24,
        globals_len,
        32,
        "globals",
        decode_compact_global,
    )?;
    let exports = decode_compact_array_field(
        &descriptor,
        32,
        exports_len,
        24,
        "exports",
        decode_compact_export,
    )?;
    Ok(Plan {
        arguments,
        globals,
        exports,
        limits: LoweringLimits {
            max_instructions: usize_from_u64(
                compact_u64(&descriptor, 48, "max_instructions")?,
                "max_instructions",
            )?,
            max_calls: usize_from_u64(compact_u64(&descriptor, 56, "max_calls")?, "max_calls")?,
            max_alloca_bytes: usize_from_u64(
                compact_u64(&descriptor, 64, "max_alloca_bytes")?,
                "max_alloca_bytes",
            )?,
        },
    })
}

fn decode_compact_argument(memory: CompactMemory) -> Result<ArgumentBinding, PassError> {
    compact_zero(&memory, 28, 4, "argument reserved")?;
    match compact_u32(&memory, 0, "argument kind")? {
        ARGUMENT_SYMBOLIC => Ok(ArgumentBinding::Scalar(ScalarBinding::Symbolic)),
        ARGUMENT_CONCRETE => Ok(ArgumentBinding::Scalar(ScalarBinding::Concrete(
            compact_u64(&memory, 8, "argument value")?,
        ))),
        ARGUMENT_REGION => Ok(ArgumentBinding::Region(RegionBinding {
            writable: compact_u32(&memory, 4, "argument writable")? != 0,
            bytes: decode_compact_array_field(
                &memory,
                16,
                compact_u32(&memory, 24, "argument bytes_len")?,
                4,
                "argument bytes",
                decode_compact_region_byte,
            )?,
        })),
        kind => Err(PassError::new(format!("unknown argument kind {kind}"))),
    }
}

fn decode_compact_global(memory: CompactMemory) -> Result<GlobalBinding, PassError> {
    compact_zero(&memory, 12, 4, "global binding reserved")?;
    compact_zero(&memory, 28, 4, "global binding reserved2")?;
    Ok(GlobalBinding {
        name: c_string_from_pointer(
            compact_pointer(&memory, 0, "global binding name")?,
            "global binding name",
        )?,
        region: RegionBinding {
            writable: compact_u32(&memory, 8, "global writable")? != 0,
            bytes: decode_compact_array_field(
                &memory,
                16,
                compact_u32(&memory, 24, "global bytes_len")?,
                4,
                "global bytes",
                decode_compact_region_byte,
            )?,
        },
    })
}

fn decode_compact_region_byte(memory: CompactMemory) -> Result<RegionByte, PassError> {
    compact_zero(&memory, 2, 2, "region byte reserved")?;
    match compact_byte(&memory, 0, "region byte kind")? {
        BYTE_CONCRETE => Ok(RegionByte::Concrete(compact_byte(
            &memory,
            1,
            "region byte value",
        )?)),
        BYTE_SYMBOLIC => Ok(RegionByte::Symbolic),
        kind => Err(PassError::new(format!("unknown region byte kind {kind}"))),
    }
}

fn decode_compact_export(memory: CompactMemory) -> Result<Export, PassError> {
    match compact_u32(&memory, 0, "export kind")? {
        EXPORT_RETURN => Ok(Export::Return),
        EXPORT_ARGUMENT_MEMORY => Ok(Export::ArgumentMemory {
            argument: compact_u32(&memory, 4, "export argument")? as usize,
            offset: compact_u32(&memory, 16, "export offset")? as usize,
            len: compact_u32(&memory, 20, "export len")? as usize,
        }),
        EXPORT_GLOBAL_MEMORY => Ok(Export::GlobalMemory {
            name: c_string_from_pointer(
                compact_pointer(&memory, 8, "export global name")?,
                "export global name",
            )?,
            offset: compact_u32(&memory, 16, "export offset")? as usize,
            len: compact_u32(&memory, 20, "export len")? as usize,
        }),
        kind => Err(PassError::new(format!("unknown export kind {kind}"))),
    }
}

fn decode_compact_array<T>(
    pointer: inkwell::llvm_sys::prelude::LLVMValueRef,
    count: u32,
    element_bytes: usize,
    decode: impl Fn(CompactMemory) -> Result<T, PassError>,
) -> Result<Vec<T>, PassError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let memory = compact_memory(pointer)?;
    let total = usize::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(element_bytes))
        .ok_or_else(|| PassError::new("compact descriptor array is too large"))?;
    if memory.bytes.len() != total {
        return Err(PassError::new(format!(
            "compact descriptor array has {} bytes; expected {total}",
            memory.bytes.len()
        )));
    }
    (0..count as usize)
        .map(|index| decode(compact_slice(&memory, index * element_bytes, element_bytes)))
        .collect()
}

fn decode_compact_array_field<T>(
    memory: &CompactMemory,
    offset: usize,
    count: u32,
    element_bytes: usize,
    what: &str,
    decode: impl Fn(CompactMemory) -> Result<T, PassError>,
) -> Result<Vec<T>, PassError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    decode_compact_array(
        compact_pointer(memory, offset, what)?,
        count,
        element_bytes,
        decode,
    )
}

fn compact_memory(
    raw: inkwell::llvm_sys::prelude::LLVMValueRef,
) -> Result<CompactMemory, PassError> {
    let mut memory = CompactMemory {
        bytes: Vec::new(),
        pointers: BTreeMap::new(),
    };
    // The root is the descriptor/array global itself. Nested globals are
    // relocatable fields and therefore must remain pointer cells rather than
    // being recursively expanded into their initializers.
    let root = if !unsafe { LLVMIsAGlobalVariable(raw) }.is_null() {
        let initializer = unsafe { LLVMGetInitializer(raw) };
        if initializer.is_null() {
            return Err(PassError::new(
                "compact descriptor global must have a constant initializer",
            ));
        }
        initializer
    } else {
        raw
    };
    append_compact_value(root, &mut memory)?;
    Ok(memory)
}

fn append_compact_value(
    raw: inkwell::llvm_sys::prelude::LLVMValueRef,
    memory: &mut CompactMemory,
) -> Result<(), PassError> {
    if !unsafe { LLVMIsAConstantInt(raw) }.is_null() {
        let width = unsafe { inkwell::llvm_sys::core::LLVMGetIntTypeWidth(LLVMTypeOf(raw)) };
        let bytes = usize::try_from((width + 7) / 8).unwrap();
        let value = unsafe { LLVMConstIntGetZExtValue(raw) };
        memory
            .bytes
            .extend((0..bytes).map(|shift| (value >> (shift * 8)) as u8));
        return Ok(());
    }
    if !unsafe { LLVMIsAConstantAggregateZero(raw) }.is_null() {
        let zeros = compact_zero_type_bytes(unsafe { LLVMTypeOf(raw) })?;
        let offset = memory.bytes.len();
        memory.bytes.resize(offset + zeros, 0);
        return Ok(());
    }
    let kind = unsafe { LLVMGetTypeKind(LLVMTypeOf(raw)) };
    if kind == LLVMTypeKind::LLVMPointerTypeKind {
        let offset = memory.bytes.len();
        memory.pointers.insert(offset, raw);
        memory.bytes.resize(offset + COMPACT_POINTER_BYTES, 0);
        return Ok(());
    }
    if !unsafe { LLVMIsAConstantDataSequential(raw) }.is_null() {
        let mut string_len = 0usize;
        // SAFETY: ConstantDataSequential is the only LLVM value class for
        // which LLVMGetAsString is defined.
        let string = unsafe { LLVMGetAsString(raw, &mut string_len) };
        if string.is_null() {
            return Err(PassError::new(
                "compact constant data has no byte representation",
            ));
        }
        // SAFETY: LLVM owns string_len bytes for this constant data array.
        memory
            .bytes
            .extend(unsafe { std::slice::from_raw_parts(string.cast::<u8>(), string_len) });
        return Ok(());
    }
    let operands = unsafe { LLVMGetNumOperands(raw) };
    if operands <= 0 {
        return Err(PassError::new(
            "compact descriptor contains an unsupported constant",
        ));
    }
    for index in 0..operands as u32 {
        append_compact_value(unsafe { LLVMGetOperand(raw, index) }, memory)?;
    }
    Ok(())
}

fn compact_zero_type_bytes(
    type_: inkwell::llvm_sys::prelude::LLVMTypeRef,
) -> Result<usize, PassError> {
    match unsafe { LLVMGetTypeKind(type_) } {
        LLVMTypeKind::LLVMIntegerTypeKind => {
            let width = unsafe { inkwell::llvm_sys::core::LLVMGetIntTypeWidth(type_) };
            usize::try_from((width + 7) / 8)
                .map_err(|_| PassError::new("compact zero integer is too wide"))
        }
        LLVMTypeKind::LLVMArrayTypeKind => {
            let count = usize::try_from(unsafe { LLVMGetArrayLength2(type_) })
                .map_err(|_| PassError::new("compact zero array is too large"))?;
            count
                .checked_mul(compact_zero_type_bytes(unsafe {
                    LLVMGetElementType(type_)
                })?)
                .ok_or_else(|| PassError::new("compact zero array is too large"))
        }
        _ => Err(PassError::new(
            "compact descriptor contains an unsupported zero aggregate",
        )),
    }
}

fn compact_slice(memory: &CompactMemory, offset: usize, len: usize) -> CompactMemory {
    CompactMemory {
        bytes: memory.bytes[offset..offset + len].to_vec(),
        pointers: memory
            .pointers
            .range(offset..offset + len)
            .map(|(&cell, &pointer)| (cell - offset, pointer))
            .collect(),
    }
}

fn compact_u64(memory: &CompactMemory, offset: usize, what: &str) -> Result<u64, PassError> {
    let bytes = memory
        .bytes
        .get(offset..offset + 8)
        .ok_or_else(|| PassError::new(format!("compact {what} is out of bounds")))?;
    Ok(bytes.iter().enumerate().fold(0u64, |value, (index, byte)| {
        value | (u64::from(*byte) << (index * 8))
    }))
}

fn compact_u32(memory: &CompactMemory, offset: usize, what: &str) -> Result<u32, PassError> {
    let bytes = memory
        .bytes
        .get(offset..offset + 4)
        .ok_or_else(|| PassError::new(format!("compact {what} is out of bounds")))?;
    Ok(bytes.iter().enumerate().fold(0u32, |value, (index, byte)| {
        value | (u32::from(*byte) << (index * 8))
    }))
}

fn compact_byte(memory: &CompactMemory, offset: usize, what: &str) -> Result<u8, PassError> {
    memory
        .bytes
        .get(offset)
        .copied()
        .ok_or_else(|| PassError::new(format!("compact {what} is out of bounds")))
}

fn compact_pointer(
    memory: &CompactMemory,
    offset: usize,
    what: &str,
) -> Result<inkwell::llvm_sys::prelude::LLVMValueRef, PassError> {
    memory.pointers.get(&offset).copied().ok_or_else(|| {
        PassError::new(format!(
            "compact {what} must be a relocatable pointer constant"
        ))
    })
}

fn compact_zero(
    memory: &CompactMemory,
    offset: usize,
    len: usize,
    what: &str,
) -> Result<(), PassError> {
    let bytes = memory
        .bytes
        .get(offset..offset + len)
        .ok_or_else(|| PassError::new(format!("compact {what} is out of bounds")))?;
    if bytes.iter().any(|byte| *byte != 0) {
        return Err(PassError::new(format!("{what} must be zero")));
    }
    Ok(())
}

fn aggregate_fields(
    global_or_constant: inkwell::llvm_sys::prelude::LLVMValueRef,
    what: &str,
) -> Result<Vec<inkwell::llvm_sys::prelude::LLVMValueRef>, PassError> {
    let raw = if unsafe { LLVMIsAGlobalVariable(global_or_constant) }.is_null() {
        global_or_constant
    } else {
        // SAFETY: checked as a global above.
        let initializer = unsafe { LLVMGetInitializer(global_or_constant) };
        if initializer.is_null() {
            return Err(PassError::new(format!(
                "{what} must have a constant initializer"
            )));
        }
        initializer
    };
    let count = unsafe { LLVMGetNumOperands(raw) };
    if count < 0 {
        return Err(PassError::new(format!(
            "{what} is not a constant aggregate"
        )));
    }
    Ok((0..count as u32)
        .map(|index| unsafe { LLVMGetOperand(raw, index) })
        .collect())
}

fn require_fields<T>(fields: &[T], expected: usize, what: &str) -> Result<(), PassError> {
    if fields.len() != expected {
        return Err(PassError::new(format!(
            "{what} has {} fields; expected {expected} for Cirrus ABI V1",
            fields.len()
        )));
    }
    Ok(())
}

fn require_version(
    fields: &[inkwell::llvm_sys::prelude::LLVMValueRef],
    what: &str,
) -> Result<(), PassError> {
    if int_u32(fields[0], &format!("{what} version"))? != ABI_VERSION_V1 {
        return Err(PassError::new(format!(
            "{what} has unsupported ABI version"
        )));
    }
    Ok(())
}

fn require_zero(
    raw: inkwell::llvm_sys::prelude::LLVMValueRef,
    what: &str,
) -> Result<(), PassError> {
    if int_u64(raw, what)? != 0 {
        return Err(PassError::new(format!("{what} must be zero")));
    }
    Ok(())
}

fn int_u32(raw: inkwell::llvm_sys::prelude::LLVMValueRef, what: &str) -> Result<u32, PassError> {
    let value = int_u64(raw, what)?;
    u32::try_from(value).map_err(|_| PassError::new(format!("{what} does not fit u32")))
}

fn int_u64(raw: inkwell::llvm_sys::prelude::LLVMValueRef, what: &str) -> Result<u64, PassError> {
    if unsafe { LLVMIsAConstantInt(raw) }.is_null() {
        return Err(PassError::new(format!("{what} must be a constant integer")));
    }
    // SAFETY: checked as a constant integer above.
    Ok(unsafe { LLVMConstIntGetZExtValue(raw) as u64 })
}

fn usize_from_u64(value: u64, what: &str) -> Result<usize, PassError> {
    usize::try_from(value).map_err(|_| PassError::new(format!("{what} does not fit target usize")))
}

fn decode_array_from_pointer<T>(
    pointer: inkwell::llvm_sys::prelude::LLVMValueRef,
    expected: u32,
    decode: impl Fn(inkwell::llvm_sys::prelude::LLVMValueRef) -> Result<T, PassError>,
) -> Result<Vec<T>, PassError> {
    if expected == 0 {
        return Ok(Vec::new());
    }
    let global = global_from_pointer(pointer, "array pointer")?;
    if !global.is_constant() {
        return Err(PassError::new("array pointer must name a constant global"));
    }
    let initializer = unsafe { LLVMGetInitializer(global.as_value_ref()) };
    if initializer.is_null() {
        return Err(PassError::new(
            "array pointer must name a global constant initializer",
        ));
    }
    let count = unsafe { LLVMGetNumOperands(initializer) };
    if count != expected as i32 {
        return Err(PassError::new(format!(
            "array has {count} entries but descriptor declares {expected}"
        )));
    }
    (0..expected)
        .map(|index| decode(unsafe { LLVMGetOperand(initializer, index) }))
        .collect()
}

fn function_from_pointer<'ctx>(
    raw: inkwell::llvm_sys::prelude::LLVMValueRef,
    what: &str,
) -> Result<FunctionValue<'ctx>, PassError> {
    let raw = strip_pointer(raw, what)?;
    if unsafe { LLVMIsAFunction(raw) }.is_null() {
        return Err(PassError::new(format!(
            "{what} must be a direct LLVM function reference"
        )));
    }
    // SAFETY: checked as a function above and the surrounding module outlives
    // the inkwell value.
    unsafe { FunctionValue::new(raw) }.ok_or_else(|| PassError::new(format!("{what} is invalid")))
}

fn global_from_pointer<'ctx>(
    raw: inkwell::llvm_sys::prelude::LLVMValueRef,
    what: &str,
) -> Result<inkwell::values::GlobalValue<'ctx>, PassError> {
    let raw = strip_pointer(raw, what)?;
    if unsafe { LLVMIsAGlobalVariable(raw) }.is_null() {
        return Err(PassError::new(format!(
            "{what} must be a pointer to a global constant"
        )));
    }
    // SAFETY: checked as a global variable above and the module outlives it.
    // SAFETY: checked as a global variable above and the module outlives it.
    Ok(unsafe { inkwell::values::GlobalValue::new(raw) })
}

fn strip_pointer(
    mut raw: inkwell::llvm_sys::prelude::LLVMValueRef,
    what: &str,
) -> Result<inkwell::llvm_sys::prelude::LLVMValueRef, PassError> {
    for _ in 0..4 {
        if !unsafe { LLVMIsAFunction(raw) }.is_null()
            || !unsafe { LLVMIsAGlobalVariable(raw) }.is_null()
        {
            return Ok(raw);
        }
        if unsafe { LLVMIsAConstantExpr(raw) }.is_null() {
            break;
        }
        let operands = unsafe { LLVMGetNumOperands(raw) };
        if operands < 1 {
            break;
        }
        raw = unsafe { LLVMGetOperand(raw, 0) };
    }
    Err(PassError::new(format!(
        "{what} must be a direct global/function pointer constant"
    )))
}

fn c_string_from_pointer(
    raw: inkwell::llvm_sys::prelude::LLVMValueRef,
    what: &str,
) -> Result<String, PassError> {
    let global = global_from_pointer(raw, what)?;
    if !global.is_constant() {
        return Err(PassError::new(format!(
            "{what} must point to a constant byte string"
        )));
    }
    let initializer = unsafe { LLVMGetInitializer(global.as_value_ref()) };
    if initializer.is_null() {
        return Err(PassError::new(format!(
            "{what} must point to a constant byte string"
        )));
    }
    let bytes = if !unsafe { LLVMIsAConstantDataSequential(initializer) }.is_null() {
        let mut len = 0usize;
        // SAFETY: ConstantDataSequential is the only LLVM value class for
        // which LLVMGetAsString is defined.
        let bytes = unsafe { LLVMGetAsString(initializer, &mut len) };
        if bytes.is_null() {
            return Err(PassError::new(format!(
                "{what} must point to a NUL-terminated byte string"
            )));
        }
        // SAFETY: LLVM returned len readable bytes.
        unsafe { std::slice::from_raw_parts(bytes.cast::<u8>(), len) }.to_vec()
    } else {
        compact_memory(global.as_value_ref())?.bytes
    };
    if bytes.is_empty() {
        return Err(PassError::new(format!(
            "{what} must point to a non-empty NUL-terminated byte string"
        )));
    }
    let bytes = bytes
        .strip_suffix(&[0])
        .ok_or_else(|| PassError::new(format!("{what} must be NUL-terminated")))?;
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|_| PassError::new(format!("{what} must be UTF-8")))
}

fn value_name(raw: inkwell::llvm_sys::prelude::LLVMValueRef) -> Result<String, PassError> {
    // FunctionValue exposes the same name handling as the frontend and avoids
    // relying on an LLVM-owned NUL terminator.
    // SAFETY: every call site passes an LLVM function value.
    let function = unsafe { FunctionValue::new(raw) }
        .ok_or_else(|| PassError::new("expected named LLVM function"))?;
    std::str::from_utf8(function.get_name().to_bytes())
        .map(str::to_owned)
        .map_err(|_| PassError::new("LLVM function name is not UTF-8"))
}

fn emit_program_abi<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    output_name: &str,
    program: &cirrus_recompile_core::PreparedProgram,
) -> Result<(), PassError> {
    let i32_type = context.i32_type();
    let inputs_name = format!("{output_name}_inputs");
    let outputs_name = format!("{output_name}_outputs");
    let abi_name = format!("{output_name}_abi");
    if module
        .get_global(&inputs_name)
        .is_some_and(|global| global.get_initializer().is_some())
        || module
            .get_global(&outputs_name)
            .is_some_and(|global| global.get_initializer().is_some())
        || module
            .get_global(&abi_name)
            .is_some_and(|global| global.get_initializer().is_some())
    {
        return Err(PassError::new(format!(
            "generated ABI globals for `{output_name}` collide with existing symbols"
        )));
    }
    let inputs = program
        .inputs
        .iter()
        .map(|slot| i32_type.const_int(slot.0 as u64, false))
        .collect::<Vec<_>>();
    let outputs = program
        .outputs
        .iter()
        .map(|slot| i32_type.const_int(slot.0 as u64, false))
        .collect::<Vec<_>>();
    let input_type = i32_type.array_type(inputs.len() as u32);
    let output_type = i32_type.array_type(outputs.len() as u32);
    let input_global = reusable_abi_global(module, input_type, &inputs_name)?;
    input_global.set_initializer(&i32_type.const_array(&inputs));
    input_global.set_constant(true);
    input_global.set_linkage(Linkage::External);
    let output_global = reusable_abi_global(module, output_type, &outputs_name)?;
    output_global.set_initializer(&i32_type.const_array(&outputs));
    output_global.set_constant(true);
    output_global.set_linkage(Linkage::External);

    let ptr_type = context.ptr_type(AddressSpace::default());
    let default_abi_type = context.struct_type(
        &[
            i32_type.into(),
            i32_type.into(),
            i32_type.into(),
            i32_type.into(),
            i32_type.into(),
            ptr_type.into(),
            ptr_type.into(),
        ],
        false,
    );
    let (abi, abi_type) = reusable_abi_struct_global(module, default_abi_type, &abi_name)?;
    let fields = [
        i32_type.const_int(ABI_VERSION_V1 as u64, false).into(),
        i32_type.const_zero().into(),
        i32_type.const_int(program.slots as u64, false).into(),
        i32_type
            .const_int(program.inputs.len() as u64, false)
            .into(),
        i32_type
            .const_int(program.outputs.len() as u64, false)
            .into(),
        input_global.as_pointer_value().into(),
        output_global.as_pointer_value().into(),
    ];
    abi.set_initializer(&abi_type.const_named_struct(&fields));
    abi.set_constant(true);
    abi.set_linkage(Linkage::External);
    Ok(())
}

fn reusable_abi_struct_global<'ctx>(
    module: &Module<'ctx>,
    default_type: StructType<'ctx>,
    name: &str,
) -> Result<(inkwell::values::GlobalValue<'ctx>, StructType<'ctx>), PassError> {
    if let Some(global) = module.get_global(name) {
        if global.get_initializer().is_some() {
            return Err(PassError::new(format!(
                "generated ABI global `{name}` collides with an existing definition"
            )));
        }
        let AnyTypeEnum::StructType(type_) = global.get_value_type() else {
            return Err(PassError::new(format!(
                "ABI declaration `{name}` must have CirrusProgramAbiV1 structure type"
            )));
        };
        if !is_program_abi_type(type_) {
            return Err(PassError::new(format!(
                "ABI declaration `{name}` does not match CirrusProgramAbiV1"
            )));
        }
        return Ok((global, type_));
    }
    Ok((module.add_global(default_type, None, name), default_type))
}

fn is_program_abi_type(type_: StructType<'_>) -> bool {
    let fields = type_.get_field_types();
    fields.len() == 7
        && fields[..5].iter().all(|field| {
            matches!(field, BasicTypeEnum::IntType(integer) if integer.get_bit_width() == 32)
        })
        && fields[5].is_pointer_type()
        && fields[6].is_pointer_type()
}

fn reusable_abi_global<'ctx, Type: BasicType<'ctx>>(
    module: &Module<'ctx>,
    type_: Type,
    name: &str,
) -> Result<inkwell::values::GlobalValue<'ctx>, PassError> {
    if let Some(global) = module.get_global(name) {
        if global.get_initializer().is_some() || global.get_value_type() != type_.as_any_type_enum()
        {
            return Err(PassError::new(format!(
                "generated ABI global `{name}` collides with an existing definition"
            )));
        }
        return Ok(global);
    }
    Ok(module.add_global(type_, None, name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::memory_buffer::MemoryBuffer;

    fn module<'ctx>(context: &'ctx Context, source: &str) -> Module<'ctx> {
        context
            .create_module_from_ir(MemoryBuffer::create_from_memory_range_copy(
                source.as_bytes(),
                "pass.ll",
            ))
            .unwrap()
    }

    #[test]
    fn marker_lowering_emits_companion_and_metadata() {
        let context = Context::create();
        let module = module(
            &context,
            r#"
                @desc = private constant { i32, i32, ptr, i32, i32, ptr, ptr, i32, i32, { i64, i64, i64 } }
                    { i32 1, i32 0, ptr @args, i32 1, i32 0, ptr null, ptr @exports, i32 1, i32 0, { i64, i64, i64 } { i64 1000, i64 1000, i64 1024 } }
                @args = private constant [1 x { i32, i32, i64, ptr, i32, i32 }]
                    [{ i32, i32, i64, ptr, i32, i32 } { i32 1, i32 0, i64 0, ptr null, i32 0, i32 0 }]
                @exports = private constant [1 x { i32, i32, ptr, i32, i32 }]
                    [{ i32, i32, ptr, i32, i32 } { i32 1, i32 0, ptr null, i32 0, i32 0 }]
                @keep = private global i8 7
                @llvm.used = appending global [1 x ptr] [ptr @keep], section "llvm.metadata"
                declare void @__cirrus_entry(ptr, ptr)
                define i8 @kernel(i8 %x) {
                entry:
                    %out = xor i8 %x, 7
                    ret i8 %out
                }
                define void @marker() {
                entry:
                    call void @__cirrus_entry(ptr @kernel, ptr @desc)
                    ret void
                }
            "#,
        );
        assert!(lower_module_with_pass(&context, &module).unwrap());
        assert!(module.get_function("__cirrus_kernel").is_some());
        assert!(module.get_global("__cirrus_kernel_abi").is_some());
        assert!(module.get_function("marker").is_none());
        assert!(module.get_function(MARKER).is_none());
        assert!(module.get_global("keep").is_some());
    }

    #[test]
    fn conflicting_markers_are_rejected() {
        let context = Context::create();
        let module = module(
            &context,
            r#"
                @desc = private constant { i32, i32, ptr, i32, i32, ptr, ptr, i32, i32, { i64, i64, i64 } }
                    { i32 1, i32 0, ptr @args, i32 1, i32 0, ptr null, ptr @exports, i32 1, i32 0, { i64, i64, i64 } { i64 1000, i64 1000, i64 1024 } }
                @desc2 = private constant { i32, i32, ptr, i32, i32, ptr, ptr, i32, i32, { i64, i64, i64 } }
                    { i32 1, i32 0, ptr @args2, i32 1, i32 0, ptr null, ptr @exports, i32 1, i32 0, { i64, i64, i64 } { i64 1000, i64 1000, i64 1024 } }
                @args = private constant [1 x { i32, i32, i64, ptr, i32, i32 }]
                    [{ i32, i32, i64, ptr, i32, i32 } { i32 1, i32 0, i64 0, ptr null, i32 0, i32 0 }]
                @args2 = private constant [1 x { i32, i32, i64, ptr, i32, i32 }]
                    [{ i32, i32, i64, ptr, i32, i32 } { i32 2, i32 0, i64 2, ptr null, i32 0, i32 0 }]
                @exports = private constant [1 x { i32, i32, ptr, i32, i32 }]
                    [{ i32, i32, ptr, i32, i32 } { i32 1, i32 0, ptr null, i32 0, i32 0 }]
                declare void @__cirrus_entry(ptr, ptr)
                define i8 @kernel(i8 %x) { ret i8 %x }
                define void @marker() {
                    call void @__cirrus_entry(ptr @kernel, ptr @desc)
                    call void @__cirrus_entry(ptr @kernel, ptr @desc2)
                    ret void
                }
            "#,
        );
        assert!(
            lower_module_with_pass(&context, &module)
                .unwrap_err()
                .to_string()
                .contains("conflicting")
        );
    }

    #[test]
    fn config_exact_and_prefix_selectors_have_deterministic_output_names() {
        let context = Context::create();
        let module = module(
            &context,
            r#"
                @desc = private constant { i32, i32, ptr, i32, i32, ptr, ptr, i32, i32, { i64, i64, i64 } }
                    { i32 1, i32 0, ptr @args, i32 1, i32 0, ptr null, ptr @exports, i32 1, i32 0, { i64, i64, i64 } { i64 1000, i64 1000, i64 1024 } }
                @args = private constant [1 x { i32, i32, i64, ptr, i32, i32 }]
                    [{ i32, i32, i64, ptr, i32, i32 } { i32 1, i32 0, i64 0, ptr null, i32 0, i32 0 }]
                @exports = private constant [1 x { i32, i32, ptr, i32, i32 }]
                    [{ i32, i32, ptr, i32, i32 } { i32 1, i32 0, ptr null, i32 0, i32 0 }]
                @source_prefix = private constant [4 x i8] c"src\00"
                @output_prefix = private constant [5 x i8] c"cir_\00"
                @selectors = private constant [3 x { ptr, ptr, i32, i32 }]
                    [{ ptr, ptr, i32, i32 } { ptr @src_kernel, ptr @desc, i32 2, i32 0 },
                     { ptr, ptr, i32, i32 } { ptr @explicit_kernel, ptr @desc, i32 1, i32 0 },
                     { ptr, ptr, i32, i32 } { ptr @other_kernel, ptr @desc, i32 2, i32 0 }]
                @__cirrus_module_config = constant { i32, i32, ptr, ptr, ptr, i32, i32 }
                    { i32 1, i32 0, ptr @source_prefix, ptr @output_prefix, ptr @selectors, i32 3, i32 0 }
                define i8 @src_kernel(i8 %x) { %out = xor i8 %x, 1 ret i8 %out }
                define i8 @explicit_kernel(i8 %x) { %out = xor i8 %x, 2 ret i8 %out }
                define i8 @other_kernel(i8 %x) { %out = xor i8 %x, 3 ret i8 %out }
            "#,
        );
        assert!(lower_module_with_pass(&context, &module).unwrap());
        assert!(module.get_function("cir_src_kernel").is_some());
        assert!(module.get_function("cir_explicit_kernel").is_some());
        assert!(module.get_function("cir_other_kernel").is_none());
        assert!(module.get_global(CONFIG).is_none());
        // A later extension-point invocation sees the completed companions,
        // preserves all analyses, and does not grow the module again.
        assert!(!lower_module_with_pass(&context, &module).unwrap());
    }

    #[test]
    fn marker_signature_is_validated_before_lowering() {
        let context = Context::create();
        let module = module(
            &context,
            r#"
                declare void @__cirrus_entry(ptr)
                define i8 @kernel(i8 %x) { ret i8 %x }
                define void @marker() {
                entry:
                    call void @__cirrus_entry(ptr @kernel)
                    ret void
                }
            "#,
        );
        assert!(
            lower_module_with_pass(&context, &module)
                .unwrap_err()
                .to_string()
                .contains("__cirrus_entry")
        );
    }
}
