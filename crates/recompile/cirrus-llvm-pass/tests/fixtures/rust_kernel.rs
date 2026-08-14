#![no_std]

use cirrus_llvm_pass_api::{
    cirrus_entry_marker, cirrus_module_config, CirrusArgumentV1, CirrusEntryDescriptorV1,
    CirrusExportV1, CirrusLoweringLimitsV1, CirrusModuleConfigV1, ABI_VERSION_V1,
    ARGUMENT_SYMBOLIC, EXPORT_RETURN,
};

#[unsafe(no_mangle)]
pub extern "C" fn kernel(value: u8) -> u8 {
    value ^ 7
}

static ARGUMENTS: [CirrusArgumentV1; 1] = [CirrusArgumentV1 {
    kind: ARGUMENT_SYMBOLIC,
    writable: 0,
    value: 0,
    bytes: core::ptr::null(),
    bytes_len: 0,
    reserved: 0,
}];
static EXPORTS: [CirrusExportV1; 1] = [CirrusExportV1 {
    kind: EXPORT_RETURN,
    argument: 0,
    name: core::ptr::null(),
    offset: 0,
    len: 0,
}];
static DESCRIPTOR: CirrusEntryDescriptorV1 = CirrusEntryDescriptorV1 {
    version: ABI_VERSION_V1,
    reserved: 0,
    arguments: ARGUMENTS.as_ptr(),
    arguments_len: 1,
    globals_len: 0,
    globals: core::ptr::null(),
    exports: EXPORTS.as_ptr(),
    exports_len: 1,
    reserved2: 0,
    limits: CirrusLoweringLimitsV1 {
        max_instructions: 10_000,
        max_calls: 1_000,
        max_alloca_bytes: 4_096,
    },
};

cirrus_entry_marker!(MARKER, kernel, &DESCRIPTOR);

static SOURCE_PREFIX: [u8; 1] = [0];
static OUTPUT_PREFIX: [u8; 10] = *b"__cirrus_\0";
cirrus_module_config!(CirrusModuleConfigV1 {
    version: ABI_VERSION_V1,
    reserved: 0,
    source_prefix: SOURCE_PREFIX.as_ptr(),
    output_prefix: OUTPUT_PREFIX.as_ptr(),
    selectors: core::ptr::null(),
    selectors_len: 0,
    reserved2: 0,
});

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}
