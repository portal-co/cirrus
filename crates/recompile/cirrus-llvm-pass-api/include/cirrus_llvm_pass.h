#ifndef CIRRUS_LLVM_PASS_H
#define CIRRUS_LLVM_PASS_H

#include <stdint.h>

#define CIRRUS_ABI_VERSION_V1 1u
#define CIRRUS_SELECT_EXACT 1u
#define CIRRUS_SELECT_PREFIX 2u
#define CIRRUS_ARGUMENT_SYMBOLIC 1u
#define CIRRUS_ARGUMENT_CONCRETE 2u
#define CIRRUS_ARGUMENT_REGION 3u
#define CIRRUS_BYTE_CONCRETE 1u
#define CIRRUS_BYTE_SYMBOLIC 2u
#define CIRRUS_EXPORT_RETURN 1u
#define CIRRUS_EXPORT_ARGUMENT_MEMORY 2u
#define CIRRUS_EXPORT_GLOBAL_MEMORY 3u

typedef struct { uint8_t kind, value; uint8_t reserved[2]; } CirrusRegionByteV1;
typedef struct { uint32_t kind, writable; uint64_t value; const CirrusRegionByteV1 *bytes; uint32_t bytes_len, reserved; } CirrusArgumentV1;
typedef struct { const char *name; uint32_t writable, reserved; const CirrusRegionByteV1 *bytes; uint32_t bytes_len, reserved2; } CirrusGlobalV1;
typedef struct { uint32_t kind, argument; const char *name; uint32_t offset, len; } CirrusExportV1;
typedef struct { uint64_t max_instructions, max_calls, max_alloca_bytes; } CirrusLoweringLimitsV1;
typedef struct { uint32_t version, reserved; const CirrusArgumentV1 *arguments; uint32_t arguments_len, globals_len; const CirrusGlobalV1 *globals; const CirrusExportV1 *exports; uint32_t exports_len, reserved2; CirrusLoweringLimitsV1 limits; } CirrusEntryDescriptorV1;
typedef struct { const void *target; const CirrusEntryDescriptorV1 *descriptor; uint32_t flags, reserved; } CirrusSelectorV1;
typedef struct { uint32_t version, reserved; const char *source_prefix, *output_prefix; const CirrusSelectorV1 *selectors; uint32_t selectors_len, reserved2; } CirrusModuleConfigV1;
typedef struct { uint32_t version, flags, slots, input_count, output_count; const uint32_t *inputs, *outputs; } CirrusProgramAbiV1;

/* This declaration must never survive to a final link: cirrus-llvm-pass
 * validates and erases retained marker functions which call it. */
extern void __cirrus_entry(const void *target, const CirrusEntryDescriptorV1 *descriptor);

#if defined(__clang__)
#define CIRRUS_USED __attribute__((used, noinline))
#define CIRRUS_ENTRY_MARKER(name, target, descriptor) \
    static CIRRUS_USED void name(void) { __cirrus_entry((const void *)(target), (descriptor)); }
#else
#define CIRRUS_USED
#define CIRRUS_ENTRY_MARKER(name, target, descriptor) \
    static void name(void) { __cirrus_entry((const void *)(target), (descriptor)); }
#endif

/* One module may provide this exact-name policy object instead of, or in
 * addition to, direct marker calls. It is retained in llvm.used. */
#define CIRRUS_MODULE_CONFIG(initializer) \
    static CIRRUS_USED const CirrusModuleConfigV1 __cirrus_module_config = (initializer)

#endif
