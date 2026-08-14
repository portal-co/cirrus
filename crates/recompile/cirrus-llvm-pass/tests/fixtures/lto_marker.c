#include <stdint.h>
#include "cirrus_llvm_pass.h"

extern uint8_t kernel(uint8_t);

static const CirrusArgumentV1 arguments[] = {
    { CIRRUS_ARGUMENT_SYMBOLIC, 0, 0, 0, 0, 0 },
};
static const CirrusExportV1 exports[] = {
    { CIRRUS_EXPORT_RETURN, 0, 0, 0, 0 },
};
static const CirrusEntryDescriptorV1 descriptor = {
    CIRRUS_ABI_VERSION_V1, 0, arguments, 1, 0, 0, exports, 1, 0,
    { 10000, 1000, 4096 },
};

CIRRUS_ENTRY_MARKER(cirrus_marker, kernel, &descriptor)
