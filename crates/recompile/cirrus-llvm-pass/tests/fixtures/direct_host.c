#include <stdint.h>
#include "cirrus_llvm_pass.h"

/* A no_std static Rust library retains unwind-table references to this
 * personality symbol even with panic=abort. The Cirrus fixture never unwinds,
 * so the C host supplies the conventional inert definition. */
void rust_eh_personality(void) {}

extern void __cirrus_kernel(void *, void *);
extern const CirrusProgramAbiV1 __cirrus_kernel_abi;

int main(void) {
    uint8_t buffer[128] = { 0 };
    const CirrusProgramAbiV1 *abi = &__cirrus_kernel_abi;
    if (abi->version != CIRRUS_ABI_VERSION_V1 || abi->input_count != 8 || abi->output_count != 8)
        return 10;
    for (uint32_t bit = 0; bit < 8; ++bit)
        buffer[abi->inputs[bit]] = (uint8_t)((5u >> bit) & 1u);
    __cirrus_kernel(0, buffer);
    uint8_t output = 0;
    for (uint32_t bit = 0; bit < 8; ++bit)
        output |= (uint8_t)(buffer[abi->outputs[bit]] << bit);
    return output == (uint8_t)(5u ^ 7u) ? 0 : 11;
}
