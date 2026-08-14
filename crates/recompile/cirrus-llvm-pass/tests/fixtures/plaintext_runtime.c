#include <stdint.h>

#if defined(__clang__)
#define CIRRUS_RUNTIME_EXPORT __attribute__((used, visibility("default")))
#else
#define CIRRUS_RUNTIME_EXPORT
#endif

CIRRUS_RUNTIME_EXPORT void cirrus_rt_create(void *backend, uint8_t *buffer, uint8_t value, uint32_t out) {
    (void)backend;
    buffer[out] = value != 0;
}

CIRRUS_RUNTIME_EXPORT void cirrus_rt_bitand(void *backend, uint8_t *buffer, uint32_t left, uint32_t right, uint32_t out) {
    (void)backend;
    buffer[out] = buffer[left] & buffer[right];
}

CIRRUS_RUNTIME_EXPORT void cirrus_rt_bitor(void *backend, uint8_t *buffer, uint32_t left, uint32_t right, uint32_t out) {
    (void)backend;
    buffer[out] = buffer[left] | buffer[right];
}

CIRRUS_RUNTIME_EXPORT void cirrus_rt_bitxor(void *backend, uint8_t *buffer, uint32_t left, uint32_t right, uint32_t out) {
    (void)backend;
    buffer[out] = buffer[left] ^ buffer[right];
}

CIRRUS_RUNTIME_EXPORT void cirrus_rt_mux(void *backend, uint8_t *buffer, uint32_t condition, uint32_t when_true, uint32_t when_false, uint32_t out) {
    (void)backend;
    buffer[out] = buffer[condition] ? buffer[when_true] : buffer[when_false];
}
