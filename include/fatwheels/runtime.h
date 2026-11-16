/**
 *  @file include/fatwheels/runtime.h
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Definitions of the CPU capability queries `c/fatwheels/capabilities.h` declares, and the header-only stubs
 *      of its dispatch points, whose tables only the library's dispatch unit holds.
 */
#ifndef FATWHEELS_RUNTIME_H_
#define FATWHEELS_RUNTIME_H_

#include "fatwheels/matmul.h"

#if FATWHEELS_ARCH_X8664_
#include <cpuid.h>     // `__get_cpuid_count`
#include <immintrin.h> // `_xgetbv`
#endif
#if defined(__linux__) && FATWHEELS_ARCH_X8664_
#include <sys/syscall.h> // `SYS_arch_prctl`
#endif
#if defined(__linux__) && FATWHEELS_ARCH_ARM64_
#include <sys/auxv.h> // `getauxval`, `AT_HWCAP`, `AT_HWCAP2`
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__linux__) && FATWHEELS_ARCH_X8664_
long syscall(long number, ...); // `unistd.h` declares it only under `_GNU_SOURCE`, which strict C modes lack
#endif

FATWHEELS_API fw_capability_t fw_capabilities_compiled_cpu(void) {
    return (fw_capability_t)(fw_cap_serial_k |                                     //
                             fw_cap_haswell_k * FATWHEELS_TARGET_HASWELL |         //
                             fw_cap_skylake_k * FATWHEELS_TARGET_SKYLAKE |         //
                             fw_cap_sapphireamx_k * FATWHEELS_TARGET_SAPPHIREAMX | //
                             fw_cap_sve_k * FATWHEELS_TARGET_SVE |                 //
                             fw_cap_sme_k * FATWHEELS_TARGET_SME);
}

#if FATWHEELS_ARCH_X8664_
#if defined(__clang__) || defined(__GNUC__)
__attribute__((target("xsave")))
#endif
static inline unsigned long long fw_xcr0_(void) {
    return _xgetbv(0);
}
#endif

FATWHEELS_API fw_capability_t fw_capabilities_detected_cpu(void) {
    unsigned capabilities = fw_cap_serial_k;
#if FATWHEELS_ARCH_X8664_
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid_count(1, 0, &eax, &ebx, &ecx, &edx) || !(ecx & bit_OSXSAVE)) return fw_cap_serial_k;
    int const fma = (ecx & bit_FMA) != 0;
    unsigned long long const xcr0 = fw_xcr0_();
    int const ymm = (xcr0 & 0x6) == 0x6, zmm = (xcr0 & 0xE6) == 0xE6, tiles = (xcr0 & 0x60000) == 0x60000;
    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
    if (ymm && fma && (ebx & bit_AVX2)) capabilities |= fw_cap_haswell_k;
    unsigned const skylake = bit_AVX512F | bit_AVX512CD | bit_AVX512VL | bit_AVX512BW | bit_AVX512DQ;
    if ((capabilities & fw_cap_haswell_k) && zmm && (ebx & skylake) == skylake) capabilities |= fw_cap_skylake_k;
    int amx = (capabilities & fw_cap_skylake_k) && (edx & (1u << 24)) && (edx & (1u << 22)); // AMX-TILE, AMX-BF16
#if defined(__linux__)
    // Linux grants the tile data state per process only on request; 0x1023 is ARCH_REQ_XCOMP_PERM, 18 its feature.
    amx = amx && syscall(SYS_arch_prctl, 0x1023, 18) == 0;
#else
    amx = amx && tiles;
#endif
    (void)tiles;
    if (amx) capabilities |= fw_cap_sapphireamx_k;
#elif FATWHEELS_ARCH_ARM64_ && defined(__linux__)
    unsigned long const hwcap = getauxval(AT_HWCAP), hwcap2 = getauxval(AT_HWCAP2);
    if (hwcap & (1ul << 22)) capabilities |= fw_cap_sve_k;                                     // HWCAP_SVE
    if ((capabilities & fw_cap_sve_k) && (hwcap2 & (1ul << 23))) capabilities |= fw_cap_sme_k; // HWCAP2_SME
#endif
    return (fw_capability_t)capabilities;
}

FATWHEELS_API fw_capability_t fw_capabilities_enabled_cpu(void) {
    return (fw_capability_t)(fw_capabilities_compiled_cpu() & fw_capabilities_detected_cpu());
}

#if FATWHEELS_HEADER_ONLY
FATWHEELS_API fw_matmul_f32_t fw_matmul_f32_find(fw_capability_t capabilities) {
    (void)capabilities;
    return NULL;
}

FATWHEELS_API fw_status_t fw_matmul_f32_best(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                             fw_size_t columns, fw_size_t depth, fw_capability_t capabilities,
                                             void *stream) {
    (void)a, (void)b, (void)c, (void)rows, (void)columns, (void)depth, (void)capabilities, (void)stream;
    return fw_missing_library_k;
}

FATWHEELS_API fw_device_services_t const *fw_device_services(fw_capability_t capabilities) {
    (void)capabilities;
    return NULL;
}

FATWHEELS_API fw_size_t fw_devices(fw_device_t *into, fw_size_t capacity) {
    (void)into, (void)capacity;
    return 0;
}

FATWHEELS_API fw_status_t fw_placement_resolve(fw_device_t const *table, fw_size_t table_count,
                                               fw_size_t const *positions, fw_size_t count, fw_placement_t *placement) {
    (void)table, (void)table_count, (void)positions, (void)count, (void)placement;
    return fw_missing_library_k;
}
#endif // FATWHEELS_HEADER_ONLY

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // FATWHEELS_RUNTIME_H_
