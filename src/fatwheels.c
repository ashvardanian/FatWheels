/**
 *  @file src/fatwheels.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The library's capability queries and kernel tables, over the kernels its capability units define.
 *
 *  Each dispatch point keeps one table per capability group, the CPU's and each GPU vendor's: a null slot, then one
 *  kernel per capability bit the build holds, ascending. A pick smears the best shared bit down and counts the
 *  group's bits at or below it, so it indexes the table without a branch per capability. A monoversioned build
 *  leaves only its best CPU kernel in the table, so it runs one version of each operation. Each GPU vendor's runtime
 *  services form one more table, indexed by the same group.
 */
#undef FATWHEELS_TARGET_SERIAL
#define FATWHEELS_TARGET_SERIAL 1 // declared here for the table, defined in its own unit
#include "fatwheels/fatwheels.h"
#include "fatwheels/runtime.h"

/** One group's kernels of one dispatch point: a null slot 0, then a kernel per bit of @c capabilities. */
typedef struct {
    fw_capability_t capabilities;
    fw_kernel_punned_t const *kernels;
} fw_capability_kernels_t;

typedef enum {
    fw_capability_group_cpu_k,
    fw_capability_group_cuda_k,
    fw_capability_group_rocm_k,
    fw_capability_groups_k,
} fw_capability_group_t;

/** What the CPU runs in this process, set once as the library loads. */
static fw_capability_t fw_capabilities_enabled_cpu_ = fw_cap_serial_k;

__attribute__((constructor)) static void fw_capabilities_enable_cpu_(void) {
    fw_capabilities_enabled_cpu_ = fw_capabilities_enabled_cpu();
}

/** Every bit of @p x at or below its highest set one, or zero. */
static inline uint64_t fw_u64_smear_down_(uint64_t x) {
    x |= x >> 1, x |= x >> 2, x |= x >> 4, x |= x >> 8, x |= x >> 16, x |= x >> 32;
    return x;
}

/** The group @p capabilities describes: each GPU vendor's bits sit above the CPU's. */
static inline fw_capability_group_t fw_capability_group_of_(fw_capability_t capabilities) {
    return (fw_capability_group_t)((capabilities >= fw_cap_cuda_k) + (capabilities >= fw_cap_rocm_k));
}

/** The kernel of the best capability @p capabilities shares with its group's kernels and, on the CPU, the enabled
 *  ones, or null from slot 0. */
static inline fw_kernel_punned_t fw_kernel_pick_(fw_capability_t capabilities,
                                                 fw_capability_kernels_t const groups[fw_capability_groups_k]) {
    fw_capability_group_t const group_index = fw_capability_group_of_(capabilities);
    fw_capability_kernels_t const *group = &groups[group_index];
    uint64_t const clamp = group_index ? ~(uint64_t)0 : (uint64_t)fw_capabilities_enabled_cpu_;
    uint64_t const at_or_below = fw_u64_smear_down_((uint64_t)capabilities & group->capabilities & clamp);
    return group->kernels[__builtin_popcountll(group->capabilities & at_or_below)];
}

static fw_capability_kernels_t const *fw_matmul_f32_capabilities(void) {
    enum {
        best_k = FATWHEELS_MULTIVERSIONED ? 0
        : FATWHEELS_TARGET_SME            ? fw_cap_sme_index_k
        : FATWHEELS_TARGET_SVE            ? fw_cap_sve_index_k
        : FATWHEELS_TARGET_SAPPHIREAMX    ? fw_cap_sapphireamx_index_k
        : FATWHEELS_TARGET_SKYLAKE        ? fw_cap_skylake_index_k
        : FATWHEELS_TARGET_HASWELL        ? fw_cap_haswell_index_k
                                          : fw_cap_serial_index_k,
    };
    static fw_kernel_punned_t const cpu[] = {
        NULL,
        best_k > (int)fw_cap_serial_index_k ? NULL : (fw_kernel_punned_t)&fw_matmul_f32_serial,
#if FATWHEELS_TARGET_HASWELL
        best_k > (int)fw_cap_haswell_index_k ? NULL : (fw_kernel_punned_t)&fw_matmul_f32_haswell,
#endif
#if FATWHEELS_TARGET_SKYLAKE
        best_k > (int)fw_cap_skylake_index_k ? NULL : (fw_kernel_punned_t)&fw_matmul_f32_skylake,
#endif
#if FATWHEELS_TARGET_SAPPHIREAMX
        best_k > (int)fw_cap_sapphireamx_index_k ? NULL : (fw_kernel_punned_t)&fw_matmul_f32_sapphireamx,
#endif
#if FATWHEELS_TARGET_SVE
        best_k > (int)fw_cap_sve_index_k ? NULL : (fw_kernel_punned_t)&fw_matmul_f32_sve,
#endif
#if FATWHEELS_TARGET_SME
        best_k > (int)fw_cap_sme_index_k ? NULL : (fw_kernel_punned_t)&fw_matmul_f32_sme,
#endif
    };
    static fw_kernel_punned_t const cuda[] = {
        NULL,
#if FATWHEELS_TARGET_CUDA
        (fw_kernel_punned_t)&fw_matmul_f32_cuda,
#endif
#if FATWHEELS_TARGET_AMPERE
        (fw_kernel_punned_t)&fw_matmul_f32_ampere,
#endif
#if FATWHEELS_TARGET_ADA
        (fw_kernel_punned_t)&fw_matmul_f32_ada,
#endif
    };
    static fw_kernel_punned_t const rocm[] = {
        NULL,
#if FATWHEELS_TARGET_ROCM
        (fw_kernel_punned_t)&fw_matmul_f32_rocm,
#endif
#if FATWHEELS_TARGET_CDNA3
        (fw_kernel_punned_t)&fw_matmul_f32_cdna3,
#endif
    };
    static fw_capability_kernels_t const groups[fw_capability_groups_k] = {
        {(fw_capability_t)(fw_cap_serial_k | fw_cap_haswell_k * FATWHEELS_TARGET_HASWELL |
                           fw_cap_skylake_k * FATWHEELS_TARGET_SKYLAKE |
                           fw_cap_sapphireamx_k * FATWHEELS_TARGET_SAPPHIREAMX | fw_cap_sve_k * FATWHEELS_TARGET_SVE |
                           fw_cap_sme_k * FATWHEELS_TARGET_SME),
         cpu},
        {(fw_capability_t)(fw_cap_cuda_k * FATWHEELS_TARGET_CUDA | fw_cap_ampere_k * FATWHEELS_TARGET_AMPERE |
                           fw_cap_ada_k * FATWHEELS_TARGET_ADA),
         cuda},
        {(fw_capability_t)(fw_cap_rocm_k * FATWHEELS_TARGET_ROCM | fw_cap_cdna3_k * FATWHEELS_TARGET_CDNA3), rocm},
    };
    return groups;
}

FATWHEELS_API fw_matmul_f32_t fw_matmul_f32_find(fw_capability_t capabilities) {
    return (fw_matmul_f32_t)fw_kernel_pick_(capabilities, fw_matmul_f32_capabilities());
}

FATWHEELS_API fw_status_t fw_matmul_f32_best(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                             fw_size_t columns, fw_size_t depth, fw_capability_t capabilities,
                                             void *stream) {
    fw_matmul_f32_t const kernel = fw_matmul_f32_find(capabilities);
    return kernel ? kernel(a, b, c, rows, columns, depth, stream) : fw_missing_capability_k;
}

#if FATWHEELS_TARGET_CUDA
extern fw_device_services_t const fw_device_services_cuda_;
#endif
#if FATWHEELS_TARGET_ROCM
extern fw_device_services_t const fw_device_services_rocm_;
#endif

FATWHEELS_API fw_device_services_t const *fw_device_services(fw_capability_t capabilities) {
    static fw_device_services_t const *const groups[fw_capability_groups_k] = {
        NULL,
#if FATWHEELS_TARGET_CUDA
        &fw_device_services_cuda_,
#else
        NULL,
#endif
#if FATWHEELS_TARGET_ROCM
        &fw_device_services_rocm_,
#else
        NULL,
#endif
    };
    return groups[fw_capability_group_of_(capabilities)];
}
