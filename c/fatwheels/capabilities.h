/**
 *  @file c/fatwheels/capabilities.h
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief What this build compiled, what this machine runs, and the kernel for a capability.
 *
 *  - compiled: the capabilities whose kernels this build holds, from the targets the build set.
 *  - detected: the capabilities this CPU and its operating system run.
 *  - for a GPU, the loader answers both at once: a generation is compiled for a device when the driver loads that
 *    generation's kernels on it.
 */
#ifndef FATWHEELS_CAPABILITIES_H_
#define FATWHEELS_CAPABILITIES_H_

#include "fatwheels/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** The CPU capabilities this build holds kernels for. */
FATWHEELS_API fw_capability_t fw_capabilities_compiled_cpu(void);

/** The CPU capabilities this machine runs; on Linux it also asks the kernel for the AMX tile state. */
FATWHEELS_API fw_capability_t fw_capabilities_detected_cpu(void);

/** The CPU capabilities both compiled and detected, which CPU dispatch picks among. */
FATWHEELS_API fw_capability_t fw_capabilities_enabled_cpu(void);

/** The kernel of the best capability in @p capabilities this build holds and, on the CPU, enables, or `NULL`.
 *  A mask names one group: the CPU's, CUDA's or ROCm's. Header-only builds have no tables and return `NULL`. */
FATWHEELS_API fw_matmul_f32_t fw_matmul_f32_find(fw_capability_t capabilities);

/** Runs the kernel @ref fw_matmul_f32_find picks, or returns `fw_missing_capability_k`. */
FATWHEELS_API fw_status_t fw_matmul_f32_best(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                             fw_size_t columns, fw_size_t depth, fw_capability_t capabilities,
                                             void *stream);

/** One GPU vendor's runtime services, so host code moves memory and loads its own device code through one table
 *  whichever vendor runs it. Every entry reports failures as `fw_device_failure_k`. */
typedef struct {
    fw_capability_t (*loaded)(int device); // the generations whose kernels the driver loads on @p device
    fw_status_t (*allocate)(void **pointer, fw_size_t bytes);
    fw_status_t (*release)(void *pointer);
    fw_status_t (*copy_to_device)(void *target, void const *source, fw_size_t bytes, void *stream);
    fw_status_t (*copy_to_host)(void *target, void const *source, fw_size_t bytes, void *stream);
    fw_status_t (*synchronize)(void *stream);
    fw_status_t (*load_module)(void **module, void const *image); // a fatbin or a code object the caller embeds
    fw_status_t (*unload_module)(void *module);
    fw_status_t (*find_kernel)(void **kernel, void *module, char const *name);
    fw_status_t (*launch)(void *kernel, unsigned const grid[3], unsigned const block[3], void **arguments,
                          void *stream);
} fw_device_services_t;

/** The services of the GPU vendor whose capability group @p capabilities names, or `NULL` where this build lacks
 *  that vendor or @p capabilities names the CPU. */
FATWHEELS_API fw_device_services_t const *fw_device_services(fw_capability_t capabilities);

/** The GPU generations whose kernels load on @p device, or 0 without a driver or device. */
#if FATWHEELS_ARCH_CUDA_
FATWHEELS_API fw_capability_t fw_capabilities_loaded_cuda(int device);
#endif
#if FATWHEELS_ARCH_ROCM_
FATWHEELS_API fw_capability_t fw_capabilities_loaded_rocm(int device);
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#if FATWHEELS_HEADER_ONLY
#include "fatwheels/runtime.h"
#endif

#endif // FATWHEELS_CAPABILITIES_H_
