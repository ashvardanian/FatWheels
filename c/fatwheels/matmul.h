/**
 *  @file c/fatwheels/matmul.h
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Declarations of every capability's `matmul` kernel, each under its target alone.
 *
 *  A unit that sets a target to 1 sees the kernel's declaration whatever its own flags enable, so a unit compiled at
 *  the floor can call kernels the library holds. Definitions live in `include/fatwheels/<capability>.h`, which a
 *  header-only unit includes through the tail below and each capability unit includes for its own capability.
 */
#ifndef FATWHEELS_MATMUL_H_
#define FATWHEELS_MATMUL_H_

#include "fatwheels/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if FATWHEELS_TARGET_SERIAL
FATWHEELS_API fw_status_t fw_matmul_f32_serial(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                        fw_size_t columns, fw_size_t depth, void *stream);
#endif
#if FATWHEELS_TARGET_HASWELL
FATWHEELS_API fw_status_t fw_matmul_f32_haswell(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                         fw_size_t columns, fw_size_t depth, void *stream);
#endif
#if FATWHEELS_TARGET_SKYLAKE
FATWHEELS_API fw_status_t fw_matmul_f32_skylake(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                         fw_size_t columns, fw_size_t depth, void *stream);
#endif
#if FATWHEELS_TARGET_SAPPHIREAMX
FATWHEELS_API fw_status_t fw_matmul_f32_sapphireamx(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                             fw_size_t columns, fw_size_t depth, void *stream);
#endif
#if FATWHEELS_TARGET_SVE
FATWHEELS_API fw_status_t fw_matmul_f32_sve(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                     fw_size_t columns, fw_size_t depth, void *stream);
#endif
#if FATWHEELS_TARGET_SME
FATWHEELS_API fw_status_t fw_matmul_f32_sme(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                     fw_size_t columns, fw_size_t depth, void *stream);
#endif

/*  GPU kernels take device pointers and a stream, @c cudaStream_t or @c hipStream_t, and return without waiting. */
#if FATWHEELS_TARGET_CUDA
FATWHEELS_API fw_status_t fw_matmul_f32_cuda(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                             fw_size_t columns, fw_size_t depth, void *stream);
#endif
#if FATWHEELS_TARGET_AMPERE
FATWHEELS_API fw_status_t fw_matmul_f32_ampere(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                               fw_size_t columns, fw_size_t depth, void *stream);
#endif
#if FATWHEELS_TARGET_ADA
FATWHEELS_API fw_status_t fw_matmul_f32_ada(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                            fw_size_t columns, fw_size_t depth, void *stream);
#endif
#if FATWHEELS_TARGET_ROCM
FATWHEELS_API fw_status_t fw_matmul_f32_rocm(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                             fw_size_t columns, fw_size_t depth, void *stream);
#endif
#if FATWHEELS_TARGET_CDNA3
FATWHEELS_API fw_status_t fw_matmul_f32_cdna3(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                              fw_size_t columns, fw_size_t depth, void *stream);
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#if FATWHEELS_HEADER_ONLY
#include "fatwheels/serial.h"
#include "fatwheels/haswell.h"
#include "fatwheels/skylake.h"
#include "fatwheels/sapphireamx.h"
#include "fatwheels/sve.h"
#include "fatwheels/sme.h"
#include "fatwheels/cuda.cuh"
#include "fatwheels/ampere.cuh"
#include "fatwheels/ada.cuh"
#include "fatwheels/rocm.cuh"
#include "fatwheels/cdna3.cuh"
#endif // FATWHEELS_HEADER_ONLY

#endif // FATWHEELS_MATMUL_H_
