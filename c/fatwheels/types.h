/**
 *  @file c/fatwheels/types.h
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Scalar types, linkage, and the two flags every capability carries.
 *
 *  Two flags per capability, read in this order by every capability header:
 *
 *  - `FATWHEELS_ARCH_<ARCH>_<CAPABILITY>_` says the capability's code compiles in this unit. It is detected here, from
 *    the unit's flags or from compiling as a GPU vendor, and only this library's own build may raise or zero it, with
 *    the verdict of its probe. It never reads a target. A capability's flag also holds wherever the flag of a
 *    capability built on it holds, so building blocks compile wherever something uses them.
 *  - `FATWHEELS_TARGET_<CAPABILITY>` says this unit has the capability's kernels: the umbrella declares them, and a
 *    header-only unit or the capability's own unit defines them. It is public: a consumer may set it to 0 to drop
 *    kernels, or to 1 to see the declarations of kernels a linked library holds.
 */
#ifndef FATWHEELS_TYPES_H_
#define FATWHEELS_TYPES_H_

#include <stddef.h> // `size_t`
#include <stdint.h> // `uint64_t`

#if !defined(FATWHEELS_HEADER_ONLY)
#define FATWHEELS_HEADER_ONLY 0
#endif

/*  A multiversioned build dispatches among every capability it holds; a monoversioned one keeps one per operation. */
#if !defined(FATWHEELS_MULTIVERSIONED)
#define FATWHEELS_MULTIVERSIONED 1
#endif

/*  Header-only kernels have internal linkage, so units compiled for different capabilities never merge them. */
#if FATWHEELS_HEADER_ONLY
#define FATWHEELS_API static inline
#else
#define FATWHEELS_API __attribute__((visibility("default")))
#endif

#pragma region Architectures

/*  CPU architectures exclude GPU device passes, which never see CPU intrinsics. */
#if !defined(FATWHEELS_ARCH_X8664_)
#if defined(__x86_64__) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#define FATWHEELS_ARCH_X8664_ 1
#else
#define FATWHEELS_ARCH_X8664_ 0
#endif
#endif

#if !defined(FATWHEELS_ARCH_ARM64_)
#if defined(__aarch64__) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#define FATWHEELS_ARCH_ARM64_ 1
#else
#define FATWHEELS_ARCH_ARM64_ 0
#endif
#endif

/*  A vendor flag says the vendor's runtime is in this build: compiling as that vendor, or a host unit of a library
 *  that links its GPU units, whose build sets it. */
#if !defined(FATWHEELS_ARCH_CUDA_)
#if defined(__CUDACC__) && !defined(__HIP__)
#define FATWHEELS_ARCH_CUDA_ 1
#else
#define FATWHEELS_ARCH_CUDA_ 0
#endif
#endif

#if !defined(FATWHEELS_ARCH_ROCM_)
#if defined(__HIP__)
#define FATWHEELS_ARCH_ROCM_ 1
#else
#define FATWHEELS_ARCH_ROCM_ 0
#endif
#endif

#pragma endregion Architectures

#pragma region CPU Capabilities

/*  The serial kernels: every header-only unit defines them, and in the library only its serial unit does. */
#if !defined(FATWHEELS_TARGET_SERIAL)
#define FATWHEELS_TARGET_SERIAL FATWHEELS_HEADER_ONLY
#endif

/*  Haswell: AVX2 and FMA. */
#if !defined(FATWHEELS_ARCH_X8664_HASWELL_) || (FATWHEELS_ARCH_X8664_HASWELL_ && !FATWHEELS_ARCH_X8664_)
#undef FATWHEELS_ARCH_X8664_HASWELL_
#if FATWHEELS_ARCH_X8664_ && defined(__AVX2__) && defined(__FMA__)
#define FATWHEELS_ARCH_X8664_HASWELL_ 1
#else
#define FATWHEELS_ARCH_X8664_HASWELL_ 0
#endif
#endif
#if !defined(FATWHEELS_TARGET_HASWELL) || (FATWHEELS_TARGET_HASWELL && !FATWHEELS_ARCH_X8664_)
#undef FATWHEELS_TARGET_HASWELL
#if FATWHEELS_ARCH_X8664_HASWELL_
#define FATWHEELS_TARGET_HASWELL 1
#else
#define FATWHEELS_TARGET_HASWELL 0
#endif
#endif

/*  Skylake: the AVX-512 subsets every server core since Skylake-SP carries. */
#if !defined(FATWHEELS_ARCH_X8664_SKYLAKE_) || (FATWHEELS_ARCH_X8664_SKYLAKE_ && !FATWHEELS_ARCH_X8664_)
#undef FATWHEELS_ARCH_X8664_SKYLAKE_
#if FATWHEELS_ARCH_X8664_ && defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512VL__) && \
    defined(__AVX512BW__) && defined(__AVX512DQ__)
#define FATWHEELS_ARCH_X8664_SKYLAKE_ 1
#else
#define FATWHEELS_ARCH_X8664_SKYLAKE_ 0
#endif
#endif
#if !defined(FATWHEELS_TARGET_SKYLAKE) || (FATWHEELS_TARGET_SKYLAKE && !FATWHEELS_ARCH_X8664_)
#undef FATWHEELS_TARGET_SKYLAKE
#if FATWHEELS_ARCH_X8664_SKYLAKE_
#define FATWHEELS_TARGET_SKYLAKE 1
#else
#define FATWHEELS_TARGET_SKYLAKE 0
#endif
#endif

/*  Sapphire Rapids' AMX: BF16 tile multiplies over Skylake's AVX-512. */
#if !defined(FATWHEELS_ARCH_X8664_SAPPHIREAMX_) || (FATWHEELS_ARCH_X8664_SAPPHIREAMX_ && !FATWHEELS_ARCH_X8664_)
#undef FATWHEELS_ARCH_X8664_SAPPHIREAMX_
#if FATWHEELS_ARCH_X8664_ && defined(__AMX_TILE__) && defined(__AMX_BF16__) && defined(__AVX512F__) && \
    defined(__AVX512BW__)
#define FATWHEELS_ARCH_X8664_SAPPHIREAMX_ 1
#else
#define FATWHEELS_ARCH_X8664_SAPPHIREAMX_ 0
#endif
#endif
#if !defined(FATWHEELS_TARGET_SAPPHIREAMX) || (FATWHEELS_TARGET_SAPPHIREAMX && !FATWHEELS_ARCH_X8664_)
#undef FATWHEELS_TARGET_SAPPHIREAMX
#if FATWHEELS_ARCH_X8664_SAPPHIREAMX_
#define FATWHEELS_TARGET_SAPPHIREAMX 1
#else
#define FATWHEELS_TARGET_SAPPHIREAMX 0
#endif
#endif

/*  SVE: scalable vectors of any length the core picks. */
#if !defined(FATWHEELS_ARCH_ARM64_SVE_) || (FATWHEELS_ARCH_ARM64_SVE_ && !FATWHEELS_ARCH_ARM64_)
#undef FATWHEELS_ARCH_ARM64_SVE_
#if FATWHEELS_ARCH_ARM64_ && defined(__ARM_FEATURE_SVE)
#define FATWHEELS_ARCH_ARM64_SVE_ 1
#else
#define FATWHEELS_ARCH_ARM64_SVE_ 0
#endif
#endif
#if !defined(FATWHEELS_TARGET_SVE) || (FATWHEELS_TARGET_SVE && !FATWHEELS_ARCH_ARM64_)
#undef FATWHEELS_TARGET_SVE
#if FATWHEELS_ARCH_ARM64_SVE_
#define FATWHEELS_TARGET_SVE 1
#else
#define FATWHEELS_TARGET_SVE 0
#endif
#endif

/*  SME: outer products into the ZA tile, in streaming mode, over SVE's loads. */
#if !defined(FATWHEELS_ARCH_ARM64_SME_) || (FATWHEELS_ARCH_ARM64_SME_ && !FATWHEELS_ARCH_ARM64_)
#undef FATWHEELS_ARCH_ARM64_SME_
#if FATWHEELS_ARCH_ARM64_ && defined(__ARM_FEATURE_SME)
#define FATWHEELS_ARCH_ARM64_SME_ 1
#else
#define FATWHEELS_ARCH_ARM64_SME_ 0
#endif
#endif
#if !defined(FATWHEELS_TARGET_SME) || (FATWHEELS_TARGET_SME && !FATWHEELS_ARCH_ARM64_)
#undef FATWHEELS_TARGET_SME
#if FATWHEELS_ARCH_ARM64_SME_
#define FATWHEELS_TARGET_SME 1
#else
#define FATWHEELS_TARGET_SME 0
#endif
#endif

/*  Building blocks follow what is built on them, top-down. Each target above took a literal 0 or 1 from its own
 *  detection, so raising a base's flag here opens its building blocks and never its kernels. */
#if FATWHEELS_ARCH_X8664_SAPPHIREAMX_
#undef FATWHEELS_ARCH_X8664_SKYLAKE_
#define FATWHEELS_ARCH_X8664_SKYLAKE_ 1
#endif
#if FATWHEELS_ARCH_X8664_SKYLAKE_
#undef FATWHEELS_ARCH_X8664_HASWELL_
#define FATWHEELS_ARCH_X8664_HASWELL_ 1
#endif
#if FATWHEELS_ARCH_ARM64_SME_
#undef FATWHEELS_ARCH_ARM64_SVE_
#define FATWHEELS_ARCH_ARM64_SVE_ 1
#endif

#pragma endregion CPU Capabilities

#pragma region GPU Capabilities

/*  Every compiler of a vendor builds every generation's building blocks: they are `static __forceinline__` device
 *  functions, emitted only where a kernel compiled for that generation calls them. */
#if FATWHEELS_ARCH_CUDA_ && defined(__CUDACC__)
#define FATWHEELS_ARCH_CUDA_AMPERE_ 1
#define FATWHEELS_ARCH_CUDA_ADA_ 1
#else
#define FATWHEELS_ARCH_CUDA_AMPERE_ 0
#define FATWHEELS_ARCH_CUDA_ADA_ 0
#endif
#if FATWHEELS_ARCH_ROCM_ && defined(__HIP__)
#define FATWHEELS_ARCH_ROCM_CDNA3_ 1
#else
#define FATWHEELS_ARCH_ROCM_CDNA3_ 0
#endif

/*  A vendor's floor kernels exist wherever its runtime does; a generation's only where the build names it, since a
 *  unit cannot see which codes it compiles for. */
#if !defined(FATWHEELS_TARGET_CUDA) || (FATWHEELS_TARGET_CUDA && !FATWHEELS_ARCH_CUDA_)
#undef FATWHEELS_TARGET_CUDA
#define FATWHEELS_TARGET_CUDA FATWHEELS_ARCH_CUDA_
#endif
#if !defined(FATWHEELS_TARGET_AMPERE) || (FATWHEELS_TARGET_AMPERE && !FATWHEELS_ARCH_CUDA_)
#undef FATWHEELS_TARGET_AMPERE
#define FATWHEELS_TARGET_AMPERE 0
#endif
#if !defined(FATWHEELS_TARGET_ADA) || (FATWHEELS_TARGET_ADA && !FATWHEELS_ARCH_CUDA_)
#undef FATWHEELS_TARGET_ADA
#define FATWHEELS_TARGET_ADA 0
#endif
#if !defined(FATWHEELS_TARGET_ROCM) || (FATWHEELS_TARGET_ROCM && !FATWHEELS_ARCH_ROCM_)
#undef FATWHEELS_TARGET_ROCM
#define FATWHEELS_TARGET_ROCM FATWHEELS_ARCH_ROCM_
#endif
#if !defined(FATWHEELS_TARGET_CDNA3) || (FATWHEELS_TARGET_CDNA3 && !FATWHEELS_ARCH_ROCM_)
#undef FATWHEELS_TARGET_CDNA3
#define FATWHEELS_TARGET_CDNA3 0
#endif

#pragma endregion GPU Capabilities

#if FATWHEELS_ARCH_X8664_HASWELL_
#include <immintrin.h> // `__m256`, `__m512`, the AMX tile intrinsics
#endif
#if FATWHEELS_ARCH_ARM64_SVE_
#include <arm_sve.h> // `svfloat32_t`
#endif
#if FATWHEELS_ARCH_ARM64_SME_
#include <arm_sme.h> // `svmopa_za32_f32_m`
#endif

#if defined(__cplusplus)
extern "C" {
#endif

typedef size_t fw_size_t;
typedef float fw_f32_t;

typedef enum {
    fw_success_k = 0,
    fw_missing_library_k,    // a header-only build has no dispatch
    fw_missing_capability_k, // the kernel is not in this build, or this machine does not run it
    fw_device_failure_k,     // the GPU runtime reported an error
} fw_status_t;

/** A capability's bit position: CPU capabilities first, then each GPU vendor's above the last, ascending by age. */
typedef enum {
    fw_cap_serial_index_k = 0,
    fw_cap_haswell_index_k,
    fw_cap_skylake_index_k,
    fw_cap_sapphireamx_index_k,
    fw_cap_sve_index_k,
    fw_cap_sme_index_k,
    fw_cap_cuda_index_k = 8,
    fw_cap_ampere_index_k,
    fw_cap_ada_index_k,
    fw_cap_rocm_index_k = 12,
    fw_cap_cdna3_index_k,
} fw_capability_index_t;

/** One bit per capability; a mask names several. */
typedef enum {
    fw_cap_serial_k = 1u << fw_cap_serial_index_k,
    fw_cap_haswell_k = 1u << fw_cap_haswell_index_k,
    fw_cap_skylake_k = 1u << fw_cap_skylake_index_k,
    fw_cap_sapphireamx_k = 1u << fw_cap_sapphireamx_index_k,
    fw_cap_sve_k = 1u << fw_cap_sve_index_k,
    fw_cap_sme_k = 1u << fw_cap_sme_index_k,
    fw_cap_cuda_k = 1u << fw_cap_cuda_index_k,
    fw_cap_ampere_k = 1u << fw_cap_ampere_index_k,
    fw_cap_ada_k = 1u << fw_cap_ada_index_k,
    fw_cap_rocm_k = 1u << fw_cap_rocm_index_k,
    fw_cap_cdna3_k = 1u << fw_cap_cdna3_index_k,
} fw_capability_t;

/** `C = A * B` over row-major f32 matrices: A is @p rows by @p depth, B is @p depth by @p columns. GPU kernels take
 *  device pointers and run on @p stream, a `cudaStream_t` or `hipStream_t`; CPU kernels ignore it. */
typedef fw_status_t (*fw_matmul_f32_t)(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                       fw_size_t columns, fw_size_t depth, void *stream);

/** Any kernel, as dispatch tables store them. */
typedef void (*fw_kernel_punned_t)(void);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // FATWHEELS_TYPES_H_
