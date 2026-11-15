/**
 *  @file include/fatwheels/cdna3.cuh
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief CDNA3 `matmul`: one wavefront per 16 by 16 tile of C, `v_mfma_f32_16x16x16_bf16` on ROCm's tiles.
 *
 *  Operands round to bf16 and products accumulate in f32, so the result holds f32 sums of bf16 products. Runs on
 *  MI300's gfx942 and MI350's gfx950, whose wavefronts are 64 lanes wide.
 */
#ifndef FATWHEELS_CDNA3_CUH_
#define FATWHEELS_CDNA3_CUH_

#include "fatwheels/types.h"

#if FATWHEELS_ARCH_ROCM_
#if FATWHEELS_ARCH_ROCM_CDNA3_

#include "fatwheels/rocm.cuh" // `fw_load_tile_rocm`

#if defined(__cplusplus)
extern "C" {
#endif

typedef short fw_bf16x4_cdna3_t __attribute__((ext_vector_type(4)));
typedef float fw_f32x4_cdna3_t __attribute__((ext_vector_type(4)));

/** Building block: one lane's four MFMA operands from a 16-row tile in shared memory, rounded to bf16, to nearest
 *  with ties to even: row `lane % 16`, columns from `4 * (lane / 16)`. A tile of B's columns gives B's operands. */
static __device__ __forceinline__ fw_bf16x4_cdna3_t fw_load_fragment_cdna3(fw_f32_t const *tile, unsigned stride) {
    unsigned const lane = threadIdx.x & 63;
    fw_bf16x4_cdna3_t fragment;
#pragma unroll
    for (unsigned e = 0; e != 4; ++e) {
        unsigned const bits = __float_as_uint(tile[(lane % 16) * stride + 4 * (lane / 16) + e]);
        fragment[e] = (short)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
    }
    return fragment;
}

/** Building block: @p c + A * B over a 16 by 16 bf16 A and a 16 by 16 bf16 B. */
static __device__ __forceinline__ fw_f32x4_cdna3_t fw_mma_bf16_cdna3(fw_f32x4_cdna3_t c, fw_bf16x4_cdna3_t a,
                                                                     fw_bf16x4_cdna3_t b) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

#if FATWHEELS_TARGET_CDNA3
static __global__ void fw_matmul_f32_cdna3_kernel_(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                                   fw_size_t columns, fw_size_t depth) {
    __shared__ fw_f32_t tile_a[16][17], tile_b[16][17]; // `tile_b` holds B's columns as rows
    fw_size_t const first_row = blockIdx.y * 16, first_column = blockIdx.x * 16;
    fw_f32x4_cdna3_t sums = {0, 0, 0, 0};
    for (fw_size_t p = 0; p < depth; p += 16) {
        fw_load_tile_rocm(a, rows, depth, first_row, p, &tile_a[0][0], 17, false);
        fw_load_tile_rocm(b, depth, columns, p, first_column, &tile_b[0][0], 17, true);
        __syncthreads();
        sums = fw_mma_bf16_cdna3(sums, fw_load_fragment_cdna3(&tile_a[0][0], 17),
                                 fw_load_fragment_cdna3(&tile_b[0][0], 17));
        __syncthreads();
    }
    unsigned const lane = threadIdx.x & 63;
    for (unsigned e = 0; e != 4; ++e) {
        fw_size_t const row = first_row + 4 * (lane / 16) + e, column = first_column + lane % 16;
        if (row < rows && column < columns) c[row * columns + column] = sums[e];
    }
}

FATWHEELS_API fw_status_t fw_matmul_f32_cdna3(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                              fw_size_t columns, fw_size_t depth, void *stream) {
    if (!rows || !columns) return fw_success_k;
    dim3 const grid((unsigned)((columns + 15) / 16), (unsigned)((rows + 15) / 16));
    fw_matmul_f32_cdna3_kernel_<<<grid, 64, 0, (hipStream_t)stream>>>(a, b, c, rows, columns, depth);
    return hipGetLastError() == hipSuccess ? fw_success_k : fw_device_failure_k;
}
#endif // FATWHEELS_TARGET_CDNA3

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // FATWHEELS_ARCH_ROCM_CDNA3_
#endif // FATWHEELS_ARCH_ROCM_
#endif // FATWHEELS_CDNA3_CUH_
