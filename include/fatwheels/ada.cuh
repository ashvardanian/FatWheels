/**
 *  @file include/fatwheels/ada.cuh
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Ada `matmul`: one warp per 16 by 16 tile of C, two `mma.sync.m16n8k32` over e4m3, on Ampere's fragments.
 *
 *  Operands round to e4m3 and products accumulate in f32. The FP8 `mma.sync` needs compute capability 8.9 and PTX
 *  ISA 8.4, so only units compiled for 8.9 and later may set this target.
 */
#ifndef FATWHEELS_ADA_CUH_
#define FATWHEELS_ADA_CUH_

#include "fatwheels/types.h"

#if FATWHEELS_ARCH_CUDA_
#if FATWHEELS_ARCH_CUDA_ADA_

#include <cuda_fp8.h> // `__nv_cvt_float2_to_fp8x2`

#include "fatwheels/ampere.cuh" // `fw_load_fragment_ampere`, `fw_store_fragment_ampere`

#if defined(__cplusplus)
extern "C" {
#endif

/** Building block: four f32 values rounded to e4m3, saturating, the first in the lowest byte. */
static __device__ __forceinline__ unsigned fw_pack_e4m3x4_ada(fw_f32_t const *values) {
    unsigned const low = __nv_cvt_float2_to_fp8x2(make_float2(values[0], values[1]), __NV_SATFINITE, __NV_E4M3);
    unsigned const high = __nv_cvt_float2_to_fp8x2(make_float2(values[2], values[3]), __NV_SATFINITE, __NV_E4M3);
    return low | (high << 16);
}

/** Building block: @p c += A * B over a 16 by 32 e4m3 A and a 32 by 8 e4m3 B. */
static __device__ __forceinline__ void fw_mma_e4m3_ada(fw_f32_t *c, unsigned const *a, unsigned b_low,
                                                       unsigned b_high) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_low), "r"(b_high));
}

#if FATWHEELS_TARGET_ADA
/** Ada's own zeroing kernel over Ampere's block, so each generation's unit owns every kernel it launches. */
static __global__ void fw_zero_f32_ada_kernel_(fw_f32_t *c, fw_size_t count) { fw_zero_f32_ampere(c, count); }

static __global__ void fw_matmul_f32_ada_kernel_(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                                 fw_size_t columns, fw_size_t depth) {
    __shared__ fw_f32_t tile_a[16][33], tile_b[16][33]; // `tile_b` holds B's columns as rows, as `.col` reads them
    fw_size_t const first_row = blockIdx.y * 16, first_column = blockIdx.x * 16;
    fw_f32_t sums[2][4] = {{0}};
    for (fw_size_t p = 0; p < depth; p += 32) {
        fw_load_tile_cuda(a, rows, depth, first_row, p, &tile_a[0][0], 33, false);
        fw_load_tile_cuda(a, rows, depth, first_row, p + 16, &tile_a[0][16], 33, false);
        fw_load_tile_cuda(b, depth, columns, p, first_column, &tile_b[0][0], 33, true);
        fw_load_tile_cuda(b, depth, columns, p + 16, first_column, &tile_b[0][16], 33, true);
        __syncthreads();
        fw_f32_t a_values[16], b_values[16];
        fw_load_fragment_ampere(&tile_a[0][0], 33, 4, a_values);
        fw_load_fragment_ampere(&tile_b[0][0], 33, 4, b_values);
        unsigned a_registers[4], b_registers[4];
        for (unsigned r = 0; r != 4; ++r) {
            a_registers[r] = fw_pack_e4m3x4_ada(a_values + 4 * r);
            b_registers[r] = fw_pack_e4m3x4_ada(b_values + 4 * r);
        }
        fw_mma_e4m3_ada(sums[0], a_registers, b_registers[0], b_registers[2]);
        fw_mma_e4m3_ada(sums[1], a_registers, b_registers[1], b_registers[3]);
        __syncthreads();
    }
    fw_store_fragment_ampere(sums[0], c, rows, columns, first_row, first_column);
    fw_store_fragment_ampere(sums[1], c, rows, columns, first_row, first_column + 8);
}

FATWHEELS_API fw_status_t fw_matmul_f32_ada(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                            fw_size_t columns, fw_size_t depth, void *stream) {
    if (!rows || !columns) return fw_success_k;
    dim3 const grid((unsigned)((columns + 15) / 16), (unsigned)((rows + 15) / 16));
    if (depth) fw_matmul_f32_ada_kernel_<<<grid, 32, 0, (cudaStream_t)stream>>>(a, b, c, rows, columns, depth);
    else
        fw_zero_f32_ada_kernel_<<<(unsigned)((rows * columns + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
            c, rows * columns);
    return cudaGetLastError() == cudaSuccess ? fw_success_k : fw_device_failure_k;
}
#endif // FATWHEELS_TARGET_ADA

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // FATWHEELS_ARCH_CUDA_ADA_
#endif // FATWHEELS_ARCH_CUDA_
#endif // FATWHEELS_ADA_CUH_
