/**
 *  @file include/fatwheels/ampere.cuh
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Ampere `matmul`: one warp per 16 by 16 tile of C, two `mma.sync.m16n8k16` over bf16, on CUDA's tiles.
 *
 *  Operands round to bf16 and products accumulate in f32, so the result holds f32 sums of bf16 products. Runs on
 *  compute capability 8.0 and every GPU since.
 */
#ifndef FATWHEELS_AMPERE_CUH_
#define FATWHEELS_AMPERE_CUH_

#include "fatwheels/types.h"

#if FATWHEELS_ARCH_CUDA_
#if FATWHEELS_ARCH_CUDA_AMPERE_

#include "fatwheels/cuda.cuh" // `fw_load_tile_cuda`

#if defined(__cplusplus)
extern "C" {
#endif

/** Building block: one lane's four `mma.sync` A registers, as f32, from a 16-row tile in shared memory, @p pack
 *  values per register: 2 for 16-bit operands, 4 for 8-bit ones. A tile of B's columns gives two B fragments. */
static __device__ __forceinline__ void fw_load_fragment_ampere(fw_f32_t const *tile, unsigned stride, unsigned pack,
                                                               fw_f32_t *values) {
    unsigned const lane = threadIdx.x & 31, group = lane >> 2, member = lane & 3;
#pragma unroll
    for (unsigned r = 0; r != 4; ++r)
#pragma unroll
        for (unsigned e = 0; e != pack; ++e)
            values[r * pack + e] = tile[(group + 8 * (r & 1)) * stride + member * pack + 4 * pack * (r >> 1) + e];
}

/** Building block: two f32 values rounded to bf16, to nearest with ties to even, the first in the low half. */
static __device__ __forceinline__ unsigned fw_pack_bf16x2_ampere(fw_f32_t low, fw_f32_t high) {
    unsigned const low_bits = __float_as_uint(low), high_bits = __float_as_uint(high);
    unsigned const low_rounded = (low_bits + 0x7FFFu + ((low_bits >> 16) & 1u)) >> 16;
    unsigned const high_rounded = (high_bits + 0x7FFFu + ((high_bits >> 16) & 1u)) >> 16;
    return low_rounded | (high_rounded << 16);
}

/** Building block: @p c += A * B over a 16 by 16 bf16 A and a 16 by 8 bf16 B. */
static __device__ __forceinline__ void fw_mma_bf16_ampere(fw_f32_t *c, unsigned const *a, unsigned b_low,
                                                          unsigned b_high) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, " "{%8, %9}, "
                                                                                                            "{%0, %1, " "%2, %3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_low), "r"(b_high));
}

/** Building block: one lane's four `mma.sync` accumulators into the 16 by 8 block of C at @p first_row and
 *  @p first_column, skipping values past its edges. */
static __device__ __forceinline__ void fw_store_fragment_ampere(fw_f32_t const *sums, fw_f32_t *c, fw_size_t rows,
                                                                fw_size_t columns, fw_size_t first_row,
                                                                fw_size_t first_column) {
    unsigned const lane = threadIdx.x & 31;
#pragma unroll
    for (unsigned r = 0; r != 4; ++r) {
        fw_size_t const row = first_row + (lane >> 2) + 8 * (r >> 1), column = first_column + (lane & 3) * 2 + (r & 1);
        if (row < rows && column < columns) c[row * columns + column] = sums[r];
    }
}

#if FATWHEELS_TARGET_AMPERE || FATWHEELS_TARGET_ADA
/** Without depth there is nothing to multiply, so Ampere and Ada launch this instead of their `mma.sync` kernels. */
static __global__ void fw_zero_f32_ampere_kernel_(fw_f32_t *c, fw_size_t count) {
    fw_size_t const index = (fw_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) c[index] = 0;
}
#endif // FATWHEELS_TARGET_AMPERE || FATWHEELS_TARGET_ADA

#if FATWHEELS_TARGET_AMPERE
/** The B registers of each 8-column half of a tile, as `fw_load_fragment_ampere` orders them. */
static __device__ unsigned const fw_halves_ampere_[2][2] = {{0, 2}, {1, 3}};

static __global__ void fw_matmul_f32_ampere_kernel_(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                                    fw_size_t columns, fw_size_t depth) {
    __shared__ fw_f32_t tile_a[16][17], tile_b[16][17]; // `tile_b` holds B's columns as rows, as `.col` reads them
    fw_size_t const first_row = blockIdx.y * 16, first_column = blockIdx.x * 16;
    fw_f32_t sums[2][4] = {{0}};
    for (fw_size_t p = 0; p < depth; p += 16) {
        fw_load_tile_cuda(a, rows, depth, first_row, p, &tile_a[0][0], 17, false);
        fw_load_tile_cuda(b, depth, columns, p, first_column, &tile_b[0][0], 17, true);
        __syncthreads();
        fw_f32_t a_values[8], b_values[8];
        fw_load_fragment_ampere(&tile_a[0][0], 17, 2, a_values);
        fw_load_fragment_ampere(&tile_b[0][0], 17, 2, b_values);
        unsigned a_registers[4], b_registers[4];
        for (unsigned r = 0; r != 4; ++r) {
            a_registers[r] = fw_pack_bf16x2_ampere(a_values[2 * r], a_values[2 * r + 1]);
            b_registers[r] = fw_pack_bf16x2_ampere(b_values[2 * r], b_values[2 * r + 1]);
        }
        for (unsigned half = 0; half != 2; ++half)
            fw_mma_bf16_ampere(sums[half], a_registers, b_registers[fw_halves_ampere_[half][0]],
                               b_registers[fw_halves_ampere_[half][1]]);
        __syncthreads();
    }
    fw_store_fragment_ampere(sums[0], c, rows, columns, first_row, first_column);
    fw_store_fragment_ampere(sums[1], c, rows, columns, first_row, first_column + 8);
}

FATWHEELS_API fw_status_t fw_matmul_f32_ampere(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                               fw_size_t columns, fw_size_t depth, void *stream) {
    if (!rows || !columns) return fw_success_k;
    dim3 const grid((unsigned)((columns + 15) / 16), (unsigned)((rows + 15) / 16));
    if (depth) fw_matmul_f32_ampere_kernel_<<<grid, 32, 0, (cudaStream_t)stream>>>(a, b, c, rows, columns, depth);
    else
        fw_zero_f32_ampere_kernel_<<<(unsigned)((rows * columns + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
            c, rows * columns);
    return cudaGetLastError() == cudaSuccess ? fw_success_k : fw_device_failure_k;
}
#endif // FATWHEELS_TARGET_AMPERE

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // FATWHEELS_ARCH_CUDA_AMPERE_
#endif // FATWHEELS_ARCH_CUDA_
#endif // FATWHEELS_AMPERE_CUH_
