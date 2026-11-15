/**
 *  @file include/fatwheels/cuda.cuh
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief CUDA `matmul` for every NVIDIA GPU: 16 by 16 tiles of C, one thread per value, FMA over shared memory.
 */
#ifndef FATWHEELS_CUDA_CUH_
#define FATWHEELS_CUDA_CUH_

#include "fatwheels/types.h"

/*  Host units of a CUDA build carry the vendor flag too, but only a CUDA compiler parses device code. */
#if FATWHEELS_ARCH_CUDA_ && defined(__CUDACC__)

#include <cuda_runtime.h> // `cudaStream_t`, `cudaGetLastError`

#if defined(__cplusplus)
extern "C" {
#endif

/** Building block: the block loads a 16 by 16 tile of a row-major matrix into shared memory, zeros past its edges,
 *  row by row or, if @p transposed, column by column. */
static __device__ __forceinline__ void fw_load_tile_cuda(fw_f32_t const *source, fw_size_t source_rows,
                                                         fw_size_t source_columns, fw_size_t first_row,
                                                         fw_size_t first_column, fw_f32_t *tile, unsigned stride,
                                                         bool transposed) {
    for (unsigned index = threadIdx.y * blockDim.x + threadIdx.x; index < 256; index += blockDim.x * blockDim.y) {
        unsigned const row = index / 16, column = index % 16;
        bool const inside = first_row + row < source_rows && first_column + column < source_columns;
        fw_f32_t const value = inside ? source[(first_row + row) * source_columns + first_column + column] : 0;
        tile[transposed ? column * stride + row : row * stride + column] = value;
    }
}

#if FATWHEELS_TARGET_CUDA
static __global__ void fw_matmul_f32_cuda_kernel_(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                                  fw_size_t columns, fw_size_t depth) {
    __shared__ fw_f32_t tile_a[16][17], tile_b[16][17];
    fw_size_t const row = blockIdx.y * 16 + threadIdx.y, column = blockIdx.x * 16 + threadIdx.x;
    fw_f32_t sum = 0;
    for (fw_size_t p = 0; p < depth; p += 16) {
        fw_load_tile_cuda(a, rows, depth, blockIdx.y * 16, p, &tile_a[0][0], 17, false);
        fw_load_tile_cuda(b, depth, columns, p, blockIdx.x * 16, &tile_b[0][0], 17, false);
        __syncthreads();
        for (unsigned q = 0; q != 16; ++q) sum = fmaf(tile_a[threadIdx.y][q], tile_b[q][threadIdx.x], sum);
        __syncthreads();
    }
    if (row < rows && column < columns) c[row * columns + column] = sum;
}

FATWHEELS_API fw_status_t fw_matmul_f32_cuda(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                             fw_size_t columns, fw_size_t depth, void *stream) {
    if (!rows || !columns) return fw_success_k;
    dim3 const grid((unsigned)((columns + 15) / 16), (unsigned)((rows + 15) / 16)), block(16, 16);
    fw_matmul_f32_cuda_kernel_<<<grid, block, 0, (cudaStream_t)stream>>>(a, b, c, rows, columns, depth);
    return cudaGetLastError() == cudaSuccess ? fw_success_k : fw_device_failure_k;
}
#endif // FATWHEELS_TARGET_CUDA

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // FATWHEELS_ARCH_CUDA_ && defined(__CUDACC__)
#endif // FATWHEELS_CUDA_CUH_
