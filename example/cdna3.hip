/**
 *  @file example/cdna3.hip
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The example's own CDNA3 kernel over FatWheels' public building blocks, built to a code object its host
 * embeds.
 */
#include <fatwheels/cdna3.cuh> // `fw_load_tile_rocm`, `fw_load_fragment_cdna3`, `fw_mma_bf16_cdna3`

/** One 16 by 16 tile of C from a 16 by 32 A and a 32 by 16 B, row-major, rounded to bf16; one wavefront. */
extern "C" __global__ void example_score_cdna3_kernel(float const *a, float const *b, float *c) {
    __shared__ float tile_a[16][33], tile_b[16][33];
    fw_load_tile_rocm(a, 16, 32, 0, 0, &tile_a[0][0], 33, false);
    fw_load_tile_rocm(a, 16, 32, 0, 16, &tile_a[0][16], 33, false);
    fw_load_tile_rocm(b, 32, 16, 0, 0, &tile_b[0][0], 33, true);
    fw_load_tile_rocm(b, 32, 16, 16, 0, &tile_b[0][16], 33, true);
    __syncthreads();
    fw_f32x4_cdna3_t sums = {0, 0, 0, 0};
    sums =
        fw_mma_bf16_cdna3(sums, fw_load_fragment_cdna3(&tile_a[0][0], 33), fw_load_fragment_cdna3(&tile_b[0][0], 33));
    sums =
        fw_mma_bf16_cdna3(sums, fw_load_fragment_cdna3(&tile_a[0][16], 33), fw_load_fragment_cdna3(&tile_b[0][16], 33));
    unsigned const lane = threadIdx.x & 63;
    for (unsigned e = 0; e != 4; ++e) c[(4 * (lane / 16) + e) * 16 + lane % 16] = sums[e];
}
