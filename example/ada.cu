/**
 *  @file example/ada.cu
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The example's own Ada kernel over FatWheels' public building blocks, built to a fatbin its host embeds.
 */
#include <fatwheels/ada.cuh> // `fw_load_tile_cuda`, `fw_load_fragment_ampere`, `fw_mma_e4m3_ada`

/** One 16 by 16 tile of C from a 16 by 32 A and a 32 by 16 B, row-major, rounded to e4m3; one warp. */
extern "C" __global__ void example_score_ada_kernel(float const *a, float const *b, float *c) {
    __shared__ float tile_a[16][33], tile_b[16][33];
    fw_load_tile_cuda(a, 16, 32, 0, 0, &tile_a[0][0], 33, false);
    fw_load_tile_cuda(a, 16, 32, 0, 16, &tile_a[0][16], 33, false);
    fw_load_tile_cuda(b, 32, 16, 0, 0, &tile_b[0][0], 33, true);
    fw_load_tile_cuda(b, 32, 16, 16, 0, &tile_b[0][16], 33, true);
    __syncthreads();
    float a_values[16], b_values[16];
    fw_load_fragment_ampere(&tile_a[0][0], 33, 4, a_values);
    fw_load_fragment_ampere(&tile_b[0][0], 33, 4, b_values);
    unsigned a_registers[4], b_registers[4];
    for (unsigned r = 0; r != 4; ++r) {
        a_registers[r] = fw_pack_e4m3x4_ada(a_values + 4 * r);
        b_registers[r] = fw_pack_e4m3x4_ada(b_values + 4 * r);
    }
    float sums[2][4] = {{0}};
    fw_mma_e4m3_ada(sums[0], a_registers, b_registers[0], b_registers[2]);
    fw_mma_e4m3_ada(sums[1], a_registers, b_registers[1], b_registers[3]);
    fw_store_fragment_ampere(sums[0], c, 16, 16, 0, 0);
    fw_store_fragment_ampere(sums[1], c, 16, 16, 0, 8);
}
