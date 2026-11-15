/**
 *  @file probes/skylake.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Whether the toolchain builds the skylake kernel, the AVX-512 subsets of Skylake-SP, at the unit's floor.
 */
#define FATWHEELS_HEADER_ONLY 1
#define FATWHEELS_ARCH_X8664_SKYLAKE_ 1
#include <fatwheels/types.h>
#include <fatwheels/skylake.h> // `fw_matmul_f32_skylake`

int main(void) {
    fw_f32_t a[4] = {1, 2, 3, 4}, b[4] = {1, 0, 0, 1}, c[4];
    fw_matmul_f32_skylake(a, b, c, 2, 2, 2, NULL);
    return c[3] != 4;
}
