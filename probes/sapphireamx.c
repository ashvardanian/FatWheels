/**
 *  @file probes/sapphireamx.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Whether the toolchain builds the sapphireamx kernel, AMX tiles in BF16, at the unit's floor.
 */
#define FATWHEELS_HEADER_ONLY 1
#define FATWHEELS_ARCH_X8664_SAPPHIREAMX_ 1
#include <fatwheels/types.h>
#include <fatwheels/sapphireamx.h> // `fw_matmul_f32_sapphireamx`

int main(void) {
    fw_f32_t a[4] = {1, 2, 3, 4}, b[4] = {1, 0, 0, 1}, c[4];
    fw_matmul_f32_sapphireamx(a, b, c, 2, 2, 2, NULL);
    return c[3] != 4;
}
