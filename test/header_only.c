/**
 *  @file test/header_only.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief A user with no build system: its own flags decide every capability, as `cc -mavx2 -mfma` would.
 */
#include <stdio.h> // `printf`

#define FATWHEELS_HEADER_ONLY 1
#include <fatwheels/fatwheels.h>

int main(void) {
    fw_f32_t a[4] = {1, 2, 3, 4}, b[4] = {1, 0, 0, 1}, c[4] = {0};
    fw_matmul_f32_serial(a, b, c, 2, 2, 2, NULL);
    printf("compiled here: %#x\n", fw_capabilities_compiled_cpu());
#if FATWHEELS_TARGET_HASWELL
    fw_matmul_f32_haswell(a, b, c, 2, 2, 2, NULL);
#endif
#if FATWHEELS_TARGET_SKYLAKE
    fw_matmul_f32_skylake(a, b, c, 2, 2, 2, NULL);
#endif
#if FATWHEELS_TARGET_SAPPHIREAMX
    fw_matmul_f32_sapphireamx(a, b, c, 2, 2, 2, NULL);
#endif
    return c[3] != 4;
}
