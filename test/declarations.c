/**
 *  @file test/declarations.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief A unit whose flags enable no capability, calling the Haswell kernel the linked library holds.
 */
#include <stdio.h> // `printf`

#include <fatwheels/fatwheels.h> // `fw_matmul_f32_haswell`, declared because the build set its target to 1

int main(void) {
    fw_f32_t a[4] = {1, 2, 3, 4}, b[4] = {1, 0, 0, 1}, c[4] = {0};
    if (!(fw_capabilities_detected_cpu() & fw_cap_haswell_k)) return printf("no Haswell here\n"), 0;
    fw_matmul_f32_haswell(a, b, c, 2, 2, 2, NULL);
    printf("Haswell from the library: %.0f %.0f %.0f %.0f\n", c[0], c[1], c[2], c[3]);
    return c[3] != 4;
}
