/**
 *  @file example/cpu.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Runs every kernel of the example the library's verdicts compiled and this machine runs, against serial.
 */
#include <stdio.h> // `printf`

#define FATWHEELS_HEADER_ONLY 1
#include <fatwheels/fatwheels.h> // `fw_capabilities_detected_cpu`

#include "example.h"

int main(void) {
    fw_f32_t a[16 * 32], b[32 * 16];
    for (int i = 0; i != 16 * 32; ++i) a[i] = (fw_f32_t)(i % 7 - 3), b[i] = (fw_f32_t)(i % 5 - 2);
    fw_capability_t const detected = fw_capabilities_detected_cpu();
    float const expected = example_score_serial(a, b);
    int failures = 0;
#if FATWHEELS_TARGET_HASWELL
    if (detected & fw_cap_haswell_k) failures += example_score_haswell(a, b) != expected, printf("haswell ran\n");
#endif
#if FATWHEELS_TARGET_SAPPHIREAMX
    if (detected & fw_cap_sapphireamx_k)
        failures += example_score_sapphireamx(a, b) != expected, printf("sapphireamx ran\n");
#endif
    (void)detected;
    printf("serial %.0f, %d failures\n", expected, failures);
    return failures;
}
