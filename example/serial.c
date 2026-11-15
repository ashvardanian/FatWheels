/**
 *  @file example/serial.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The example's @c serial unit, inlining FatWheels' public serial kernel.
 */
#define FATWHEELS_HEADER_ONLY 1
#include <fatwheels/fatwheels.h>

#include "example.h"

float example_score_serial(fw_f32_t const *a, fw_f32_t const *b) {
    fw_f32_t c[16 * 16];
    fw_matmul_f32_serial(a, b, c, 16, 16, 32, NULL);
    float sum = 0;
    for (int i = 0; i != 16 * 16; ++i) sum += c[i];
    return sum;
}
