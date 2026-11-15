/**
 *  @file example/haswell.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The example's @c haswell unit, inlining FatWheels' public haswell kernel.
 */
#define FATWHEELS_HEADER_ONLY 1
#include <fatwheels/fatwheels.h>

#include "example.h"

#if FATWHEELS_TARGET_HASWELL
float example_score_haswell(fw_f32_t const *a, fw_f32_t const *b) {
    fw_f32_t c[16 * 16];
    fw_matmul_f32_haswell(a, b, c, 16, 16, 32, NULL);
    float sum = 0;
    for (int i = 0; i != 16 * 16; ++i) sum += c[i];
    return sum;
}
#endif // FATWHEELS_TARGET_HASWELL
