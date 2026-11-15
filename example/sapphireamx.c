/**
 *  @file example/sapphireamx.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The example's @c sapphireamx unit, inlining FatWheels' public sapphireamx kernel.
 */
#define FATWHEELS_HEADER_ONLY 1
#include <fatwheels/fatwheels.h>

#include "example.h"

#if FATWHEELS_TARGET_SAPPHIREAMX
float example_score_sapphireamx(fw_f32_t const *a, fw_f32_t const *b) {
    fw_f32_t c[16 * 16];
    fw_matmul_f32_sapphireamx(a, b, c, 16, 16, 32, NULL);
    float sum = 0;
    for (int i = 0; i != 16 * 16; ++i) sum += c[i];
    return sum;
}
#endif // FATWHEELS_TARGET_SAPPHIREAMX
