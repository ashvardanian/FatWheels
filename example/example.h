/**
 *  @file example/example.h
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The example's own kernels, one unit per capability, declared under the library's public targets.
 */
#ifndef FATWHEELS_EXAMPLE_H_
#define FATWHEELS_EXAMPLE_H_

#include <fatwheels/types.h>

/** Sums `A * B`, a score the example computes per capability, with A of 16 by 32 and B of 32 by 16. */
float example_score_serial(fw_f32_t const *a, fw_f32_t const *b);
#if FATWHEELS_TARGET_HASWELL
float example_score_haswell(fw_f32_t const *a, fw_f32_t const *b);
#endif
#if FATWHEELS_TARGET_SAPPHIREAMX
float example_score_sapphireamx(fw_f32_t const *a, fw_f32_t const *b);
#endif

#endif // FATWHEELS_EXAMPLE_H_
