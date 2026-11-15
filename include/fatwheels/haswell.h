/**
 *  @file include/fatwheels/haswell.h
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Haswell `matmul`: one row of C at a time, eight columns per `vfmadd231ps`.
 */
#ifndef FATWHEELS_HASWELL_H_
#define FATWHEELS_HASWELL_H_

#include "fatwheels/types.h"

#if FATWHEELS_ARCH_X8664_
#if FATWHEELS_ARCH_X8664_HASWELL_

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "fma")
#endif

/** Building block: eight columns of one row of C, starting at column @p j, over the whole depth. */
static inline void fw_row_f32x8_haswell(fw_f32_t const *a_row, fw_f32_t const *b, fw_f32_t *c_row, fw_size_t columns,
                                        fw_size_t depth, fw_size_t j) {
    __m256 sum = _mm256_setzero_ps();
    for (fw_size_t p = 0; p != depth; ++p)
        sum = _mm256_fmadd_ps(_mm256_set1_ps(a_row[p]), _mm256_loadu_ps(b + p * columns + j), sum);
    _mm256_storeu_ps(c_row + j, sum);
}

/** Building block: the columns from @p j to the end of one row of C, one at a time. */
static inline void fw_row_tail_f32_haswell(fw_f32_t const *a_row, fw_f32_t const *b, fw_f32_t *c_row, fw_size_t columns,
                                           fw_size_t depth, fw_size_t j) {
    for (; j != columns; ++j) {
        fw_f32_t sum = 0;
        for (fw_size_t p = 0; p != depth; ++p) sum += a_row[p] * b[p * columns + j];
        c_row[j] = sum;
    }
}

#if FATWHEELS_TARGET_HASWELL
FATWHEELS_API fw_status_t fw_matmul_f32_haswell(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                         fw_size_t columns, fw_size_t depth, void *stream) {
    (void)stream;
    for (fw_size_t i = 0; i != rows; ++i) {
        fw_size_t j = 0;
        for (; j + 8 <= columns; j += 8) fw_row_f32x8_haswell(a + i * depth, b, c + i * columns, columns, depth, j);
        fw_row_tail_f32_haswell(a + i * depth, b, c + i * columns, columns, depth, j);
    }
    return fw_success_k;
}
#endif // FATWHEELS_TARGET_HASWELL

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // FATWHEELS_ARCH_X8664_HASWELL_
#endif // FATWHEELS_ARCH_X8664_
#endif // FATWHEELS_HASWELL_H_
