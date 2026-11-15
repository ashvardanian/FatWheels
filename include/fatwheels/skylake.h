/**
 *  @file include/fatwheels/skylake.h
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Skylake `matmul`: sixteen columns per `vfmadd231ps zmm`, Haswell's blocks for the rest.
 */
#ifndef FATWHEELS_SKYLAKE_H_
#define FATWHEELS_SKYLAKE_H_

#include "fatwheels/types.h"

#if FATWHEELS_ARCH_X8664_
#if FATWHEELS_ARCH_X8664_SKYLAKE_

#include "fatwheels/haswell.h" // `fw_row_f32x8_haswell`, `fw_row_tail_f32_haswell`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma,avx512f,avx512cd,avx512vl,avx512bw,avx512dq"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "fma", "avx512f", "avx512cd", "avx512vl", "avx512bw", "avx512dq")
#endif

/** Building block: sixteen columns of one row of C, starting at column @p j, over the whole depth. */
static inline void fw_row_f32x16_skylake(fw_f32_t const *a_row, fw_f32_t const *b, fw_f32_t *c_row, fw_size_t columns,
                                         fw_size_t depth, fw_size_t j) {
    __m512 sum = _mm512_setzero_ps();
    for (fw_size_t p = 0; p != depth; ++p)
        sum = _mm512_fmadd_ps(_mm512_set1_ps(a_row[p]), _mm512_loadu_ps(b + p * columns + j), sum);
    _mm512_storeu_ps(c_row + j, sum);
}

/** Building block: the first @p count of sixteen floats at @p source, rounded to bf16, the rest zero. */
static inline __m256i fw_load_bf16x16_skylake(fw_f32_t const *source, fw_size_t count) {
    __mmask16 const mask = (__mmask16)(count >= 16 ? 0xFFFFu : (1u << count) - 1u);
    __m512i bits = _mm512_castps_si512(_mm512_maskz_loadu_ps(mask, source));
    __m512i const odd = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
    bits = _mm512_add_epi32(bits, _mm512_add_epi32(odd, _mm512_set1_epi32(0x7FFF))); // to nearest, ties to even
    return _mm512_cvtepi32_epi16(_mm512_srli_epi32(bits, 16));
}

#if FATWHEELS_TARGET_SKYLAKE
FATWHEELS_API fw_status_t fw_matmul_f32_skylake(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                         fw_size_t columns, fw_size_t depth, void *stream) {
    (void)stream;
    for (fw_size_t i = 0; i != rows; ++i) {
        fw_size_t j = 0;
        for (; j + 16 <= columns; j += 16) fw_row_f32x16_skylake(a + i * depth, b, c + i * columns, columns, depth, j);
        for (; j + 8 <= columns; j += 8) fw_row_f32x8_haswell(a + i * depth, b, c + i * columns, columns, depth, j);
        fw_row_tail_f32_haswell(a + i * depth, b, c + i * columns, columns, depth, j);
    }
    return fw_success_k;
}
#endif // FATWHEELS_TARGET_SKYLAKE

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // FATWHEELS_ARCH_X8664_SKYLAKE_
#endif // FATWHEELS_ARCH_X8664_
#endif // FATWHEELS_SKYLAKE_H_
