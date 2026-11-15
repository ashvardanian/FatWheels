/**
 *  @file include/fatwheels/sapphireamx.h
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Sapphire Rapids AMX `matmul`: 16 by 16 tiles of C from `tdpbf16ps`, fed by Skylake's bf16 rounding.
 *
 *  Each tile of C accumulates 16 by 32 tiles of A against 32 by 16 tiles of B, rounded to bf16, so the result holds
 *  f32 sums of bf16 products. Linux grants the tile state only after `arch_prctl(ARCH_REQ_XCOMP_PERM)`, which
 *  @ref fw_capabilities_detected_cpu asks for before it reports this capability.
 */
#ifndef FATWHEELS_SAPPHIREAMX_H_
#define FATWHEELS_SAPPHIREAMX_H_

#include "fatwheels/types.h"

#if FATWHEELS_ARCH_X8664_
#if FATWHEELS_ARCH_X8664_SAPPHIREAMX_

#include "fatwheels/skylake.h" // `fw_load_bf16x16_skylake`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                          \
    __attribute__((target("avx2,fma,avx512f,avx512cd,avx512vl,avx512bw,avx512dq,amx-tile,amx-bf16"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "fma", "avx512f", "avx512cd", "avx512vl", "avx512bw", "avx512dq", "amx-tile", "amx-bf16")
#endif

/** Building block: configures tiles 0 to 2 as 16 rows of 64 bytes, C in f32 and A and B in bf16. */
static inline void fw_configure_tiles_sapphireamx(void) {
    _Alignas(64) unsigned char config[64] = {0};
    config[0] = 1; // palette
    for (int tile = 0; tile != 3; ++tile) {
        config[16 + 2 * tile] = 64; // bytes per row
        config[48 + tile] = 16;     // rows
    }
    _tile_loadconfig(config);
}

/** Building block: rows @p p and @p p + 1 of B from column @p j, in the pairs `tdpbf16ps` multiplies. */
static inline void fw_pack_pairs_sapphireamx(fw_f32_t const *b, fw_size_t columns, fw_size_t depth, fw_size_t p,
                                             fw_size_t j, void *pairs) {
    fw_size_t const count = columns - j < 16 ? columns - j : 16;
    __m256i const even = p < depth ? fw_load_bf16x16_skylake(b + p * columns + j, count) : _mm256_setzero_si256();
    __m256i const odd =
        p + 1 < depth ? fw_load_bf16x16_skylake(b + (p + 1) * columns + j, count) : _mm256_setzero_si256();
    __m512i const joined = _mm512_or_si512(_mm512_cvtepu16_epi32(even), //
                                           _mm512_slli_epi32(_mm512_cvtepu16_epi32(odd), 16));
    _mm512_storeu_si512(pairs, joined);
}

#if FATWHEELS_TARGET_SAPPHIREAMX
FATWHEELS_API fw_status_t fw_matmul_f32_sapphireamx(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                             fw_size_t columns, fw_size_t depth, void *stream) {
    (void)stream;
    _Alignas(64) unsigned short tile_a[16][32];
    _Alignas(64) unsigned short tile_b[16][32];
    _Alignas(64) fw_f32_t tile_c[16][16];
    fw_configure_tiles_sapphireamx();
    for (fw_size_t i = 0; i < rows; i += 16)
        for (fw_size_t j = 0; j < columns; j += 16) {
            fw_size_t const tile_rows = rows - i < 16 ? rows - i : 16;
            fw_size_t const tile_columns = columns - j < 16 ? columns - j : 16;
            _tile_zero(0);
            for (fw_size_t p = 0; p < depth; p += 32) {
                fw_size_t const tile_depth = depth - p < 32 ? depth - p : 32;
                for (fw_size_t row = 0; row != 16; ++row) {
                    __m256i low = _mm256_setzero_si256(), high = _mm256_setzero_si256();
                    if (row < tile_rows) {
                        low = fw_load_bf16x16_skylake(a + (i + row) * depth + p, tile_depth);
                        if (tile_depth > 16)
                            high = fw_load_bf16x16_skylake(a + (i + row) * depth + p + 16, tile_depth - 16);
                    }
                    _mm256_store_si256((__m256i *)&tile_a[row][0], low);
                    _mm256_store_si256((__m256i *)&tile_a[row][16], high);
                }
                for (fw_size_t pair = 0; pair != 16; ++pair)
                    fw_pack_pairs_sapphireamx(b, columns, p + tile_depth, p + 2 * pair, j, tile_b[pair]);
                _tile_loadd(1, tile_a, 64);
                _tile_loadd(2, tile_b, 64);
                _tile_dpbf16ps(0, 1, 2);
            }
            _tile_stored(0, tile_c, 64);
            for (fw_size_t row = 0; row != tile_rows; ++row)
                for (fw_size_t column = 0; column != tile_columns; ++column)
                    c[(i + row) * columns + j + column] = tile_c[row][column];
        }
    _tile_release();
    return fw_success_k;
}
#endif // FATWHEELS_TARGET_SAPPHIREAMX

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // FATWHEELS_ARCH_X8664_SAPPHIREAMX_
#endif // FATWHEELS_ARCH_X8664_
#endif // FATWHEELS_SAPPHIREAMX_H_
