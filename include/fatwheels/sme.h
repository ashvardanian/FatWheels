/**
 *  @file include/fatwheels/sme.h
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief SME `matmul`: square tiles of C as wide as the streaming vector, from `fmopa` into ZA, fed by SVE's loads.
 *
 *  Each tile of C accumulates in ZA0.S, one outer product per step of depth: a column of A against a row of B. The
 *  columns of A come from rows of A loaded into ZA1.S and read back vertically.
 */
#ifndef FATWHEELS_SME_H_
#define FATWHEELS_SME_H_

#include "fatwheels/types.h"

#if FATWHEELS_ARCH_ARM64_
#if FATWHEELS_ARCH_ARM64_SME_

#include "fatwheels/sve.h" // `fw_load_f32_sve`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv9-a+sme"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv9-a+sme")
#endif

#if FATWHEELS_TARGET_SME
__arm_locally_streaming __arm_new("za") FATWHEELS_API fw_status_t
    fw_matmul_f32_sme(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows, fw_size_t columns,
                      fw_size_t depth, void *stream) {
    (void)stream;
    fw_size_t const tile = svcntw();
    for (fw_size_t i = 0; i < rows; i += tile) {
        svbool_t const row_predicate = svwhilelt_b32_u64(i, rows);
        fw_size_t const tile_rows = rows - i < tile ? rows - i : tile;
        for (fw_size_t j = 0; j < columns; j += tile) {
            svbool_t const column_predicate = svwhilelt_b32_u64(j, columns);
            svzero_za();
            for (fw_size_t p = 0; p < depth; p += tile) {
                svbool_t const depth_predicate = svwhilelt_b32_u64(p, depth);
                fw_size_t const tile_depth = depth - p < tile ? depth - p : tile;
                // streaming mode has no gathers, so A's columns come out of ZA1 vertically
                for (fw_size_t row = 0; row != tile_rows; ++row)
                    svld1_hor_za32(1, row, depth_predicate, a + (i + row) * depth + p);
                for (fw_size_t step = 0; step != tile_depth; ++step) {
                    svfloat32_t const a_column = svread_ver_za32_f32_m(svdup_n_f32(0), row_predicate, 1, step);
                    svfloat32_t const b_row = fw_load_f32_sve(b + (p + step) * columns + j, columns - j);
                    svmopa_za32_f32_m(0, row_predicate, column_predicate, a_column, b_row);
                }
            }
            for (fw_size_t row = 0; row != tile_rows; ++row)
                svst1_hor_za32(0, row, column_predicate, c + (i + row) * columns + j);
        }
    }
    return fw_success_k;
}
#endif // FATWHEELS_TARGET_SME

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // FATWHEELS_ARCH_ARM64_SME_
#endif // FATWHEELS_ARCH_ARM64_
#endif // FATWHEELS_SME_H_
