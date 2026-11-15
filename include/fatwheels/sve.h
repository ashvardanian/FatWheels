/**
 *  @file include/fatwheels/sve.h
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief SVE `matmul`: one row of C at a time, a vector of columns per predicated `fmla`, at any vector length.
 */
#ifndef FATWHEELS_SVE_H_
#define FATWHEELS_SVE_H_

#include "fatwheels/types.h"

#if FATWHEELS_ARCH_ARM64_
#if FATWHEELS_ARCH_ARM64_SVE_

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

/** Building block: one row of C, a vector of columns at a time, the last one predicated. */
static inline void fw_row_f32_sve(fw_f32_t const *a_row, fw_f32_t const *b, fw_f32_t *c_row, fw_size_t columns,
                                  fw_size_t depth) {
    for (fw_size_t j = 0; j < columns; j += svcntw()) {
        svbool_t const predicate = svwhilelt_b32_u64(j, columns);
        svfloat32_t sum = svdup_n_f32(0);
        for (fw_size_t p = 0; p != depth; ++p)
            sum = svmla_n_f32_x(predicate, sum, svld1_f32(predicate, b + p * columns + j), a_row[p]);
        svst1_f32(predicate, c_row + j, sum);
    }
}

#if FATWHEELS_TARGET_SVE
FATWHEELS_API fw_status_t fw_matmul_f32_sve(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                     fw_size_t columns, fw_size_t depth, void *stream) {
    (void)stream;
    for (fw_size_t i = 0; i != rows; ++i) fw_row_f32_sve(a + i * depth, b, c + i * columns, columns, depth);
    return fw_success_k;
}
#endif // FATWHEELS_TARGET_SVE

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if FATWHEELS_ARCH_ARM64_SME_
/** Building block: the first @p count floats at @p source, up to a vector of them, the rest zero, in or out of SME's
 *  streaming mode; SVE intrinsics there need the SME extension too, and a call from a function owning ZA would save
 *  ZA around it unless inlined. */
__attribute__((target("arch=armv9-a+sme"), always_inline)) static inline svfloat32_t fw_load_f32_sve(
    fw_f32_t const *source, fw_size_t count) __arm_streaming_compatible {
    return svld1_f32(svwhilelt_b32_u64(0, count), source);
}
#endif // FATWHEELS_ARCH_ARM64_SME_

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // FATWHEELS_ARCH_ARM64_SVE_
#endif // FATWHEELS_ARCH_ARM64_
#endif // FATWHEELS_SVE_H_
