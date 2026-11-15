/**
 *  @file include/fatwheels/serial.h
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The serial `matmul`, which every CPU runs and every other capability is checked against.
 */
#ifndef FATWHEELS_SERIAL_H_
#define FATWHEELS_SERIAL_H_

#include "fatwheels/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if FATWHEELS_TARGET_SERIAL
FATWHEELS_API fw_status_t fw_matmul_f32_serial(fw_f32_t const *a, fw_f32_t const *b, fw_f32_t *c, fw_size_t rows,
                                        fw_size_t columns, fw_size_t depth, void *stream) {
    (void)stream;
    for (fw_size_t i = 0; i != rows; ++i)
        for (fw_size_t j = 0; j != columns; ++j) {
            fw_f32_t sum = 0;
            for (fw_size_t p = 0; p != depth; ++p) sum += a[i * depth + p] * b[p * columns + j];
            c[i * columns + j] = sum;
        }
    return fw_success_k;
}
#endif // FATWHEELS_TARGET_SERIAL

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // FATWHEELS_SERIAL_H_
