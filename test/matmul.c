/**
 *  @file test/matmul.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Checks the CPU dispatch tables, and every kernel they hold for this machine against serial, as QEMU and
 *      SDE run it.
 *
 *  Small whole numbers keep every product and sum exact in f32, bf16 and e4m3 alike, so any difference is a kernel
 *  bug, not rounding.
 */
#include <stdio.h> // `printf`

#define FATWHEELS_TARGET_SERIAL 1 // the reference kernel the library holds
#include "fatwheels/fatwheels.h"

int main(void) {
    enum { rows = 37, columns = 41, depth = 67 }; // primes, so every tile shape leaves a partial edge
    static fw_f32_t a[rows * depth], b[depth * columns], expected[rows * columns], got[rows * columns];
    for (int i = 0; i != rows * depth; ++i) a[i] = (fw_f32_t)(i % 7 - 3);
    for (int i = 0; i != depth * columns; ++i) b[i] = (fw_f32_t)(i % 5 - 2);
    fw_matmul_f32_serial(a, b, expected, rows, columns, depth, NULL);

    fw_capability_t const enabled = fw_capabilities_enabled_cpu();
    fw_capability_t const cpu = (fw_capability_t)(fw_cap_cuda_k - 1);
    unsigned held = 0, best = 0;
    int failures = 0;
    for (unsigned bit = 1; bit != fw_cap_cuda_k; bit <<= 1) {
        fw_matmul_f32_t const kernel = fw_matmul_f32_find((fw_capability_t)bit);
        failures += kernel && !(enabled & bit); // the table never hands out a kernel this machine cannot run
        if (!kernel) continue;
        held |= bit, best = bit;
        int differs = 0;
        kernel(a, b, got, rows, columns, depth, NULL);
        for (int i = 0; i != rows * columns; ++i) differs += got[i] != expected[i];
        printf("capability %#x: %s serial\n", bit, differs ? "differs from" : "matches");
        failures += differs != 0;
    }
    // Any mask picks the best capability it shares with the table, without a branch per capability.
    failures += fw_matmul_f32_find(cpu) != fw_matmul_f32_find((fw_capability_t)best);
    failures +=
        fw_matmul_f32_best(a, b, got, rows, columns, depth, (fw_capability_t)0, NULL) != fw_missing_capability_k;
    // A monoversioned build holds one kernel per operation.
    failures += !FATWHEELS_MULTIVERSIONED && (held & (held - 1)) != 0;
    printf("compiled %#x, enabled %#x, dispatched %#x\n", fw_capabilities_compiled_cpu(), enabled, held);
    return failures;
}
