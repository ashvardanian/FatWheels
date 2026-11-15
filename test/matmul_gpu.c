/**
 *  @file test/matmul_gpu.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Checks every GPU generation the driver loads on device 0, for each vendor this build holds, against
 *      serial, through the vendor's service table and the kernel tables alone.
 */
#include <stdio.h> // `printf`

#define FATWHEELS_TARGET_SERIAL 1 // the reference kernel the library holds
#include "fatwheels/fatwheels.h"

static int check(fw_device_services_t const *services, fw_capability_t capability, int rows, int columns, int depth) {
    static fw_f32_t a[200 * 200], b[200 * 200], expected[200 * 200], got[200 * 200];
    for (int i = 0; i != rows * depth; ++i) a[i] = (fw_f32_t)(i % 7 - 3);
    for (int i = 0; i != depth * columns; ++i) b[i] = (fw_f32_t)(i % 5 - 2);
    fw_matmul_f32_serial(a, b, expected, rows, columns, depth, NULL);
    void *device_a = NULL, *device_b = NULL, *device_c = NULL;
    fw_status_t status = services->allocate(&device_a, sizeof(a));
    if (status == fw_success_k) status = services->allocate(&device_b, sizeof(b));
    if (status == fw_success_k) status = services->allocate(&device_c, sizeof(got));
    if (status == fw_success_k) status = services->copy_to_device(device_a, a, rows * depth * 4, NULL);
    if (status == fw_success_k) status = services->copy_to_device(device_b, b, depth * columns * 4, NULL);
    if (status == fw_success_k)
        status = fw_matmul_f32_best(device_a, device_b, device_c, rows, columns, depth, capability, NULL);
    if (status == fw_success_k) status = services->copy_to_host(got, device_c, rows * columns * 4, NULL);
    if (status == fw_success_k) status = services->synchronize(NULL);
    services->release(device_a), services->release(device_b), services->release(device_c);
    int differs = 0;
    for (int i = 0; i != rows * columns; ++i) differs += got[i] != expected[i];
    printf("capability %#06x %3dx%3dx%3d: status %d, %s serial\n", capability, rows, columns, depth, status,
           differs ? "DIFFERS from" : "matches");
    return differs || status;
}

int main(void) {
    fw_capability_t const vendors[] = {fw_cap_cuda_k, fw_cap_rocm_k};
    int const shapes[][3] = {{37, 41, 67},  {1, 1, 1},   {16, 16, 16}, {33, 7, 100},
                             {128, 96, 64}, {5, 130, 3}, {7, 9, 0}};
    int failures = 0;
    for (int v = 0; v != 2; ++v) {
        fw_device_services_t const *services = fw_device_services(vendors[v]);
        if (!services) continue;
        fw_capability_t const loaded = services->loaded(0);
        printf("loaded on device 0: %#x\n", loaded);
        for (unsigned bit = vendors[v]; bit && bit <= loaded; bit <<= 1) {
            if (!(loaded & bit)) continue;
            failures += !fw_matmul_f32_find((fw_capability_t)bit);
            for (int s = 0; s != 7; ++s)
                failures += check(services, (fw_capability_t)bit, shapes[s][0], shapes[s][1], shapes[s][2]);
        }
    }
    return failures;
}
