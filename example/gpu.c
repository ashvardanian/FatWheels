/**
 *  @file example/gpu.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Loads the example's own device code, whichever vendors it was built for, through FatWheels' device
 *      services, and checks it against serial.
 *
 *  A product ships its kernels as images it loads itself, so this unit names no vendor API: the library's service
 *  table for each vendor moves the memory and loads the image the build embedded for it.
 */
#include <stdio.h> // `printf`

#define FATWHEELS_TARGET_SERIAL 1 // the reference kernel the library holds
#include <fatwheels/fatwheels.h>

// clang-format off
#if __has_embed(<example_ada.fatbin>)
_Alignas(64) static unsigned char const example_ada_image_[] = { // the drivers read images in 8-byte words
#embed <example_ada.fatbin>
};
#endif
#if __has_embed(<example_cdna3.hsaco>)
_Alignas(64) static unsigned char const example_cdna3_image_[] = {
#embed <example_cdna3.hsaco>
};
#endif
// clang-format on

/** One embedded image: the vendor whose services load it, its kernel, and the block it runs as. */
static struct {
    fw_capability_t vendor;
    void const *image;
    char const *kernel;
    unsigned block;
} const images[] = {
#if __has_embed(<example_ada.fatbin>)
    {fw_cap_cuda_k, example_ada_image_, "example_score_ada_kernel", 32},
#endif
#if __has_embed(<example_cdna3.hsaco>)
    {fw_cap_rocm_k, example_cdna3_image_, "example_score_cdna3_kernel", 64},
#endif
};

int main(void) {
    static float a[16 * 32], b[32 * 16], expected[16 * 16], got[16 * 16];
    for (int i = 0; i != 16 * 32; ++i) a[i] = (float)(i % 7 - 3), b[i] = (float)(i % 5 - 2);
    fw_matmul_f32_serial(a, b, expected, 16, 16, 32, NULL);
    int failures = 0;
    for (size_t i = 0; i != sizeof(images) / sizeof(images[0]); ++i) {
        fw_device_services_t const *services = fw_device_services(images[i].vendor);
        if (!services || !services->loaded(0)) {
            printf("%s: no device, skipped\n", images[i].kernel);
            continue;
        }
        void *module = NULL, *kernel = NULL, *device_a = NULL, *device_b = NULL, *device_c = NULL;
        void *arguments[] = {&device_a, &device_b, &device_c};
        unsigned const grid[3] = {1, 1, 1}, block[3] = {images[i].block, 1, 1};
        fw_status_t status = services->load_module(&module, images[i].image);
        if (status == fw_success_k) status = services->find_kernel(&kernel, module, images[i].kernel);
        if (status == fw_success_k) status = services->allocate(&device_a, sizeof(a));
        if (status == fw_success_k) status = services->allocate(&device_b, sizeof(b));
        if (status == fw_success_k) status = services->allocate(&device_c, sizeof(got));
        if (status == fw_success_k) status = services->copy_to_device(device_a, a, sizeof(a), NULL);
        if (status == fw_success_k) status = services->copy_to_device(device_b, b, sizeof(b), NULL);
        if (status == fw_success_k) status = services->launch(kernel, grid, block, arguments, NULL);
        if (status == fw_success_k) status = services->copy_to_host(got, device_c, sizeof(got), NULL);
        if (status == fw_success_k) status = services->synchronize(NULL);
        services->release(device_a), services->release(device_b), services->release(device_c);
        if (module) services->unload_module(module);
        int differs = status != fw_success_k;
        for (int j = 0; j != 16 * 16; ++j) differs += got[j] != expected[j];
        printf("%s: status %d, %s serial\n", images[i].kernel, status, differs ? "differs from" : "matches");
        failures += differs != 0;
    }
    return failures;
}
