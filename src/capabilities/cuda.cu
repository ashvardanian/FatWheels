/**
 *  @file src/capabilities/cuda.cu
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The @c cuda kernel, the CUDA loader query and the CUDA runtime services, defined once for the library.
 */
#include "fatwheels/fatwheels.h"
#include "fatwheels/cuda.cuh"

extern "C" void const *fw_probe_ampere_(void);
extern "C" void const *fw_probe_ada_(void);

/** Whether the driver loads @p kernel on the current device: it fails where no code in the binary runs there. */
static bool fw_loads_cuda_(void const *kernel) {
    cudaFuncAttributes attributes;
    bool const loads = cudaFuncGetAttributes(&attributes, kernel) == cudaSuccess;
    cudaGetLastError();
    return loads;
}

static int fw_count_cuda_(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess) return count;
    cudaGetLastError();
    return 0;
}

extern "C" FATWHEELS_API fw_capability_t fw_capabilities_loaded_cuda(int device) {
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) {
        cudaGetLastError();
        return (fw_capability_t)0;
    }
    unsigned loaded = fw_cap_cuda_k * fw_loads_cuda_((void const *)&fw_matmul_f32_cuda_kernel_);
#if FATWHEELS_TARGET_AMPERE
    loaded |= fw_cap_ampere_k * fw_loads_cuda_(fw_probe_ampere_());
#endif
#if FATWHEELS_TARGET_ADA
    loaded |= fw_cap_ada_k * fw_loads_cuda_(fw_probe_ada_());
#endif
    cudaSetDevice(previous);
    return (fw_capability_t)loaded;
}

static fw_status_t fw_status_cuda_(cudaError_t error) {
    return error == cudaSuccess ? fw_success_k : fw_device_failure_k;
}

static fw_status_t fw_allocate_cuda_(void **pointer, fw_size_t bytes) {
    return fw_status_cuda_(cudaMalloc(pointer, bytes));
}
static fw_status_t fw_release_cuda_(void *pointer) { return fw_status_cuda_(cudaFree(pointer)); }
static fw_status_t fw_copy_to_device_cuda_(void *target, void const *source, fw_size_t bytes, void *stream) {
    return fw_status_cuda_(cudaMemcpyAsync(target, source, bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream));
}
static fw_status_t fw_copy_to_host_cuda_(void *target, void const *source, fw_size_t bytes, void *stream) {
    return fw_status_cuda_(cudaMemcpyAsync(target, source, bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream));
}
static fw_status_t fw_synchronize_cuda_(void *stream) {
    return fw_status_cuda_(cudaStreamSynchronize((cudaStream_t)stream));
}
static fw_status_t fw_load_module_cuda_(void **module, void const *image) {
    return fw_status_cuda_(cudaLibraryLoadData((cudaLibrary_t *)module, image, NULL, NULL, 0, NULL, NULL, 0));
}
static fw_status_t fw_unload_module_cuda_(void *module) {
    return fw_status_cuda_(cudaLibraryUnload((cudaLibrary_t)module));
}
static fw_status_t fw_find_kernel_cuda_(void **kernel, void *module, char const *name) {
    return fw_status_cuda_(cudaLibraryGetKernel((cudaKernel_t *)kernel, (cudaLibrary_t)module, name));
}
static fw_status_t fw_launch_cuda_(void *kernel, unsigned const grid[3], unsigned const block[3], void **arguments,
                                   void *stream) {
    return fw_status_cuda_(cudaLaunchKernel((void const *)kernel, dim3(grid[0], grid[1], grid[2]),
                                            dim3(block[0], block[1], block[2]), arguments, 0, (cudaStream_t)stream));
}

extern "C" fw_device_services_t const fw_device_services_cuda_ = {
    fw_count_cuda_,          fw_capabilities_loaded_cuda, fw_allocate_cuda_,    fw_release_cuda_,
    fw_copy_to_device_cuda_, fw_copy_to_host_cuda_,       fw_synchronize_cuda_, fw_load_module_cuda_,
    fw_unload_module_cuda_,  fw_find_kernel_cuda_,        fw_launch_cuda_,
};
