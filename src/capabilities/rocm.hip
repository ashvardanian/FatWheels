/**
 *  @file src/capabilities/rocm.hip
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The @c rocm kernel, the ROCm loader query and the ROCm runtime services, defined once for the library.
 */
#include "fatwheels/fatwheels.h"
#include "fatwheels/rocm.cuh"

extern "C" void const *fw_probe_cdna3_(void);

/** Whether the runtime loads @p kernel on the current device: it fails where no code object runs there. */
static bool fw_loads_rocm_(void const *kernel) {
    hipFuncAttributes attributes;
    bool const loads = hipFuncGetAttributes(&attributes, kernel) == hipSuccess;
    (void)hipGetLastError();
    return loads;
}

extern "C" FATWHEELS_API fw_capability_t fw_capabilities_loaded_rocm(int device) {
    int previous = 0;
    if (hipGetDevice(&previous) != hipSuccess || hipSetDevice(device) != hipSuccess) {
        (void)hipGetLastError();
        return (fw_capability_t)0;
    }
    unsigned loaded = fw_cap_rocm_k * fw_loads_rocm_((void const *)&fw_matmul_f32_rocm_kernel_);
#if FATWHEELS_TARGET_CDNA3
    loaded |= fw_cap_cdna3_k * fw_loads_rocm_(fw_probe_cdna3_());
#endif
    (void)hipSetDevice(previous);
    return (fw_capability_t)loaded;
}

static fw_status_t fw_status_rocm_(hipError_t error) {
    return error == hipSuccess ? fw_success_k : fw_device_failure_k;
}

static fw_status_t fw_allocate_rocm_(void **pointer, fw_size_t bytes) {
    return fw_status_rocm_(hipMalloc(pointer, bytes));
}
static fw_status_t fw_release_rocm_(void *pointer) { return fw_status_rocm_(hipFree(pointer)); }
static fw_status_t fw_copy_to_device_rocm_(void *target, void const *source, fw_size_t bytes, void *stream) {
    return fw_status_rocm_(hipMemcpyAsync(target, source, bytes, hipMemcpyHostToDevice, (hipStream_t)stream));
}
static fw_status_t fw_copy_to_host_rocm_(void *target, void const *source, fw_size_t bytes, void *stream) {
    return fw_status_rocm_(hipMemcpyAsync(target, source, bytes, hipMemcpyDeviceToHost, (hipStream_t)stream));
}
static fw_status_t fw_synchronize_rocm_(void *stream) {
    return fw_status_rocm_(hipStreamSynchronize((hipStream_t)stream));
}
static fw_status_t fw_load_module_rocm_(void **module, void const *image) {
    return fw_status_rocm_(hipModuleLoadData((hipModule_t *)module, image));
}
static fw_status_t fw_unload_module_rocm_(void *module) {
    return fw_status_rocm_(hipModuleUnload((hipModule_t)module));
}
static fw_status_t fw_find_kernel_rocm_(void **kernel, void *module, char const *name) {
    return fw_status_rocm_(hipModuleGetFunction((hipFunction_t *)kernel, (hipModule_t)module, name));
}
static fw_status_t fw_launch_rocm_(void *kernel, unsigned const grid[3], unsigned const block[3], void **arguments,
                                   void *stream) {
    return fw_status_rocm_(hipModuleLaunchKernel((hipFunction_t)kernel, grid[0], grid[1], grid[2], block[0], block[1],
                                                 block[2], 0, (hipStream_t)stream, arguments, NULL));
}

extern "C" fw_device_services_t const fw_device_services_rocm_ = {
    fw_capabilities_loaded_rocm, fw_allocate_rocm_,    fw_release_rocm_,     fw_copy_to_device_rocm_,
    fw_copy_to_host_rocm_,       fw_synchronize_rocm_, fw_load_module_rocm_, fw_unload_module_rocm_,
    fw_find_kernel_rocm_,        fw_launch_rocm_,
};
