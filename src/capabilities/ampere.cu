/**
 *  @file src/capabilities/ampere.cu
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The @c ampere kernel, defined once for the FatWheels library.
 */
#undef FATWHEELS_TARGET_CUDA
#define FATWHEELS_TARGET_CUDA 0
#include "fatwheels/fatwheels.h"
#include "fatwheels/ampere.cuh"

/** The kernel @c fw_capabilities_loaded_cuda asks the driver to load. */
extern "C" void const *fw_probe_ampere_(void) { return (void const *)&fw_matmul_f32_ampere_kernel_; }
