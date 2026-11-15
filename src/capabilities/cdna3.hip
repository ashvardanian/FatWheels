/**
 *  @file src/capabilities/cdna3.hip
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The @c cdna3 kernel, defined once for the FatWheels library.
 */
#undef FATWHEELS_TARGET_ROCM
#define FATWHEELS_TARGET_ROCM 0
#include "fatwheels/fatwheels.h"
#include "fatwheels/cdna3.cuh"

/** The kernel @c fw_capabilities_loaded_rocm asks the runtime to load. */
extern "C" void const *fw_probe_cdna3_(void) { return (void const *)&fw_matmul_f32_cdna3_kernel_; }
