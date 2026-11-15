/**
 *  @file src/capabilities/sapphireamx.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The @c sapphireamx kernel, defined once for the FatWheels library.
 */
#undef FATWHEELS_TARGET_HASWELL
#define FATWHEELS_TARGET_HASWELL 0
#undef FATWHEELS_TARGET_SKYLAKE
#define FATWHEELS_TARGET_SKYLAKE 0
#include "fatwheels/fatwheels.h"
#include "fatwheels/sapphireamx.h"
