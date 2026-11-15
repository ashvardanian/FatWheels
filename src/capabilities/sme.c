/**
 *  @file src/capabilities/sme.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The @c sme kernel, defined once for the FatWheels library.
 */
#undef FATWHEELS_TARGET_SVE
#define FATWHEELS_TARGET_SVE 0
#include "fatwheels/fatwheels.h"
#include "fatwheels/sme.h"
