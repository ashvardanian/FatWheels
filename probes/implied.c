/**
 *  @file probes/implied.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Whether the floor's own predefines enable a capability, for monoversioned builds; it holds no `main`.
 */
#include <fatwheels/types.h>

#if !FATWHEELS_PROBE_MACRO
#error "The floor does not imply this capability"
#endif

int fatwheels_probe_implied;
