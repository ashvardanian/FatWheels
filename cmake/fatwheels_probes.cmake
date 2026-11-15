# Each CPU capability gets two verdicts, and the verdict target carries both onto every unit that links it:
#
#   compiles  -> FATWHEELS_ARCH_<ARCH>_<CAPABILITY>_  the toolchain builds the capability's code at the unit's floor
#   compiled  -> FATWHEELS_TARGET_<CAPABILITY>        it compiles, nobody set the target to 0, and the build kind keeps it
#
# A probe raises only its own ARCH flag, as a unit at the floor would see it from the verdict target.
add_library(fatwheels_cpu_capabilities_compiled INTERFACE)
add_library(fatwheels::cpu_capabilities_compiled ALIAS fatwheels_cpu_capabilities_compiled)
target_compile_definitions(
    fatwheels_cpu_capabilities_compiled INTERFACE FATWHEELS_MULTIVERSIONED=$<BOOL:${FATWHEELS_MULTIVERSIONED}>
)
# GPU units: every CPU ARCH flag 0, so no CPU intrinsics header reaches a device compiler; targets follow by default.
add_library(fatwheels_cpu_capabilities_disabled_ INTERFACE)

function (fatwheels_add_cpu_capability capability_name_ target_macro_ arch_macro_)
    set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
    if (NOT DEFINED fatwheels_${capability_name_}_compiles)
        try_compile(
            fatwheels_${capability_name_}_compiles ${CMAKE_BINARY_DIR}/fatwheels_probes
            ${PROJECT_SOURCE_DIR}/probes/${capability_name_}.c
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_SOURCE_DIR}/c;${PROJECT_SOURCE_DIR}/include" C_STANDARD 11
            COMPILE_DEFINITIONS ${fatwheels_unit_arch_flags_}
        )
    endif ()
    set(compiled_ ${fatwheels_${capability_name_}_compiles})
    if (DEFINED ${target_macro_} AND NOT ${target_macro_})
        set(compiled_ FALSE)
    endif ()
    if (compiled_ AND NOT FATWHEELS_MULTIVERSIONED)
        # A monoversioned build keeps what the floor's predefines imply, or what `-D <target>=1` vouches for.
        if (NOT DEFINED fatwheels_${capability_name_}_implied)
            set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
            try_compile(
                fatwheels_${capability_name_}_implied ${CMAKE_BINARY_DIR}/fatwheels_probes
                ${PROJECT_SOURCE_DIR}/probes/implied.c
                CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_SOURCE_DIR}/c;${PROJECT_SOURCE_DIR}/include" C_STANDARD 11
                COMPILE_DEFINITIONS ${fatwheels_unit_arch_flags_} -DFATWHEELS_PROBE_MACRO=${target_macro_}
            )
        endif ()
        if (NOT fatwheels_${capability_name_}_implied AND NOT ${target_macro_})
            set(compiled_ FALSE)
        endif ()
    endif ()
    message(STATUS "FatWheels ${capability_name_}: compiles ${fatwheels_${capability_name_}_compiles}, compiled ${compiled_}")
    set(fatwheels_${capability_name_}_compiled ${compiled_} CACHE INTERNAL "Whether ${target_macro_} is in this build")
    if (compiled_)
        set_property(GLOBAL APPEND PROPERTY fatwheels_capabilities_compiled ${capability_name_})
    endif ()
    target_compile_definitions(
        fatwheels_cpu_capabilities_compiled INTERFACE "${arch_macro_}=$<BOOL:${fatwheels_${capability_name_}_compiles}>"
                                                      "${target_macro_}=$<BOOL:${compiled_}>"
    )
    target_compile_definitions(fatwheels_cpu_capabilities_disabled_ INTERFACE "${arch_macro_}=0")
endfunction ()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    fatwheels_add_cpu_capability(haswell FATWHEELS_TARGET_HASWELL FATWHEELS_ARCH_X8664_HASWELL_)
    fatwheels_add_cpu_capability(skylake FATWHEELS_TARGET_SKYLAKE FATWHEELS_ARCH_X8664_SKYLAKE_)
    fatwheels_add_cpu_capability(sapphireamx FATWHEELS_TARGET_SAPPHIREAMX FATWHEELS_ARCH_X8664_SAPPHIREAMX_)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    fatwheels_add_cpu_capability(sve FATWHEELS_TARGET_SVE FATWHEELS_ARCH_ARM64_SVE_)
    fatwheels_add_cpu_capability(sme FATWHEELS_TARGET_SME FATWHEELS_ARCH_ARM64_SME_)
endif ()
