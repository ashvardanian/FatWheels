# One object library per GPU generation, each compiled only for the codes of FATWHEELS_CUDA_ARCH or
# FATWHEELS_ROCM_ARCH that generation runs on, gathered into one static library per vendor. Which generations a
# device runs is the driver's answer, through `fw_capabilities_loaded_cuda` and `fw_capabilities_loaded_rocm`.
#
#   fatwheels_<vendor>_capabilities_compiled_  -> FATWHEELS_ARCH_<VENDOR>_ and FATWHEELS_TARGET_<GENERATION> per unit
#   fatwheels::cuda, fatwheels::rocm           -> the units and the loader query

# Adds `src/capabilities/<unit_>.<extension_>` to `library_`, compiled for `codes_` alone,
# raising its target on `capabilities_`; a unit without codes is left out.
function (fatwheels_add_gpu_unit library_ capabilities_ unit_ extension_ codes_)
    if (NOT codes_)
        return()
    endif ()
    add_library(fatwheels_${unit_}_ OBJECT src/capabilities/${unit_}.${extension_})
    set_target_properties(
        fatwheels_${unit_}_ PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES "${codes_}"
                                       HIP_ARCHITECTURES "${codes_}"
    )
    target_link_libraries(
        fatwheels_${unit_}_ PRIVATE fatwheels_cpu_capabilities_disabled_ fatwheels_header ${capabilities_}
    )
    string(TOUPPER ${unit_} unit_upper_)
    target_compile_definitions(${capabilities_} INTERFACE FATWHEELS_TARGET_${unit_upper_}=1)
    target_sources(${library_} PRIVATE $<TARGET_OBJECTS:fatwheels_${unit_}_>)
    message(STATUS "FatWheels ${unit_}: compiled for ${codes_}")
endfunction ()

# Sets `output_` to the CUDA codes, like `89` or `90a-real`, whose number is at least `minimum_`.
function (fatwheels_select_cuda_codes output_ minimum_)
    set(kept_)
    foreach (code_ IN LISTS FATWHEELS_CUDA_ARCH)
        string(REGEX MATCH "^[0-9]+" number_ "${code_}")
        if (number_ GREATER_EQUAL minimum_)
            list(APPEND kept_ "${code_}")
        endif ()
    endforeach ()
    set(${output_} "${kept_}" PARENT_SCOPE)
endfunction ()

if (FATWHEELS_CUDA_ARCH)
    set(CMAKE_CUDA_ARCHITECTURES "${FATWHEELS_CUDA_ARCH}")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(fatwheels_cuda_capabilities_compiled_ INTERFACE)
    target_compile_definitions(fatwheels_cuda_capabilities_compiled_ INTERFACE FATWHEELS_ARCH_CUDA_=1)
    add_library(fatwheels_cuda STATIC)
    add_library(fatwheels::cuda ALIAS fatwheels_cuda)
    set_target_properties(fatwheels_cuda PROPERTIES LINKER_LANGUAGE CUDA POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(fatwheels_cuda PUBLIC fatwheels_header CUDA::cudart_static)
    # The floor runs everywhere; `mma.sync` over bf16 from 8.0, over e4m3 from 8.9.
    fatwheels_add_gpu_unit(fatwheels_cuda fatwheels_cuda_capabilities_compiled_ cuda cu "${FATWHEELS_CUDA_ARCH}")
    fatwheels_select_cuda_codes(fatwheels_ampere_codes_ 80)
    fatwheels_add_gpu_unit(fatwheels_cuda fatwheels_cuda_capabilities_compiled_ ampere cu "${fatwheels_ampere_codes_}")
    fatwheels_select_cuda_codes(fatwheels_ada_codes_ 89)
    fatwheels_add_gpu_unit(fatwheels_cuda fatwheels_cuda_capabilities_compiled_ ada cu "${fatwheels_ada_codes_}")
    target_link_libraries(fatwheels_static PUBLIC fatwheels_cuda PRIVATE fatwheels_cuda_capabilities_compiled_)
    set_target_properties(fatwheels_static PROPERTIES CUDA_ARCHITECTURES OFF) # its units hold no device code
endif ()

if (FATWHEELS_ROCM_ARCH)
    set(CMAKE_HIP_ARCHITECTURES "${FATWHEELS_ROCM_ARCH}")
    enable_language(HIP)
    find_package(hip REQUIRED)
    add_library(fatwheels_rocm_capabilities_compiled_ INTERFACE)
    target_compile_definitions(fatwheels_rocm_capabilities_compiled_ INTERFACE FATWHEELS_ARCH_ROCM_=1)
    add_library(fatwheels_rocm STATIC)
    add_library(fatwheels::rocm ALIAS fatwheels_rocm)
    set_target_properties(fatwheels_rocm PROPERTIES LINKER_LANGUAGE HIP POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(fatwheels_rocm PUBLIC fatwheels_header hip::host)
    # MFMA over bf16 with the `_1k` layout: MI300 on gfx942, MI350 on gfx950.
    set(fatwheels_cdna3_codes_)
    foreach (code_ IN LISTS FATWHEELS_ROCM_ARCH)
        if (code_ MATCHES "^gfx(942|950)")
            list(APPEND fatwheels_cdna3_codes_ "${code_}")
        endif ()
    endforeach ()
    fatwheels_add_gpu_unit(fatwheels_rocm fatwheels_rocm_capabilities_compiled_ rocm hip "${FATWHEELS_ROCM_ARCH}")
    fatwheels_add_gpu_unit(fatwheels_rocm fatwheels_rocm_capabilities_compiled_ cdna3 hip "${fatwheels_cdna3_codes_}")
    target_link_libraries(fatwheels_static PUBLIC fatwheels_rocm PRIVATE fatwheels_rocm_capabilities_compiled_)
    set_target_properties(fatwheels_static PROPERTIES HIP_ARCHITECTURES OFF)
endif ()

# One GPU module serves every vendor this build holds, through their service tables and the kernel tables.
if (FATWHEELS_BUILD_PYTHON)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
    python_add_library(fatwheels_python_gpu MODULE WITH_SOABI python/gpu.c)
    set_target_properties(fatwheels_python_gpu PROPERTIES OUTPUT_NAME _gpu CUDA_ARCHITECTURES OFF HIP_ARCHITECTURES OFF)
    target_link_libraries(fatwheels_python_gpu PRIVATE fatwheels_static)
    install(TARGETS fatwheels_python_gpu DESTINATION fatwheels COMPONENT fatwheels_python)
endif ()
