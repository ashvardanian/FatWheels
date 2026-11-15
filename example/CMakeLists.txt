# A higher-level project on FatWheels: taken from source, borrowing its probe verdicts, calling its public kernels
# and building blocks, and naming none of its internal flags.
cmake_minimum_required(VERSION 3.27)
project(fatwheels_example LANGUAGES C)

include(FetchContent)
FetchContent_Declare(fatwheels SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
FetchContent_MakeAvailable(fatwheels)

# One unit per capability, inlining the library's kernel for the capabilities its probes compiled.
add_executable(example_cpu cpu.c serial.c haswell.c sapphireamx.c)
target_link_libraries(example_cpu PRIVATE fatwheels::header fatwheels::cpu_capabilities_compiled)

# Its own Ada kernel over the library's Ada building blocks and none of its kernels, built to a fatbin it embeds.
if (FATWHEELS_CUDA_ARCH)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(example_ada_ OBJECT ada.cu)
    set_target_properties(example_ada_ PROPERTIES CUDA_FATBIN_COMPILATION ON CUDA_ARCHITECTURES 89)
    target_compile_definitions(example_ada_ PRIVATE FATWHEELS_TARGET_CUDA=0)
    target_link_libraries(example_ada_ PRIVATE fatwheels::header)
    set(example_embedded_ ${CMAKE_CURRENT_BINARY_DIR}/embedded)
    add_custom_command(
        OUTPUT ${example_embedded_}/example_ada.fatbin
        COMMAND ${CMAKE_COMMAND} -E make_directory ${example_embedded_}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:example_ada_> ${example_embedded_}/example_ada.fatbin
        DEPENDS example_ada_ $<TARGET_OBJECTS:example_ada_>
        COMMAND_EXPAND_LISTS
    )
    list(APPEND example_images_ ${example_embedded_}/example_ada.fatbin)
endif ()

# Its own CDNA3 kernel over the library's CDNA3 building blocks and none of its kernels, built to a code object it
# embeds.
if (FATWHEELS_ROCM_ARCH)
    enable_language(HIP)
    find_package(hip REQUIRED)
    add_library(example_cdna3_ OBJECT cdna3.hip)
    set_target_properties(example_cdna3_ PROPERTIES HIP_ARCHITECTURES gfx942)
    # Unlike nvcc, Clang leaves device code unoptimized without a build type.
    target_compile_options(example_cdna3_ PRIVATE -O3 --offload-device-only --no-gpu-bundle-output)
    target_compile_definitions(example_cdna3_ PRIVATE FATWHEELS_TARGET_ROCM=0)
    target_link_libraries(example_cdna3_ PRIVATE fatwheels::header)
    set(example_embedded_ ${CMAKE_CURRENT_BINARY_DIR}/embedded)
    add_custom_command(
        OUTPUT ${example_embedded_}/example_cdna3.hsaco
        COMMAND ${CMAKE_COMMAND} -E make_directory ${example_embedded_}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:example_cdna3_> ${example_embedded_}/example_cdna3.hsaco
        DEPENDS example_cdna3_ $<TARGET_OBJECTS:example_cdna3_>
        COMMAND_EXPAND_LISTS
    )
    list(APPEND example_images_ ${example_embedded_}/example_cdna3.hsaco)
endif ()

# One host for every vendor: it embeds whichever images were built and loads them through FatWheels' services.
if (example_images_)
    add_custom_target(example_images_ DEPENDS ${example_images_})
    add_executable(example_gpu gpu.c)
    add_dependencies(example_gpu example_images_)
    set_target_properties(example_gpu PROPERTIES C_STANDARD 23 CUDA_ARCHITECTURES OFF HIP_ARCHITECTURES OFF)
    target_compile_options(example_gpu PRIVATE --embed-dir=${example_embedded_})
    target_link_libraries(example_gpu PRIVATE fatwheels::static)
endif ()
