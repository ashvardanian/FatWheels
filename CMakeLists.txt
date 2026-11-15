# FatWheels: a small, reproducible environment for packing as many hardware generations as possible into one build.
# Every CPU capability and GPU generation is a unit of its own, gated by two flags per capability, and the probes,
# verdict targets and presets below are the parts a larger project copies.
#
#   -D FATWHEELS_MARCH=<floor>                 (default: the compiler's own floor; `x86-64-v3`, `native`, ...)
#   -D FATWHEELS_MULTIVERSIONED=ON|OFF         (default: ON, every capability the toolchain compiles)
#   -D FATWHEELS_TARGET_<CAPABILITY>=0         (drop a capability's kernels; `=1` vouches for one when monoversioned)
#   -D FATWHEELS_CUDA_ARCH=<codes>             (default: empty for no CUDA, or a list like `80;89;120`)
#   -D FATWHEELS_ROCM_ARCH=<targets>           (default: empty for no ROCm, or a list like `gfx942`)
cmake_minimum_required(VERSION 3.24)
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/VERSION fatwheels_version_)
project(fatwheels VERSION ${fatwheels_version_} LANGUAGES C)

option(FATWHEELS_BUILD_PYTHON "Compile the CPython modules over the static libraries" OFF)
option(FATWHEELS_BUILD_TEST "Compile the C test that checks every capability against serial" ${PROJECT_IS_TOP_LEVEL})
option(FATWHEELS_MULTIVERSIONED "Keep every capability above the floor, or only the best one FATWHEELS_MARCH implies" ON)
set(FATWHEELS_CUDA_ARCH "" CACHE STRING "NVIDIA codes the library builds for, like 80;89;120")
set(FATWHEELS_ROCM_ARCH "" CACHE STRING "AMD targets the library builds for, like gfx942")
set(FATWHEELS_MARCH "$ENV{FATWHEELS_MARCH}" CACHE STRING "The `-march` floor, like x86-64-v3 or native")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
if (FATWHEELS_MARCH)
    set(fatwheels_unit_arch_flags_ -march=${FATWHEELS_MARCH})
endif ()

# Headers: declarations under `c/`, definitions and building blocks under `include/`.
add_library(fatwheels_header INTERFACE)
add_library(fatwheels::header ALIAS fatwheels_header)
target_include_directories(
    fatwheels_header INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c> $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

include(cmake/fatwheels_probes.cmake)

# One unit per capability, each defining only its own kernels; the dispatch unit holds the table.
get_property(fatwheels_capabilities_ GLOBAL PROPERTY fatwheels_capabilities_compiled)
set(fatwheels_sources_ src/fatwheels.c src/capabilities/serial.c)
foreach (capability_ IN LISTS fatwheels_capabilities_)
    list(APPEND fatwheels_sources_ src/capabilities/${capability_}.c)
endforeach ()
add_library(fatwheels_static STATIC ${fatwheels_sources_})
add_library(fatwheels::static ALIAS fatwheels_static)
target_compile_options(fatwheels_static PRIVATE ${fatwheels_unit_arch_flags_} -Wall)
target_compile_definitions(fatwheels_static PUBLIC FATWHEELS_MULTIVERSIONED=$<BOOL:${FATWHEELS_MULTIVERSIONED}>)
target_link_libraries(fatwheels_static PUBLIC fatwheels_header PRIVATE fatwheels_cpu_capabilities_compiled)
set_target_properties(fatwheels_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (FATWHEELS_CUDA_ARCH OR FATWHEELS_ROCM_ARCH)
    include(cmake/fatwheels_gpu.cmake)
endif ()

if (FATWHEELS_BUILD_TEST)
    add_executable(fatwheels_test test/matmul.c)
    target_compile_options(fatwheels_test PRIVATE ${fatwheels_unit_arch_flags_})
    target_link_libraries(fatwheels_test PRIVATE fatwheels_static)
endif ()
if (FATWHEELS_BUILD_TEST AND fatwheels_haswell_compiled)
    # A unit whose flags enable nothing, seeing the declaration of a kernel the library holds.
    add_executable(fatwheels_test_declarations test/declarations.c)
    target_compile_definitions(fatwheels_test_declarations PRIVATE FATWHEELS_TARGET_HASWELL=1)
    target_link_libraries(fatwheels_test_declarations PRIVATE fatwheels_static)
endif ()

if (FATWHEELS_BUILD_TEST AND (FATWHEELS_CUDA_ARCH OR FATWHEELS_ROCM_ARCH))
    add_executable(fatwheels_test_gpu test/matmul_gpu.c)
    set_target_properties(fatwheels_test_gpu PROPERTIES CUDA_ARCHITECTURES OFF HIP_ARCHITECTURES OFF)
    target_link_libraries(fatwheels_test_gpu PRIVATE fatwheels_static)
endif ()

if (FATWHEELS_BUILD_PYTHON)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
    python_add_library(fatwheels_python MODULE WITH_SOABI python/cpu.c)
    set_target_properties(fatwheels_python PROPERTIES OUTPUT_NAME _cpu)
    target_compile_options(fatwheels_python PRIVATE ${fatwheels_unit_arch_flags_})
    target_link_libraries(fatwheels_python PRIVATE fatwheels_static)
    install(TARGETS fatwheels_python DESTINATION fatwheels COMPONENT fatwheels_python)
endif ()
