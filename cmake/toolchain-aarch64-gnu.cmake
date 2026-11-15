# AArch64 toolchain for FatWheels, driving Clang against the GNU cross sysroot, as `fatwheels:cpu` carries them.
#
#   cmake -S . -B build_aarch64 -G Ninja --toolchain cmake/toolchain-aarch64-gnu.cmake
#   cmake --build build_aarch64
#   qemu-aarch64 -cpu max,sve-max-vq=4,sme-default-vector-length=64 build_aarch64/fatwheels_test
#
# Optional inputs:
#   -D AARCH64_TRIPLE=aarch64-linux-gnu                                  # target and GNU multiarch directory
#   -D AARCH64_QEMU_CPU=max,sve-max-vq=4,sme-default-vector-length=64    # `-cpu` for the emulator
#
# Binaries link statically, so QEMU runs them with no guest loader prefix.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if (NOT DEFINED AARCH64_TRIPLE)
    set(AARCH64_TRIPLE "aarch64-linux-gnu")
endif ()
if (NOT DEFINED AARCH64_QEMU_CPU)
    set(AARCH64_QEMU_CPU "max,sve-max-vq=4,sme-default-vector-length=64")
endif ()
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES AARCH64_TRIPLE AARCH64_QEMU_CPU)

find_program(fatwheels_clang_ NAMES clang-23 clang REQUIRED)
set(CMAKE_C_COMPILER "${fatwheels_clang_}")
set(CMAKE_C_COMPILER_TARGET "${AARCH64_TRIPLE}")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld -static")

find_program(fatwheels_qemu_aarch64_ qemu-aarch64)
if (fatwheels_qemu_aarch64_)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${fatwheels_qemu_aarch64_};-cpu;${AARCH64_QEMU_CPU}")
endif ()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
