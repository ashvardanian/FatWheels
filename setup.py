"""FatWheels build: `fatwheels._cpu` always, and `_cuda` and `_hip` from one `gpu.cu` where NVCC or ROCm is.

File: setup.py
Author: Ash Vardanian
Date: August 9, 2024
"""

import os
import shutil
import subprocess
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Every listed architecture gets machine code; the newest NVIDIA one also gets PTX for future GPUs.
nvidia_architectures = ["75", "80", "89", "90", "100", "120"]
amd_architectures = ["gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1201"]

nvcc = shutil.which("nvcc")
rocm_clang = "/opt/rocm/llvm/bin/clang++" if os.path.exists("/opt/rocm/llvm/bin/clang++") else None
openmp = [] if sysconfig.get_platform().startswith("macosx") else ["-fopenmp"]
includes = [sysconfig.get_paths()["include"], os.path.dirname(os.path.abspath(__file__))]


class BuildGPU(build_ext):
    """Compiles `fatwheels/gpu.cu` with NVCC or ROCm's Clang straight into an extension module."""

    def build_extension(self, extension):
        if not extension.sources[0].endswith(".cu"):
            return super().build_extension(extension)
        output = self.get_ext_fullpath(extension.name)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        flags = ["-std=c++20", "-O3", *[f"-I{include}" for include in includes]]
        if extension.name.endswith("_cuda"):
            newest = nvidia_architectures[-1]
            codes = [f"-gencode=arch=compute_{code},code=sm_{code}" for code in nvidia_architectures]
            host = os.environ.get("CUDAHOSTCXX", "clang++")
            command = [nvcc, "-ccbin", host, "-x", "cu", "-shared", "-Xcompiler", "-fPIC", *codes]
            command += [f"-gencode=arch=compute_{newest},code=compute_{newest}", *flags]
            command += [extension.sources[0], "-o", output]
        else:
            offload = [f"--offload-arch={code}" for code in amd_architectures]
            command = [rocm_clang, "-x", "hip", "-shared", "-fPIC", *offload, *flags, "-I/opt/rocm/include"]
            command += [extension.sources[0], "-o", output, "-L/opt/rocm/lib", "-lamdhip64"]
        subprocess.run(command, check=True)


extensions = [
    Extension(
        "fatwheels._cpu",
        ["fatwheels/cpu.cpp"],
        include_dirs=includes,
        language="c++",
        extra_compile_args=["-std=c++20", "-O3", *openmp],
        extra_link_args=openmp,
    )
]
if nvcc:
    extensions.append(Extension("fatwheels._cuda", ["fatwheels/gpu.cu"]))
if rocm_clang:
    extensions.append(Extension("fatwheels._hip", ["fatwheels/gpu.cu"]))

setup(ext_modules=extensions, cmdclass={"build_ext": BuildGPU})
