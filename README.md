# FatWheels

FatWheels validates the hard part of shipping hardware-friendly software: one package that carries multi-versioned CPU kernels and both NVIDIA and AMD GPU code, and runs what each machine supports.
It is a workflow and toolchain to copy into such projects, small enough to read in an afternoon:

- __One package, every CPU.__ Each CPU kernel is written once and compiled per capability, AVX2 on x86, SVE on Arm and RVV on RISC-V, and the best one is picked when the module loads.
- __One source, both GPU vendors.__ `fatwheels/gpu.cu` builds with NVCC for NVIDIA and with ROCm's Clang for AMD, and a vendor without a runtime or a device reads as `None`.
- __Validated where it runs.__ CI checks every capability natively and under QEMU at several vector lengths, builds both vendors' modules, and confirms the CPU module links no GPU runtime.
- __Reusable images.__ The `Dockerfile` pins Clang 23, static QEMU, GenMC and Spin, CUDA and ROCm into images other projects' CI pulls as they are.

The build is one `setup.py`, with no CMake.
The two operations, array accumulation and matrix multiplication, are deliberately simple, so the packaging is what you read.
Their baseline Python and Numba implementations are in `baseline.py`.

## Layout

- `fatwheels/cpu.hpp` holds the CPU kernels: each is a generic template, and each capability is a one-line wrapper with a `target` attribute, under which the compiler vectorizes that same template.
- `fatwheels/cpu.cpp` binds them as `fatwheels._cpu`, which every machine imports.
- `fatwheels/gpu.cu` holds the GPU kernels and binds them as `fatwheels._cuda` or `fatwheels._hip`, depending on which compiler builds it.
- `fatwheels/__init__.py` imports the GPU modules lazily, so a machine without ROCm sees `fatwheels.hip` as `None`.
- `test.py` checks every kernel against the baseline, `bench.py` times them, and `test.cpp` checks the CPU kernels without Python, as QEMU runs them.

```python
import numpy as np
import fatwheels

data = np.random.rand(1 << 20).astype(np.float32)
fatwheels.capabilities()                  # ['haswell', 'serial'] on a recent x86 CPU
fatwheels.reduce(data)                    # the best capability
fatwheels.reduce(data, capability="serial")
if fatwheels.cuda:
    fatwheels.cuda.reduce(data)
if fatwheels.hip:
    fatwheels.hip.reduce(data)
```

## Installation

I'd recommend forking the repository for your own projects, but you can also clone it directly:

```bash
git clone https://github.com/ashvardanian/FatWheels.git
cd FatWheels
```

Once pulled down, you can build and run the project with `uv`:

```bash
uv pip install -e .[test]                   # builds `_cuda` and `_hip` too, where NVCC or ROCm are found
uv run pytest test.py -s -x                 # test until first failure
uv run pytest --benchmark-enable bench.py   # time every kernel
```

On Linux, Clang needs OpenMP's development package, like `libomp-dev`, and NVCC picks its host compiler from `CUDAHOSTCXX`.

## Images

CI runs in three images built from the `Dockerfile` here, which also serve other projects' CI.
Each pins the compilers, emulators and model checkers that workflows otherwise install or build on every run.

| Tag      | Platforms    | Contents                                                                         |
| :------- | :----------- | :------------------------------------------------------------------------------- |
| `cpu`    | amd64, arm64 | Ubuntu 26.04, Clang 23, CMake, Ninja, static QEMU user-mode, GCC 15 sysroots     |
| `verify` | amd64, arm64 | `cpu` plus GenMC v0.19 and Spin, with Clang 22 for GenMC and its clients         |
| `gpu`    | amd64, arm64 | CUDA 13.4 with cuBLAS, Clang 23, Clang 22 to host NVCC, and on amd64 ROCm 10.0   |

They publish as `ghcr.io/ashvardanian/fatwheels:<tag>`, and every tag also exists as `<tag>-<commit>`, which workflows should pin.
QEMU binaries answer to both `qemu-<arch>` and `qemu-<arch>-static`, and the GCC 15 sysroots cover AArch64, x86-64, RISC-V 64, LoongArch and POWER little-endian, minus the native one.
`GENMC` and `GENMC_CLANG` are already set in `verify`.

Each CI leg runs locally in its image:

```sh
docker buildx build --target cpu --build-arg JOBS=8 --tag fatwheels:cpu .
docker run --rm -v "$PWD:/work" -w /work fatwheels:cpu sh -c \
    'clang++-23 --target=aarch64-linux-gnu -std=c++20 -O2 -fopenmp-simd -static -fuse-ld=lld -I. test.cpp -o test &&
     qemu-aarch64 -cpu max,sve-max-vq=4 ./test'
```

`JOBS` bounds the QEMU and GenMC compiles, which default to every core.
On x86-64 the `gpu` target pulls AMD's 9 GB ROCm image and keeps only what HIP compilation needs, about 1 GB of it.
AMD publishes ROCm for x86-64 alone, so the Arm `gpu` image, the one Grace Hopper builds use, carries CUDA only.

Each pin is an `ARG`: `UBUNTU`, `ROCM` and `LLVM` at the top of the file, and `QEMU_VERSION` with its `QEMU_SHA256`, `GENMC_COMMIT` and `CUDA` in their stages.
Every image compiles with Clang 23, and `gpu` also carries ROCm's own build of the same major.
Two tools still need Clang 22 beside it: GenMC v0.19 in `verify`, which does not compile against LLVM 23's pass manager, and NVCC 13.4 in `gpu`.
NVCC's front end rejects libstdc++ 15's `<atomic>` under a Clang 23 host even with `-allow-unsupported-compiler`, so `gpu` sets `CUDAHOSTCXX=clang++-22`.

## Publishing

The images are built on local machines and pushed by hand, as they change only when a pin does.
Each machine builds its own architecture natively, an x86-64 Linux box for `amd64` and an Apple silicon Mac or an Arm server for `arm64`:

```sh
arch=$(docker version --format '{{.Server.Arch}}') commit=$(git rev-parse --short=7 HEAD)
for target in cpu verify gpu; do
    docker buildx build --target $target --tag ghcr.io/ashvardanian/fatwheels:$target-$commit-$arch --push .
done
```

Once both architectures are up, either machine joins them into the multi-architecture tags that workflows pin:

```sh
for target in cpu verify gpu; do
    docker buildx imagetools create --tag ghcr.io/ashvardanian/fatwheels:$target-$commit \
        --tag ghcr.io/ashvardanian/fatwheels:$target \
        ghcr.io/ashvardanian/fatwheels:$target-$commit-amd64 ghcr.io/ashvardanian/fatwheels:$target-$commit-arm64
done
```

## Reading Materials

Beginner GPGPU:

- High-level concepts: [nvidia.com](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- Nvidia CuPy UDFs: [cupy.dev](https://docs.cupy.dev/en/stable/user_guide/kernel.html)
- CUDA in Python with Numba: [numba/nvidia-cuda-tutorial](https://github.com/numba/nvidia-cuda-tutorial)
- C++ STL Parallelism on GPUs: [nvidia.com](https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/)

Advanced GPGPU:

- CUDA math intrinsics: [nvidia.com](https://docs.nvidia.com/cuda/cuda-math-api/index.html)
- Troubleshooting Nvidia hardware: [stas00/ml-engineering](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/nvidia/debug.md)
- Nvidia ISA Generator with SM89 and SM90 codes: [kuterd/nv_isa_solver](https://github.com/kuterd/nv_isa_solver)
- Multi GPU examples: [nvidia/multi-gpu-programming-models](https://github.com/NVIDIA/multi-gpu-programming-models)

Communities:

- CUDA MODE on [Discord](https://discord.com/invite/cudamode)
- r/CUDA on [Reddit](https://www.reddit.com/r/CUDA/)
- NVIDIA Developer Forums on [DevTalk](https://forums.developer.nvidia.com)
