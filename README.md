# FatWheels

FatWheels validates the design of hardware-tuned builds, multiversioned and monoversioned alike: one package or binary carrying many SIMD generations side by side, and GPU kernels for several generations and both NVIDIA and AMD in one build.
It is small enough to read in an afternoon, so the build design is what you read:

- __SIMD generations that combine.__
  Each capability builds on the ones below it, reusing their building blocks without duplicating their kernels: Haswell `vfmadd231ps` on 8 lanes, Skylake on 16 over Haswell's edges, Sapphire Rapids AMX `tdpbf16ps` tiles over Skylake's bf16 rounding, and on Arm SVE `fmla` under SME `fmopa` outer products.
- __GPU kernels that come together.__
  One build holds a shared-memory tile for every GPU, Ampere `mma.sync` in bf16, Ada `mma.sync` in e4m3, and AMD's CDNA3 `v_mfma_f32_16x16x16_bf16`, and the driver's loader says which of them a device runs.
- __Both build kinds.__
  A multiversioned build carries every capability the toolchain compiles and picks the best at runtime, for distributing to a diverse fleet.
  A monoversioned one keeps what its `-march` floor implies, for identical machines, with shorter builds and smaller binaries that run only there.
- __Validated without the hardware.__
  `test/conventions.py` turns every rule below into PASS or FAIL, and the [images](#images) carry every tool it needs: Intel SDE for AVX-512 and AMX, QEMU for SVE and SME, CUDA and ROCm for both GPU vendors.

There is one operation, `matmul`, written once per capability with its hardware's own matrix instruction.

## Two Flags

Every capability carries two flags, and every capability header reads them in the same order: the architecture, the capability's `ARCH` flag around its building blocks, then its `TARGET` around its kernels.

- `FATWHEELS_ARCH_<ARCH>_<CAPABILITY>_` says the capability's code compiles in this unit.
  `c/fatwheels/types.h` detects it from the unit's flags, or from compiling as a GPU vendor.
  The library's own build raises it with its probe's verdict, since only compiling tells that a `target` attribute builds what the flags do not enable.
  It never reads a target, and a capability's flag also holds wherever a capability built on it holds, so Skylake's unit sees Haswell's building blocks.
- `FATWHEELS_TARGET_<CAPABILITY>` says this unit has the capability's kernels.
  A CPU target defaults to its capability's own detection, a GPU floor to its vendor, and a GPU generation to 0, since a unit cannot see which codes it compiles for.
  It is public: `0` drops kernels, and `1` shows the declarations of kernels a linked library holds, even to a unit whose flags enable nothing.

A capability's unit, like `src/capabilities/sapphireamx.c`, sets the targets of the capabilities it builds on to 0, so each kernel is defined once.
GPU building blocks are `static __device__ __forceinline__`, so every generation's blocks compile under every code, and a unit that writes the floor target to 0 gets the blocks it calls and no library kernel at all.

## Building

The library builds with CMake, and the Python package with scikit-build-core over it:

```sh
cmake --preset clang-floor && cmake --build --preset clang-floor
pip install .                                    # FATWHEELS_CUDA_ARCH="80;89;120" adds `fatwheels.cuda`
pytest test/matmul.py                            # every capability this machine runs, against serial
```

- `FATWHEELS_MARCH` sets the floor every unit compiles at, like `x86-64-v3`.
- `FATWHEELS_MULTIVERSIONED=OFF` builds monoversioned, keeping only what the floor implies.
- `FATWHEELS_TARGET_<CAPABILITY>=0` drops a capability, and `=1` vouches for one in a monoversioned build.
- `FATWHEELS_CUDA_ARCH` and `FATWHEELS_ROCM_ARCH` list the GPU codes.
  Each generation's unit compiles only for the codes that run it.

```python
import numpy as np
import fatwheels

a, b = np.ones((64, 32), np.float32), np.ones((32, 48), np.float32)
fatwheels.capabilities()                         # ['skylake', 'haswell', 'serial'] on an AVX-512 machine
fatwheels.matmul(a, b)                           # the best capability
fatwheels.matmul(a, b, capability="serial")
if fatwheels.cuda:
    fatwheels.cuda.capabilities()                # ['ada', 'ampere', 'cuda'], as the driver loads them
```

## Images

Three images, built from the `Dockerfile` here, carry every tool the checks need and serve other projects' CI as they are:

| Tag      | Platforms    | Contents                                                                               |
| :------- | :----------- | :------------------------------------------------------------------------------------- |
| `cpu`    | amd64, arm64 | Ubuntu 26.04, Clang 23, GCC 15, CMake, Ninja, static QEMU, GCC 15 sysroots, Intel SDE |
| `verify` | amd64, arm64 | `cpu` plus GenMC v0.19 and Spin, with Clang 22 for GenMC and its clients               |
| `gpu`    | amd64, arm64 | CUDA 13.4, Clang 23, Clang 22 to host NVCC, and on amd64 ROCm 10.0                     |

Each image has one command:

```sh
docker run --rm -v "$PWD:/work" -w /work ghcr.io/ashvardanian/fatwheels:cpu python3 test/conventions.py cpu
docker run --rm -v "$PWD:/work" -w /work ghcr.io/ashvardanian/fatwheels:gpu python3 test/conventions.py gpu
python3 test/conventions.py gpu --run            # on the host, where the NVIDIA driver is
```

The `cpu` checks build x86 natively, run Skylake and Sapphire Rapids under SDE, and run AArch64 under QEMU with SVE and SME on and off.
Intel's licence allows redistributing SDE unmodified, and its notices ship in `/opt/sde/Licenses`.
AMD publishes ROCm for x86-64 alone, so the Arm `gpu` image carries CUDA only.
NVCC's front end rejects libstdc++ 15's `<atomic>` under a Clang 23 host, so `gpu` sets `CUDAHOSTCXX=clang++-22`, and GenMC v0.19 in `verify` does not compile against LLVM 23.
Every pin is an `ARG`: `UBUNTU`, `ROCM` and `LLVM` at the top of the file, and `QEMU_VERSION`, `SDE_VERSION`, `GENMC_COMMIT` and `CUDA` with their checksums in their stages.

## Publishing

The images are built on local machines and pushed by hand, as they change only when a pin does.
Each machine builds its own architecture natively, an x86-64 Linux box for `amd64` and an Apple silicon Mac or an Arm server for `arm64`:

```sh
arch=$(docker version --format '{{.Server.Arch}}') commit=$(git rev-parse --short=7 HEAD)
for target in cpu verify gpu; do
    docker buildx build --target $target --provenance=false --tag ghcr.io/ashvardanian/fatwheels:$target-$commit-$arch --push .
done
```

Without `--provenance=false`, BuildKit wraps each image with an attestation that GHCR lists as an `unknown/unknown` platform.
Once both architectures are up, either machine joins them into the multi-architecture tags that workflows pin:

```sh
for target in cpu verify gpu; do
    docker buildx imagetools create --tag ghcr.io/ashvardanian/fatwheels:$target-$commit \
        --tag ghcr.io/ashvardanian/fatwheels:$target \
        ghcr.io/ashvardanian/fatwheels:$target-$commit-amd64 ghcr.io/ashvardanian/fatwheels:$target-$commit-arm64
done
```
