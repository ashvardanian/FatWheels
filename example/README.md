# Example

This folder is a higher-level project built on FatWheels, the way an application builds on a library of hardware-specific kernels.
It shows what such a project may reuse, and `test/conventions.py` checks that it reuses nothing else:

- __Probe verdicts.__
  It takes FatWheels from source and links `fatwheels::cpu_capabilities_compiled`, so it never probes the toolchain itself and sees exactly the capabilities the library compiled.
- __Public kernels, one unit per capability.__
  `haswell.c` and `sapphireamx.c` inline the library's kernels under its public `FATWHEELS_TARGET_<CAPABILITY>` flags, and each object carries its own capability's kernel and no other.
- __GPU building blocks without GPU kernels.__
  `ada.cu` writes `FATWHEELS_TARGET_CUDA=0` and calls Ada's `mma.sync` block from its own kernel, and `cdna3.hip` writes `FATWHEELS_TARGET_ROCM=0` and calls CDNA3's MFMA block.
  Each image holds its one kernel and no library kernel or table.
- __Device code it loads itself, through one host.__
  `gpu.c` embeds whichever images were built and loads each through FatWheels' service table for its vendor, so it names no CUDA or HIP function.

It names no flag or function with a trailing `_`, which FatWheels keeps for itself.

```sh
cmake -S example -B build_example -G Ninja -D FATWHEELS_CUDA_ARCH="80;89;120"
cmake --build build_example && build_example/example_cpu && build_example/example_gpu
```
