# syntax=docker/dockerfile:1

# Toolchain images for building and validating code that ships multi-versioned CPU kernels and
# NVIDIA and AMD GPU code in one package. FatWheels' CI runs in them, and so can any other project's.
#
# - `cpu`: Clang 23, static QEMU user-mode and GCC 15 sysroots, to cross-compile every CPU
#   capability and run it under emulation.
# - `verify`: `cpu` plus GenMC and Spin, to model-check concurrent code.
# - `gpu`: CUDA 13.4, plus ROCm 10.0 on x86-64, to compile NVIDIA and AMD device code side by side.
#
# Each publishes as `ghcr.io/ashvardanian/fatwheels:<target>-<commit>`, and every pin is an `ARG`.

ARG UBUNTU=ubuntu:26.04
# ROCm 10.0's Clang is LLVM 23 too, matching `LLVM`; 10.1 moved to 24.
ARG ROCM=rocm/dev-ubuntu-26.04:10.0.0-full
ARG LLVM=23
ARG TARGETARCH

# Installs the given packages of one LLVM major version from apt.llvm.org, by default Clang, LLD, the LLVM tools and
# the sanitizer runtimes.
FROM scratch AS scripts
COPY --chmod=755 <<'EOF' /install-llvm
#!/bin/sh
set -eu
. /etc/os-release
version=$1
shift
[ $# -gt 0 ] || set -- "clang-$version" "lld-$version" "llvm-$version" "libclang-rt-$version-dev"
keyring=/usr/share/keyrings/llvm.gpg
[ -f "$keyring" ] || curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o "$keyring"
echo "deb [signed-by=$keyring] http://apt.llvm.org/$VERSION_CODENAME/ llvm-toolchain-$VERSION_CODENAME-$version main" \
    >"/etc/apt/sources.list.d/llvm-$version.list"
apt-get update
apt-get install -y --no-install-recommends "$@"
EOF
# Keeps out what no build or check runs: GCC's LTO dumpers, LLVM's benchmark generator, and cuBLAS's 1 GB of static
# archives, which NVCC never links unless asked.
COPY <<'EOF' /dpkg-excludes
path-exclude=/usr/bin/*lto-dump-*
path-exclude=/usr/lib/llvm-*/bin/llvm-exegesis
path-exclude=/usr/local/cuda-*/targets/*/lib/libcublas*_static.a
EOF

FROM ${UBUNTU} AS qemu
ARG QEMU_VERSION=11.1.2
ARG QEMU_SHA256=731b5681e4bb18be313231579b8efd0296c5b015fa36dc533874b639ba838016
ARG JOBS=
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl xz-utils build-essential ninja-build pkg-config python3 python3-venv flex bison \
        libglib2.0-dev libpcre2-dev zlib1g-dev
# Distribution qemu-user lags the extensions dispatch probes for, like `Zvbb` and `Zvfh` in `riscv_hwprobe`.
RUN curl -fsSLo /tmp/qemu.tar.xz "https://download.qemu.org/qemu-${QEMU_VERSION}.tar.xz" && \
    echo "${QEMU_SHA256}  /tmp/qemu.tar.xz" | sha256sum -c - && \
    tar -xJf /tmp/qemu.tar.xz -C /tmp && \
    cd "/tmp/qemu-${QEMU_VERSION}" && \
    ./configure --prefix=/opt/qemu --static --disable-system --disable-tools --disable-docs -Dstrip=true \
        --target-list=aarch64-linux-user,x86_64-linux-user,riscv64-linux-user,loongarch64-linux-user,ppc64le-linux-user && \
    make -j"${JOBS:-$(nproc)}" install

FROM ${UBUNTU} AS cpu
ARG LLVM
# Every stage inherits Ubuntu's own title, description and version labels unless it sets its own.
LABEL org.opencontainers.image.source=https://github.com/ashvardanian/FatWheels \
      org.opencontainers.image.title="FatWheels CPU" \
      org.opencontainers.image.description="Clang 23, QEMU and cross sysroots to build and emulate CPU capabilities" \
      org.opencontainers.image.version=""
ENV DEBIAN_FRONTEND=noninteractive
COPY --from=scripts /install-llvm /usr/local/bin/install-llvm
COPY --from=scripts /dpkg-excludes /etc/dpkg/dpkg.cfg.d/fatwheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git cmake ninja-build make python3 xz-utils binutils && \
    install-llvm "${LLVM}" && \
    apt-get install -y --no-install-recommends "libomp-${LLVM}-dev" && \
    rm -rf /var/lib/apt/lists/*
# Clang cross-compiles against these sysroots: GCC 15 is the first to scope LASX and POWER9 per function.
RUN native=$(uname -m | tr _ -) && \
    for triple in aarch64 x86-64 riscv64 loongarch64 powerpc64le; do \
        [ "$triple" = "$native" ] || packages="$packages g++-15-$triple-linux-gnu"; \
    done && \
    apt-get update && apt-get install -y --no-install-recommends $packages && \
    rm -rf /var/lib/apt/lists/*
COPY --from=qemu /opt/qemu/bin/ /usr/local/bin/
RUN for emulator in /usr/local/bin/qemu-*; do ln -s "${emulator##*/}" "${emulator}-static"; done

# GenMC v0.19 supports LLVM 19 to 22, and its passes lack the `printPipeline` LLVM 23's pass manager requires, so the
# model checker and its clients compile with Clang 22 until a GenMC release supports 23.
FROM cpu AS genmc
ARG GENMC_COMMIT=9f6c4c0772d0c42b325681581acc6a0f3eb9b5f7
ARG JOBS=
RUN install-llvm 22 clang-22 llvm-22-dev && \
    apt-get install -y --no-install-recommends libffi-dev zlib1g-dev libedit-dev
RUN git init -q /tmp/genmc && \
    git -C /tmp/genmc fetch -q --depth 1 https://github.com/MPI-SWS/genmc "${GENMC_COMMIT}" && \
    git -C /tmp/genmc checkout -q FETCH_HEAD && \
    cmake -S /tmp/genmc -B /tmp/genmc/build -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/opt/genmc \
        -D CMAKE_C_COMPILER=clang-22 -D CMAKE_CXX_COMPILER=clang++-22 -D LLVM_DIR=/usr/lib/llvm-22/lib/cmake/llvm && \
    cmake --build /tmp/genmc/build --parallel ${JOBS:-$(nproc)} && \
    cmake --install /tmp/genmc/build --strip

FROM cpu AS verify
LABEL org.opencontainers.image.title="FatWheels Verify" \
      org.opencontainers.image.description="The CPU image plus GenMC and Spin, to model-check concurrent code"
# Spin preprocesses its models with `gcc`, which the cross compilers do not provide.
RUN install-llvm 22 clang-22 && \
    apt-get install -y --no-install-recommends gcc spin && \
    rm -rf /var/lib/apt/lists/*
# `genmc` links `libLLVM` from Clang 22 and includes its runtime headers in every client it compiles.
COPY --from=genmc /opt/genmc/bin/genmc /opt/genmc/bin/
COPY --from=genmc /opt/genmc/include/genmc/runtime/ /opt/genmc/include/genmc/runtime/
ENV GENMC=/opt/genmc/bin/genmc GENMC_CLANG=clang++-22

# AMD's image carries every precompiled ROCm math and ML library, about 30 GB of them. HIP compilation
# needs only ROCm's Clang, without its Fortran and MLIR parts, the HIP runtime and the headers: about 1 GB.
FROM ${ROCM} AS rocm-amd64
RUN mkdir -p /rocm && cd /opt/rocm/core && \
    tar -cf - --no-wildcards-match-slash \
        --exclude='lib/llvm/lib/*.a' --exclude='lib/llvm/lib/*MLIR*' --exclude='lib/llvm/lib/*mlir*' \
        include libexec share/hip lib/cmake lib/llvm lib/rocm_sysdeps bin/hip* lib/libamdhip64.so* \
        lib/libhsa-runtime64.so* lib/libamd_comgr.so* lib/librocm_kpack.so* lib/librocprofiler-register.so* | \
        tar -xf - -C /rocm

# AMD publishes ROCm for x86-64 alone, so the Arm image, as for Grace Hopper, carries CUDA only.
FROM ${UBUNTU} AS rocm-arm64
RUN mkdir -p /rocm

FROM rocm-${TARGETARCH} AS rocm

# ROCm's Clang compiles AMD code objects. NVCC 13.4 hosts on Clang 22: with 23, even under
# `-allow-unsupported-compiler`, its front end rejects libstdc++ 15's `<atomic>`, so every unit using it fails.
FROM ${UBUNTU} AS gpu
ARG CUDA=13.4
ARG LLVM
LABEL org.opencontainers.image.source=https://github.com/ashvardanian/FatWheels \
      org.opencontainers.image.title="FatWheels GPU" \
      org.opencontainers.image.description="CUDA 13.4, plus ROCm 10.0 on x86-64, for NVIDIA and AMD device code" \
      org.opencontainers.image.version=""
ENV DEBIAN_FRONTEND=noninteractive
COPY --from=scripts /install-llvm /usr/local/bin/install-llvm
COPY --from=scripts /dpkg-excludes /etc/dpkg/dpkg.cfg.d/fatwheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git cmake ninja-build make python3 python3-dev python3-venv xz-utils binutils && \
    repository=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604 && \
    platform=$(uname -m | sed 's/aarch64/sbsa/') && \
    curl -fsSLo /tmp/cuda-keyring.deb "${repository}/${platform}/cuda-keyring_1.1-1_all.deb" && \
    dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb && \
    install-llvm 22 clang-22 && install-llvm "${LLVM}" && \
    suffix=$(echo "${CUDA}" | tr . -) && \
    apt-get install -y --no-install-recommends "libomp-${LLVM}-dev" \
        "cuda-minimal-build-${suffix}" "cuda-driver-dev-${suffix}" "libcublas-dev-${suffix}" && \
    rm -rf /var/lib/apt/lists/*
# AMD's own layout: the versioned tree behind `core`, and the paths CMake and Clang look for linked to it.
COPY --from=rocm /rocm /opt/rocm/core
RUN if [ -d /opt/rocm/core/lib ]; then \
        for path in bin include lib libexec; do ln -s core/$path /opt/rocm/$path; done && \
        ln -s core/lib/llvm /opt/rocm/llvm && ln -s core/lib/llvm/amdgcn /opt/rocm/amdgcn && \
        echo /opt/rocm/core/lib >/etc/ld.so.conf.d/rocm.conf && ldconfig; \
    else rm -r /opt/rocm; fi
ENV PATH=/usr/local/cuda-${CUDA}/bin:${PATH} CUDAHOSTCXX=clang++-22
