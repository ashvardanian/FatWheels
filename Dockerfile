# syntax=docker/dockerfile:1

# Toolchain images for building and validating code that ships multi-versioned CPU kernels and
# NVIDIA and AMD GPU code in one package. FatWheels' CI runs in them, and so can any other project's.
#
# - `cpu`: Clang 23, GCC 15, static QEMU user-mode, GCC 15 sysroots and, on x86-64, Intel SDE, to cross-compile
#   every CPU capability and run it under emulation.
# - `verify`: GenMC, Spin and the Clang 22 that GenMC compiles its clients with, to model-check concurrent code.
# - `gpu`: CUDA 13.5, plus ROCm 10.0 on x86-64, to compile NVIDIA and AMD device code side by side.
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
ARG QEMU_VERSION=11.1.3
ARG QEMU_SHA256=5a8289c7940dc77d3873c81a6c29ba3afe1446fc9d36e77fa1658cdbecb84b22
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

# SDE emulates the x86 capabilities QEMU lacks, AVX-512 and AMX among them; Intel's licence allows redistributing it
# unmodified with its notices, which stay in `/opt/sde/Licenses`.
FROM ${UBUNTU} AS sde-amd64
ARG SDE_MIRROR=924984
ARG SDE_VERSION=10.13.1-2026-07-28
ARG SDE_SHA256=94e97d623fec54385686e1e7ba65ebc9941748c05ee451423948334892bf2b50
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils && \
    curl -fsSLo /tmp/sde.tar.xz "https://downloadmirror.intel.com/${SDE_MIRROR}/sde-external-${SDE_VERSION}-lin.tar.xz" && \
    echo "${SDE_SHA256}  /tmp/sde.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/sde && tar -xJf /tmp/sde.tar.xz --strip-components=1 -C /opt/sde

FROM ${UBUNTU} AS sde-arm64
RUN mkdir -p /opt/sde

FROM sde-${TARGETARCH} AS sde

FROM ${UBUNTU} AS cpu
ARG LLVM
# Every stage inherits Ubuntu's own title, description and version labels unless it sets its own.
LABEL org.opencontainers.image.source=https://github.com/ashvardanian/FatWheels \
      org.opencontainers.image.title="FatWheels CPU" \
      org.opencontainers.image.description="Clang 23, GCC 15, QEMU, SDE and cross sysroots to build and emulate CPU capabilities" \
      org.opencontainers.image.version=""
ENV DEBIAN_FRONTEND=noninteractive
COPY --from=scripts /install-llvm /usr/local/bin/install-llvm
COPY --from=scripts /dpkg-excludes /etc/dpkg/dpkg.cfg.d/fatwheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git cmake ninja-build make python3 python3-pytest xz-utils binutils gcc-15 g++-15 && \
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
COPY --from=sde /opt/sde /opt/sde
RUN if [ -x /opt/sde/sde64 ]; then ln -s /opt/sde/sde64 /usr/local/bin/sde64; else rm -r /opt/sde; fi

# GenMC v0.20 supports LLVM 19 to 22, and its passes lack the `printPipeline` LLVM 23's pass manager requires, so the
# model checker and its clients compile with Clang 22 until a GenMC release supports 23.
FROM ${UBUNTU} AS genmc
ARG GENMC_COMMIT=fab5bba7d9efaf76990227c2f59d62c11c7ea2a3
ARG JOBS=
COPY --from=scripts /install-llvm /usr/local/bin/install-llvm
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git cmake ninja-build libffi-dev zlib1g-dev libedit-dev && \
    install-llvm 22 clang-22 llvm-22-dev
RUN git init -q /tmp/genmc && \
    git -C /tmp/genmc fetch -q --depth 1 https://github.com/MPI-SWS/genmc "${GENMC_COMMIT}" && \
    git -C /tmp/genmc checkout -q FETCH_HEAD && \
    cmake -S /tmp/genmc -B /tmp/genmc/build -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/opt/genmc \
        -D CMAKE_C_COMPILER=clang-22 -D CMAKE_CXX_COMPILER=clang++-22 -D CMAKE_PREFIX_PATH=/usr/lib/llvm-22 && \
    cmake --build /tmp/genmc/build --parallel ${JOBS:-$(nproc)} && \
    cmake --install /tmp/genmc/build --strip

# Spin's last tag, 6.5.2, dates from 2019; its master carries 6.5.3. `char` is unsigned on Arm, where the LTL
# lexer's `c != -1` would never see the end of its input.
FROM ${UBUNTU} AS spin
ARG SPIN_COMMIT=e21106ff7e58afe79275b3ca10179556afa97aa2
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git make gcc libc6-dev bison && \
    git init -q /tmp/spin && \
    git -C /tmp/spin fetch -q --depth 1 https://github.com/nimble-code/Spin "${SPIN_COMMIT}" && \
    git -C /tmp/spin checkout -q FETCH_HEAD && \
    make -C /tmp/spin/Src CC="gcc -fsigned-char" && \
    install -D /tmp/spin/Src/spin /opt/spin/bin/spin

# Neither checker needs the CPU image's emulators or sysroots, so this one starts from Ubuntu itself.
FROM ${UBUNTU} AS verify
LABEL org.opencontainers.image.source=https://github.com/ashvardanian/FatWheels \
      org.opencontainers.image.title="FatWheels Verify" \
      org.opencontainers.image.description="GenMC, Spin and Clang 22, to model-check concurrent code" \
      org.opencontainers.image.version=""
ENV DEBIAN_FRONTEND=noninteractive
COPY --from=scripts /install-llvm /usr/local/bin/install-llvm
# Spin preprocesses its models with `gcc`, the runner compiles them with `cc`, and GenMC's clients take the C and C++
# standard headers from `g++`; `git` lists the cited sources.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg git g++ && \
    install-llvm 22 clang-22 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=spin /opt/spin/bin/spin /usr/local/bin/
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
        --exclude='lib/llvm/bin/*flang*' --exclude='lib/llvm/bin/mlir-*' --exclude='lib/llvm/bin/fir-opt' \
        --exclude='lib/llvm/bin/bbc' --exclude='lib/llvm/bin/tco' \
        include libexec share/hip lib/cmake lib/llvm lib/rocm_sysdeps bin/hip* lib/libamdhip64.so* \
        lib/libhsa-runtime64.so* lib/libamd_comgr.so* lib/librocm_kpack.so* lib/librocprofiler-register.so* | \
        tar -xf - -C /rocm

# AMD publishes ROCm for x86-64 alone, so the Arm image, as for Grace Hopper, carries CUDA only.
FROM ${UBUNTU} AS rocm-arm64
RUN mkdir -p /rocm

FROM rocm-${TARGETARCH} AS rocm

# ROCm's Clang compiles AMD code objects. NVCC 13.5 supports Clang hosts up to 22, and with 23, even under
# `-allow-unsupported-compiler`, its front end rejects libstdc++ 15's `<atomic>`, so every unit using it fails.
FROM ${UBUNTU} AS gpu
ARG CUDA=13.5
ARG LLVM
LABEL org.opencontainers.image.source=https://github.com/ashvardanian/FatWheels \
      org.opencontainers.image.title="FatWheels GPU" \
      org.opencontainers.image.description="CUDA 13.5, plus ROCm 10.0 on x86-64, for NVIDIA and AMD device code" \
      org.opencontainers.image.version=""
ENV DEBIAN_FRONTEND=noninteractive
COPY --from=scripts /install-llvm /usr/local/bin/install-llvm
COPY --from=scripts /dpkg-excludes /etc/dpkg/dpkg.cfg.d/fatwheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git cmake ninja-build make python3 python3-dev python3-pytest python3-venv xz-utils binutils && \
    repository=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604 && \
    platform=$(uname -m | sed 's/aarch64/sbsa/') && \
    curl -fsSLo /tmp/cuda-keyring.deb "${repository}/${platform}/cuda-keyring_1.1-1_all.deb" && \
    dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb && \
    install-llvm 22 clang-22 && install-llvm "${LLVM}" && \
    suffix=$(echo "${CUDA}" | tr . -) && \
    apt-get install -y --no-install-recommends "libomp-${LLVM}-dev" \
        "cuda-minimal-build-${suffix}" "cuda-driver-dev-${suffix}" "cuda-nvdisasm-${suffix}" "libcublas-dev-${suffix}" && \
    rm -rf /var/lib/apt/lists/*
# AMD's own layout: the versioned tree behind `core`, and the paths CMake and Clang look for linked to it.
COPY --from=rocm /rocm /opt/rocm/core
RUN if [ -d /opt/rocm/core/lib ]; then \
        for path in bin include lib libexec; do ln -s core/$path /opt/rocm/$path; done && \
        ln -s core/lib/llvm /opt/rocm/llvm && ln -s core/lib/llvm/amdgcn /opt/rocm/amdgcn && \
        echo /opt/rocm/core/lib >/etc/ld.so.conf.d/rocm.conf && ldconfig; \
    else rm -r /opt/rocm; fi
ENV PATH=/usr/local/cuda-${CUDA}/bin:${PATH} CUDAHOSTCXX=clang++-22
