/**
 *  @file fatwheels/cpu.hpp
 *  @author Ash Vardanian
 *  @date August 10, 2024
 *  @brief CPU kernels written once, compiled once per capability, and picked at runtime.
 *
 *  Every kernel is a generic template forced inline, and each capability is a wrapper with a @c target attribute,
 *  under which the compiler vectorizes that same template for AVX2, SVE or RVV; without the forcing, a large kernel
 *  stays an out-of-line call compiled for the baseline. OpenMP threads call those wrappers rather
 *  than run inside them, as Clang outlines an OpenMP region without its parent's target features.
 */
#pragma once
#include <algorithm> // `std::min`
#include <cstddef>   // `std::size_t`
#include <span>      // `std::span`
#include <cstring>   // `std::memset`

#if defined(__x86_64__)
#include <cpuid.h>     // `__get_cpuid`, `__get_cpuid_count`
#include <immintrin.h> // `_xgetbv`
#elif defined(__linux__) && (defined(__aarch64__) || defined(__riscv))
#include <sys/auxv.h> // `getauxval`
#endif

#include "fatwheels/types.hpp"

namespace ashvardanian::fatwheels {

/** The capabilities with kernels of their own; @c serial_k runs on every CPU. */
enum class capability_t { serial_k, haswell_k, sve_k, rvv_k };

inline char const *capability_name(capability_t capability) noexcept {
    switch (capability) {
    case capability_t::haswell_k: return "haswell";
    case capability_t::sve_k: return "sve";
    case capability_t::rvv_k: return "rvv";
    default: return "serial";
    }
}

#if defined(__x86_64__)
__attribute__((target("xsave"))) inline bool cpu_runs_haswell() noexcept {
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx) || !(ecx & bit_OSXSAVE) || !(ecx & bit_FMA)) return false;
    if ((_xgetbv(0) & 6) != 6) return false; // the OS must save the YMM registers, or AVX2 traps
    return __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) && (ebx & bit_AVX2);
}
#endif

/** The best capability this CPU runs, probed once. */
inline capability_t capability_best() noexcept {
#if defined(__x86_64__)
    static capability_t const best = cpu_runs_haswell() ? capability_t::haswell_k : capability_t::serial_k;
#elif defined(__linux__) && defined(__aarch64__)
    static capability_t const best = getauxval(AT_HWCAP) & HWCAP_SVE ? capability_t::sve_k : capability_t::serial_k;
#elif defined(__linux__) && defined(__riscv)
    static capability_t const best =
        getauxval(AT_HWCAP) & (1ul << ('V' - 'A')) ? capability_t::rvv_k : capability_t::serial_k;
#else
    static capability_t const best = capability_t::serial_k;
#endif
    return best;
}

template <typename scalar_type>
__attribute__((always_inline)) inline reduce_type<scalar_type> reduce_chunk(
    std::span<scalar_type const> data) noexcept {
    reduce_type<scalar_type> sum = 0;
#pragma omp simd reduction(+ : sum)
    for (std::size_t i = 0; i < data.size(); i++) sum += data[i];
    return sum;
}

/** Computes the output tile at row @p i and column @p j, staging both input tiles on the stack. */
template <typename scalar_type, cell_idx_t tile_size>
__attribute__((always_inline)) inline void matmul_tile(matmul_operands<scalar_type> const &operands, cell_idx_t i,
                                                       cell_idx_t j) noexcept {
    auto const &[matrix_a, matrix_b, matrix_c, num_rows_a, num_cols_b, num_cols_a, stride_a, stride_b, stride_c] =
        operands;
    scalar_type local_tile_a[tile_size][tile_size];
    scalar_type local_tile_b[tile_size][tile_size];
    matmul_type<scalar_type> local_tile_c[tile_size][tile_size];

    // Initialize the local tile to zero
    std::memset(local_tile_c, 0, tile_size * tile_size * sizeof(matmul_type<scalar_type>));

    for (cell_idx_t k = 0; k < num_cols_a; k += tile_size) {
        // Load tiles into local memory
        for (cell_idx_t ii = 0; ii < tile_size; ++ii)
            for (cell_idx_t kk = 0; kk < tile_size; ++kk)
                local_tile_a[ii][kk] =                           //
                    (i + ii < num_rows_a && k + kk < num_cols_a) //
                        ? matrix_a[(i + ii) * stride_a + (k + kk)]
                        : 0;

        for (cell_idx_t kk = 0; kk < tile_size; ++kk)
            for (cell_idx_t jj = 0; jj < tile_size; ++jj)
                local_tile_b[kk][jj] =                           //
                    (k + kk < num_cols_a && j + jj < num_cols_b) //
                        ? matrix_b[(k + kk) * stride_b + (j + jj)]
                        : 0;

        // Perform multiplication on the local tiles
        for (cell_idx_t ii = 0; ii < tile_size; ++ii)
            for (cell_idx_t jj = 0; jj < tile_size; ++jj)
                for (cell_idx_t kk = 0; kk < tile_size; ++kk)
                    local_tile_c[ii][jj] +=
                        static_cast<matmul_type<scalar_type>>(local_tile_a[ii][kk]) * local_tile_b[kk][jj];
    }

    // Write the result back to the output matrix
    for (cell_idx_t ii = 0; ii < tile_size; ++ii)
        for (cell_idx_t jj = 0; jj < tile_size; ++jj)
            if (i + ii < num_rows_a && j + jj < num_cols_b)
                matrix_c[(i + ii) * stride_c + (j + jj)] = local_tile_c[ii][jj];
}

template <typename scalar_type>
reduce_type<scalar_type> reduce_chunk_serial(std::span<scalar_type const> data) noexcept {
    return reduce_chunk(data);
}
template <typename scalar_type, cell_idx_t tile_size>
void matmul_tile_serial(matmul_operands<scalar_type> const &operands, cell_idx_t i, cell_idx_t j) noexcept {
    matmul_tile<scalar_type, tile_size>(operands, i, j);
}

#if defined(__x86_64__)
template <typename scalar_type>
__attribute__((target("avx2,fma"))) reduce_type<scalar_type> reduce_chunk_haswell(
    std::span<scalar_type const> data) noexcept {
    return reduce_chunk(data);
}
template <typename scalar_type, cell_idx_t tile_size>
__attribute__((target("avx2,fma"))) void matmul_tile_haswell(matmul_operands<scalar_type> const &operands, cell_idx_t i,
                                                             cell_idx_t j) noexcept {
    matmul_tile<scalar_type, tile_size>(operands, i, j);
}
#elif defined(__aarch64__)
template <typename scalar_type>
__attribute__((target("sve"))) reduce_type<scalar_type> reduce_chunk_sve(std::span<scalar_type const> data) noexcept {
    return reduce_chunk(data);
}
template <typename scalar_type, cell_idx_t tile_size>
__attribute__((target("sve"))) void matmul_tile_sve(matmul_operands<scalar_type> const &operands, cell_idx_t i,
                                                    cell_idx_t j) noexcept {
    matmul_tile<scalar_type, tile_size>(operands, i, j);
}
#elif defined(__riscv)
template <typename scalar_type>
__attribute__((target("arch=+v"))) reduce_type<scalar_type> reduce_chunk_rvv(
    std::span<scalar_type const> data) noexcept {
    return reduce_chunk(data);
}
template <typename scalar_type, cell_idx_t tile_size>
__attribute__((target("arch=+v"))) void matmul_tile_rvv(matmul_operands<scalar_type> const &operands, cell_idx_t i,
                                                        cell_idx_t j) noexcept {
    matmul_tile<scalar_type, tile_size>(operands, i, j);
}
#endif

template <typename scalar_type>
using reduce_chunk_kernel = reduce_type<scalar_type> (*)(std::span<scalar_type const>) noexcept;

template <typename scalar_type>
using matmul_tile_kernel = void (*)(matmul_operands<scalar_type> const &, cell_idx_t, cell_idx_t) noexcept;

template <typename scalar_type>
reduce_chunk_kernel<scalar_type> reduce_chunk_for(capability_t capability) noexcept {
    switch (capability) {
#if defined(__x86_64__)
    case capability_t::haswell_k: return &reduce_chunk_haswell<scalar_type>;
#elif defined(__aarch64__)
    case capability_t::sve_k: return &reduce_chunk_sve<scalar_type>;
#elif defined(__riscv)
    case capability_t::rvv_k: return &reduce_chunk_rvv<scalar_type>;
#endif
    default: return &reduce_chunk_serial<scalar_type>;
    }
}

template <typename scalar_type, cell_idx_t tile_size>
matmul_tile_kernel<scalar_type> matmul_tile_for(capability_t capability) noexcept {
    switch (capability) {
#if defined(__x86_64__)
    case capability_t::haswell_k: return &matmul_tile_haswell<scalar_type, tile_size>;
#elif defined(__aarch64__)
    case capability_t::sve_k: return &matmul_tile_sve<scalar_type, tile_size>;
#elif defined(__riscv)
    case capability_t::rvv_k: return &matmul_tile_rvv<scalar_type, tile_size>;
#endif
    default: return &matmul_tile_serial<scalar_type, tile_size>;
    }
}

/**
 *  @brief Sums the elements of @p data, splitting them across OpenMP threads.
 *  @param[in] capability The kernels to run, like @c capability_best() reports; @c serial_k runs on every CPU.
 */
template <typename scalar_type>
reduce_type<scalar_type> reduce(std::span<scalar_type const> data, capability_t capability) noexcept {
    reduce_chunk_kernel<scalar_type> const kernel = reduce_chunk_for<scalar_type>(capability);
    constexpr std::size_t chunk_length = 1 << 16;
    reduce_type<scalar_type> sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (std::size_t start = 0; start < data.size(); start += chunk_length)
        sum += kernel(data.subspan(start, std::min(chunk_length, data.size() - start)));
    return sum;
}

/**
 *  @brief Computes `C = A * B` in @c tile_size tiles, splitting the tiles across OpenMP threads.
 *  @param[in] operands The matrices, with C written in place.
 *  @param[in] capability The kernels to run, like @c capability_best() reports; @c serial_k runs on every CPU.
 */
template <typename scalar_type, cell_idx_t tile_size>
void matmul(matmul_operands<scalar_type> const &operands, capability_t capability) noexcept {
    matmul_tile_kernel<scalar_type> const kernel = matmul_tile_for<scalar_type, tile_size>(capability);
#pragma omp parallel for collapse(2)
    for (cell_idx_t i = 0; i < operands.num_rows_a; i += tile_size)
        for (cell_idx_t j = 0; j < operands.num_cols_b; j += tile_size) kernel(operands, i, j);
}

} // namespace ashvardanian::fatwheels
