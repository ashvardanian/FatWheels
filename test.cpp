/**
 *  @file test.cpp
 *  @author Ash Vardanian
 *  @date August 10, 2024
 *  @brief Checks the best capability's CPU kernels against the serial ones, without Python, as QEMU runs them.
 */
#include <cstdint> // `std::int32_t`, `std::int64_t`
#include <cstdio>  // `std::printf`
#include <span>    // `std::span`
#include <vector>  // `std::vector`

#include "fatwheels/cpu.hpp"

namespace fw = ashvardanian::fatwheels;

namespace ashvardanian::fatwheels {

/** Multiplies two @p size by @p size matrices at @p capability and returns the product. */
template <cell_idx_t tile_size>
static std::vector<std::int64_t> matmul_square(std::vector<std::int32_t> const &a, std::vector<std::int32_t> const &b,
                                               cell_idx_t size, capability_t capability) {
    std::vector<std::int64_t> c(a.size());
    matmul<std::int32_t, tile_size>({a.data(), b.data(), c.data(), size, size, size, size, size, size}, capability);
    return c;
}

} // namespace ashvardanian::fatwheels

int main() {
    fw::capability_t const best = fw::capability_best();
    int failures = 0;

    // Small whole numbers keep every `double` sum exact, so any difference is a kernel bug, not rounding.
    std::vector<float> data(1'000'003);
    for (std::size_t i = 0; i < data.size(); i++) data[i] = static_cast<float>(i % 100);
    double const expected = fw::reduce(std::span<float const>(data), fw::capability_t::serial_k);
    double const got = fw::reduce(std::span<float const>(data), best);
    std::printf("reduce on %s: %.0f, serial: %.0f\n", fw::capability_name(best), got, expected);
    failures += got != expected;

    // 67 is prime, so every tile size leaves a partial tile on both edges.
    constexpr fw::cell_idx_t size = 67;
    std::vector<std::int32_t> a(size * size), b(size * size);
    for (std::size_t i = 0; i < a.size(); i++)
        a[i] = static_cast<std::int32_t>(i % 7), b[i] = static_cast<std::int32_t>(i % 5);
    bool const matmul_matches = //
        fw::matmul_square<4>(a, b, size, best) == fw::matmul_square<4>(a, b, size, fw::capability_t::serial_k) &&
        fw::matmul_square<8>(a, b, size, best) == fw::matmul_square<8>(a, b, size, fw::capability_t::serial_k) &&
        fw::matmul_square<16>(a, b, size, best) == fw::matmul_square<16>(a, b, size, fw::capability_t::serial_k) &&
        fw::matmul_square<32>(a, b, size, best) == fw::matmul_square<32>(a, b, size, fw::capability_t::serial_k);
    std::printf("matmul on %s: %s serial at every tile size\n", fw::capability_name(best),
                matmul_matches ? "matches" : "differs from");
    failures += !matmul_matches;
    return failures;
}
