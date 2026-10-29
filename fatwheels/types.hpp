/**
 *  @file fatwheels/types.hpp
 *  @author Ash Vardanian
 *  @date August 10, 2024
 *  @brief Accumulator types the CPU and GPU kernels share.
 */
#pragma once
#include <cstdint> // `std::uint32_t`, `std::int64_t`

#include <type_traits> // `std::conditional_t`

namespace ashvardanian::fatwheels {

using cell_idx_t = std::uint32_t;

template <typename scalar_type>
using reduce_type = std::conditional_t< //
    std::is_floating_point_v<scalar_type>, double,
    std::conditional_t<std::is_signed_v<scalar_type>, std::int64_t, std::uint64_t>>;

template <typename scalar_type>
using matmul_type = std::conditional_t<                 //
    std::is_floating_point_v<scalar_type>, scalar_type, //
    std::conditional_t<std::is_signed_v<scalar_type>, std::int64_t, std::uint64_t>>;

/** The row-major operands of `C = A * B`: A is @c num_rows_a by @c num_cols_a, B is @c num_cols_a by @c num_cols_b. */
template <typename scalar_type>
struct matmul_operands {
    scalar_type const *matrix_a;
    scalar_type const *matrix_b;
    matmul_type<scalar_type> *matrix_c;
    cell_idx_t num_rows_a, num_cols_b, num_cols_a;
    cell_idx_t stride_a, stride_b, stride_c;
};

} // namespace ashvardanian::fatwheels
