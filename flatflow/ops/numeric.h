// SPDX-License-Identifier: Apache-2.0

#ifndef FLATFLOW_OPS_NUMERIC_H_
#define FLATFLOW_OPS_NUMERIC_H_

#include <numeric>
#include <type_traits>

namespace flatflow {

// Computes the absolute value of the integer `num`.
//
// Unlike `std::abs` whose behavior is undefined if the result cannot be
// represented by the return type, this `abs` returns the result as the unsigned
// counterpart of the argument type so that the behavior is defined for every
// argument, including the minimum representable value. Unsigned arguments are
// returned as is.
template <typename T>
constexpr std::make_unsigned_t<T> abs(T num) noexcept {
  if constexpr (std::is_signed_v<T>) {
    const auto bits = static_cast<std::make_unsigned_t<T>>(num);
    return num < 0 ? 0 - bits : bits;
  } else {
    return num;
  }
}

// Computes the greatest common divisor of the integers `m` and `n`.
//
// Unlike `std::gcd` whose behavior is undefined if the absolute value of either
// argument is not representable as a value of the common type, this `gcd`
// returns the result as the unsigned counterpart of the common type so that the
// behavior is defined for every pair of arguments, including the minimum
// representable values.
template <typename M, typename N>
constexpr std::make_unsigned_t<std::common_type_t<M, N>> gcd(M m,
                                                             N n) noexcept {
  return static_cast<std::make_unsigned_t<std::common_type_t<M, N>>>(
      std::gcd(abs(m), abs(n)));
}

// Returns a number representing sign of the integer `num`.
//
//  - `0` if the number is zero
//  - `1` if the number is positive
//  - `-1` if the number is negative
template <typename T>
constexpr T signum(T num) noexcept {
  return static_cast<T>((0 < num) - (num < 0));
}

}  // namespace flatflow

#endif  // FLATFLOW_OPS_NUMERIC_H_
