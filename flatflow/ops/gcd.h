// SPDX-License-Identifier: Apache-2.0

#ifndef FLATFLOW_OPS_GCD_H_
#define FLATFLOW_OPS_GCD_H_

#include <numeric>
#include <type_traits>

namespace flatflow {

// Computes the absolute value of the integer number `num`.
//
// Unlike `std::abs` whose behavior is undefined if the result cannot be
// represented by the return type, this returns the result as the unsigned
// counterpart of the argument type so that the behavior is defined for every
// argument, including the minimum representable value. Unsigned arguments are
// returned as is.
template <typename T>
constexpr std::make_unsigned_t<T> unsigned_abs(T num) noexcept {
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
// argument is not representable as a value of the common type, this returns the
// result as the unsigned counterpart of the common type so that the behavior is
// defined for every pair of arguments, including the minimum representable
// values.
template <typename M, typename N>
constexpr std::make_unsigned_t<std::common_type_t<M, N>> gcd(M m,
                                                             N n) noexcept {
  return static_cast<std::make_unsigned_t<std::common_type_t<M, N>>>(
      std::gcd(unsigned_abs(m), unsigned_abs(n)));
}

}  // namespace flatflow

#endif  // FLATFLOW_OPS_GCD_H_
