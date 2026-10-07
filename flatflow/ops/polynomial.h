// SPDX-License-Identifier: Apache-2.0

#ifndef FLATFLOW_OPS_POLYNOMIAL_H_
#define FLATFLOW_OPS_POLYNOMIAL_H_

#include <algorithm>
#include <array>
#include <limits>

#include "absl/base/optimization.h"
#include "absl/log/check.h"

#include "flatflow/ops/graph.h"
#include "flatflow/ops/numeric.h"
#include "flatflow/ops/scalar_type_generated.h"

namespace flatflow {

// This is a trivial class for polynomial manipulation, as an alternative to
// Boost polynomials. A notable API difference lies in the absence of division
// for polynomials over a field and over a unique factorization domain; we soon
// noticed that implementing symbolic transformations is equivalent to that of
// polynomial manipulation where the division functionality between polynomials
// is not required.
class Polynomial {
 public:
  using value_type = std::array<SymInt::value_type, 3>::value_type;
  using size_type = std::array<SymInt::value_type, 3>::size_type;

  // Constructors and assignment operators
  //
  // `Polynomial` supports construction from an arbitrary number of values, as
  // well as copy/move constructors and assignment operators.
  //
  // CAVEATS
  //
  // For the constructor below, the arguments are used to aggregate-initialize
  // the underlying fixed-size array and may not exceed the container capacity.
  template <typename... Args>
  constexpr Polynomial(Args... args) noexcept : data_{args...} {}

  constexpr Polynomial(SymInt s) noexcept : data_{} {
    std::ranges::copy(s.data, data_.begin());
  }

  constexpr Polynomial(const Polynomial &) noexcept = default;

  constexpr Polynomial &operator=(const Polynomial &) noexcept = default;

  constexpr Polynomial(Polynomial &&) noexcept = default;

  constexpr Polynomial &operator=(Polynomial &&) noexcept = default;

  // Returns a new polynomial from `*this`, scaled by `world_size` for tensor
  // parallelism.
  constexpr Polynomial with_tensor_parallel(
      value_type world_size) const noexcept {
    CHECK_GT(world_size, 0);

    if (data_[1] % world_size == 0) {
      return Polynomial(data_[0], data_[1] / world_size, data_[2]);
    } else {
      return Polynomial(data_[0], data_[1], data_[2] * world_size);
    }
  }

  // Returns a new polynomial from `*this`, scaled by `world_size` for context
  // parallelism.
  constexpr Polynomial with_context_parallel(
      value_type world_size) const noexcept {
    CHECK_GT(world_size, 0);

    if (data_[2] % world_size == 0) {
      return Polynomial(data_[0], data_[1], data_[2] / world_size);
    } else {
      return Polynomial(data_[0], data_[1] * world_size, data_[2]);
    }
  }

  // Returns a new polynomial normalized from `*this` so that the constant term
  // becomes zero and the rest are relatively prime.
  constexpr Polynomial normalized() const noexcept {
    const auto divisor = static_cast<value_type>(gcd(data_[1], data_[2]));

    switch (divisor) {
      // The only case where the divisor is zero is when both `data_[1]` and
      // `data_[2]` are zero, in which case there is no need to divide them.
      case 0:
        return Polynomial(0, 0, 0);
      // The divisor is the minimum if and only if both `data_[1]` and
      // `data_[2]` are the minimum representable value, or one of them is the
      // minimum and the other is zero. In both cases the normalized value is -1
      // for the minimum and 0 for zero, which corresponds to their respective
      // signum.
      case std::numeric_limits<value_type>::min():
        return Polynomial(0, signum(data_[1]), signum(data_[2]));
      // Otherwise the divisor is positive and can be safely used for division.
      default:
        return Polynomial(0, data_[1] / divisor, data_[2] / divisor);
    }
  }

  constexpr value_type &operator[](size_type pos) noexcept {
    return data_[pos];
  }

  constexpr value_type operator[](size_type pos) const noexcept {
    return data_[pos];
  }

  // Based on Horner's rule, evaluates a given polynomial of degree two with
  // only two multiplications and two additions, applying Horner's method.
  // See https://dl.acm.org/doi/10.5555/270146 Chapter 4.6.4, Horner's rule.
  //
  // This is optimal, since there are polynomials of degree two that cannot be
  // evaluated with fewer arithmetic operations.
  // See https://doi.org/10.1070%2Frm1966v021n01abeh004147.
  constexpr value_type operator()(value_type value) const noexcept {
    return data_[0] + value * (data_[1] + value * data_[2]);
  }

  // Operators
  //
  // `Polynomial` supports basic polynomial arithmetic as its Boost counterpart
  // does, except for division between polynomials.
  constexpr bool operator==(const Polynomial &) const noexcept = default;

  // Note that this also serves the comparison in reversed order and `!=` in
  // both orders as the equality operators are rewritten by the compiler.
  constexpr bool operator==(value_type rhs) const noexcept {
    // This is equivalent to `data_[0] == rhs && data_[1] == 0 && data_[2] == 0`
    // but is written without short-circuit evaluation, which is not subject to
    // branch prediction.
    return ((data_[0] ^ rhs) | data_[1] | data_[2]) == 0;
  }

  constexpr Polynomial &operator+=(value_type rhs) noexcept {
    data_[0] += rhs;
    return *this;
  }

  constexpr Polynomial &operator-=(value_type rhs) noexcept {
    data_[0] -= rhs;
    return *this;
  }

  constexpr Polynomial &operator*=(value_type rhs) noexcept {
    data_[0] *= rhs;
    data_[1] *= rhs;
    data_[2] *= rhs;
    return *this;
  }

  constexpr Polynomial &operator/=(value_type rhs) noexcept {
    data_[0] /= rhs;
    data_[1] /= rhs;
    data_[2] /= rhs;
    return *this;
  }

  constexpr Polynomial &operator+=(const Polynomial &rhs) noexcept {
    data_[0] += rhs[0];
    data_[1] += rhs[1];
    data_[2] += rhs[2];
    return *this;
  }

  constexpr Polynomial &operator-=(const Polynomial &rhs) noexcept {
    data_[0] -= rhs[0];
    data_[1] -= rhs[1];
    data_[2] -= rhs[2];
    return *this;
  }

  constexpr Polynomial &operator*=(const Polynomial &rhs) noexcept {
    // Note that the coefficients are updated from the highest degree since each
    // coefficient of the product depends only on the coefficients of the same
    // or lower degrees. This also holds when `rhs` refers to `*this`.
    data_[2] = data_[0] * rhs[2] + data_[1] * rhs[1] + data_[2] * rhs[0];
    data_[1] = data_[0] * rhs[1] + data_[1] * rhs[0];
    data_[0] = data_[0] * rhs[0];
    return *this;
  }

  friend constexpr Polynomial operator-(const Polynomial &rhs) noexcept {
    return Polynomial(-rhs[0], -rhs[1], -rhs[2]);
  }

  friend constexpr Polynomial operator+(const Polynomial &lhs,
                                        value_type rhs) noexcept {
    return Polynomial(lhs[0] + rhs, lhs[1], lhs[2]);
  }

  friend constexpr Polynomial operator+(value_type lhs,
                                        const Polynomial &rhs) noexcept {
    return rhs + lhs;
  }

  friend constexpr Polynomial operator+(const Polynomial &lhs,
                                        const Polynomial &rhs) noexcept {
    return Polynomial(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]);
  }

  friend constexpr Polynomial operator-(const Polynomial &lhs,
                                        value_type rhs) noexcept {
    return Polynomial(lhs[0] - rhs, lhs[1], lhs[2]);
  }

  friend constexpr Polynomial operator-(value_type lhs,
                                        const Polynomial &rhs) noexcept {
    return -rhs + lhs;
  }

  friend constexpr Polynomial operator-(const Polynomial &lhs,
                                        const Polynomial &rhs) noexcept {
    return Polynomial(lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]);
  }

  friend constexpr Polynomial operator*(const Polynomial &lhs,
                                        value_type rhs) noexcept {
    return Polynomial(lhs[0] * rhs, lhs[1] * rhs, lhs[2] * rhs);
  }

  friend constexpr Polynomial operator*(value_type lhs,
                                        const Polynomial &rhs) noexcept {
    return rhs * lhs;
  }

  friend constexpr Polynomial operator*(const Polynomial &lhs,
                                        const Polynomial &rhs) noexcept {
    return Polynomial(lhs[0] * rhs[0], lhs[0] * rhs[1] + lhs[1] * rhs[0],
                      lhs[0] * rhs[2] + lhs[1] * rhs[1] + lhs[2] * rhs[0]);
  }

  friend constexpr Polynomial operator/(const Polynomial &lhs,
                                        value_type rhs) noexcept {
    return Polynomial(lhs[0] / rhs, lhs[1] / rhs, lhs[2] / rhs);
  }

 private:
  // Unlike Boost polynomials, the coefficients are stored in a fixed-size array
  // so that every operation can be evaluated at compile time.
  std::array<SymInt::value_type, 3> data_;
};

template <typename... Args>
constexpr Polynomial polynomial(Args... args) noexcept {
  return Polynomial(args...);
}

template <>
constexpr Polynomial polynomial(ScalarType dtype) noexcept {
  switch (dtype) {
    case ScalarType::int8:
    case ScalarType::uint8:
    case ScalarType::float8_e4m3fn:
    case ScalarType::float8_e4m3fnuz:
    case ScalarType::float8_e5m2:
    case ScalarType::float8_e5m2fnuz:
      return Polynomial(1);
    case ScalarType::float16:
    case ScalarType::bfloat16:
      return Polynomial(2);
    case ScalarType::float32:
      return Polynomial(4);
    case ScalarType::float64:
    case ScalarType::bool_:
    case ScalarType::int16:
    case ScalarType::int32:
    case ScalarType::uint16:
    case ScalarType::uint32:
      return Polynomial(64);
    case ScalarType::int64:
    case ScalarType::uint64:
      return Polynomial(128);
    default:
      ABSL_UNREACHABLE();
  }
}

}  // namespace flatflow

#endif  // FLATFLOW_OPS_POLYNOMIAL_H_
