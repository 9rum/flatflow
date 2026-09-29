// SPDX-License-Identifier: Apache-2.0

#ifndef FLATFLOW_OPS_POLYNOMIAL_H_
#define FLATFLOW_OPS_POLYNOMIAL_H_

#include <array>
#include <utility>

#include "absl/log/check.h"

#include "flatflow/ops/scalar_type.h"
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
  using value_type =
      std::array<decltype(to_scale(std::declval<ScalarType>())), 3>::value_type;
  using size_type =
      std::array<decltype(to_scale(std::declval<ScalarType>())), 3>::size_type;

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

  constexpr value_type &operator[](size_type index) noexcept {
    return data_[index];
  }

  constexpr value_type operator[](size_type index) const noexcept {
    return data_[index];
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

 private:
  // Unlike Boost polynomials, the coefficients are stored in a fixed-size array
  // so that every operation can be evaluated at compile time.
  std::array<decltype(to_scale(std::declval<ScalarType>())), 3> data_;
};

template <typename... Args>
constexpr Polynomial polynomial(Args... args) noexcept {
  return Polynomial(args...);
}

}  // namespace flatflow

#endif  // FLATFLOW_OPS_POLYNOMIAL_H_
