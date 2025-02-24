// Copyright 2025 The FlatFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FLATFLOW_OPS_INTERNAL_POLYNOMIAL_H_
#define FLATFLOW_OPS_INTERNAL_POLYNOMIAL_H_

#include <array>
#include <functional>
#include <utility>

#include "flatflow/types.h"

namespace flatflow {
namespace internal {

// polynomial<>
//
// This is a trivial class for polynomial manipulation, as an alternative to
// Boost polynomials. A notable API difference stems from the absence of
// division for polynomials over a field and over a unique factorization domain;
// we soon noticed that implementing symbolic transformations is equivalent to
// that of polynomial manipulation, where the division functionality between
// polynomials is not required.
template <typename T>
  requires flatflow::arithmetic<T>
class polynomial {
 public:
  using value_type = typename std::array<T, 4>::value_type;
  using size_type = typename std::array<T, 4>::size_type;

  // Constructors and assignment operators
  //
  // `polynomial<>` supports construction from an arbitrary number of values,
  // as well as the copy/move constructors and assignment operators below.
  //
  // CAVEATS
  //
  // For the constructor below, the arguments are used to value-initialize
  // the underlying fixed size array, and may not exceed the container capacity.
  template <typename... Args>
  polynomial(Args... args) : data_{args...} {}

  polynomial(const std::array<T, 4> &data) : data_(data) {}

  polynomial(const polynomial &other) : data_(other.data()) {}

  polynomial &operator=(const polynomial &other) {
    data_ = other.data();
    return *this;
  }

  polynomial(std::array<T, 4> &&data) : data_(std::move(data)) {}

  polynomial(polynomial &&other) : data_(std::move(other.data())) {}

  polynomial &operator=(polynomial &&other) {
    data_ = std::move(other.data());
    return *this;
  }

  // Accessors
  //
  // `polynomial<>` provides access to the underlying container and associated
  // metadata, just like Boost polynomials.
  size_type size() const { return data_.size(); }

  size_type degree() const { return data_.size() - 1; }

  std::array<T, 4> &data() { return data_; }

  const std::array<T, 4> &data() const { return data_; }

  value_type &operator[](size_type index) { return data_[index]; }

  value_type operator[](size_type index) const { return data_[index]; }

  // Operators
  //
  // `polynomial<>` supports basic polynomial arithmetic, as its Boost
  // counterpart do. Advanced manipulations such as fast Fourier transform (FFT)
  // and factorization are not supported for now.
  template <typename U>
    requires flatflow::arithmetic<U>
  value_type operator()(U value) const noexcept {
    return evaluate_polynomial(*this, value);
  }

  polynomial operator+(value_type value) const {
    auto p = *this;
    p += value;
    return p;
  }

  polynomial &operator+=(value_type value) { return addition(value); }

  polynomial operator+(const polynomial &other) const {
    auto p = *this;
    p += other;
    return p;
  }

  polynomial &operator+=(const polynomial &other) { return addition(other); }

  polynomial operator-(value_type value) const {
    auto p = *this;
    p -= value;
    return p;
  }

  polynomial &operator-=(value_type value) { return subtraction(value); }

  polynomial operator-(const polynomial &other) const {
    auto p = *this;
    p -= other;
    return p;
  }

  polynomial &operator-=(const polynomial &other) { return subtraction(other); }

  polynomial operator*(value_type value) const {
    auto p = *this;
    p *= value;
    return p;
  }

  polynomial &operator*=(value_type value) { return multiplication(value); }

  polynomial operator*(const polynomial &other) const {
    auto p = *this;
    p *= other;
    return p;
  }

  polynomial &operator*=(const polynomial &other) {
    return multiplication(other);
  }

  polynomial operator/(value_type value) const {
    auto p = *this;
    p /= value;
    return p;
  }

  polynomial &operator/=(value_type value) { return division(value); }

  polynomial operator<<(value_type value) const {
    auto p = *this;
    p <<= value;
    return p;
  }

  polynomial &operator<<=(value_type value) {
    data_[0] <<= value;
    data_[1] <<= value;
    data_[2] <<= value;
    data_[3] <<= value;
    return *this;
  }

  polynomial operator>>(value_type value) const {
    auto p = *this;
    p >>= value;
    return p;
  }

  polynomial &operator>>=(value_type value) {
    data_[0] >>= value;
    data_[1] >>= value;
    data_[2] >>= value;
    data_[3] >>= value;
    return *this;
  }

  bool operator==(const polynomial &other) const {
    return data_ == other.data();
  }

  bool operator!=(const polynomial &other) const {
    return data_ != other.data();
  }

 protected:
  template <typename BinaryOp>
  polynomial &addition(value_type value, BinaryOp op) {
    data_[0] = op(data_[0], value);
    return *this;
  }

  polynomial &addition(value_type value) {
    return addition(value, std::plus());
  }

  polynomial &subtraction(value_type value) {
    return addition(value, std::minus());
  }

  template <typename BinaryOp>
  polynomial &addition(const polynomial &other, BinaryOp op) {
    data_[0] = op(data_[0], other[0]);
    data_[1] = op(data_[1], other[1]);
    data_[2] = op(data_[2], other[2]);
    data_[3] = op(data_[3], other[3]);
    return *this;
  }

  polynomial &addition(const polynomial &other) {
    return addition(other, std::plus());
  }

  polynomial &subtraction(const polynomial &other) {
    return addition(other, std::minus());
  }

  template <typename BinaryOp>
  polynomial &multiplication(value_type value, BinaryOp op) {
    data_[0] = op(data_[0], value);
    data_[1] = op(data_[1], value);
    data_[2] = op(data_[2], value);
    data_[3] = op(data_[3], value);
    return *this;
  }

  polynomial &multiplication(value_type value) {
    return multiplication(value, std::multiplies());
  }

  polynomial &division(value_type value) {
    return multiplication(value, std::divides());
  }

  polynomial &multiplication(const polynomial &other) {
    auto data = std::array<T, 4>();

    data[0] = data_[0] * other[0];
    data[1] = data_[0] * other[1] + data_[1] * other[0];
    data[2] = data_[0] * other[2] + data_[1] * other[1] + data_[2] * other[0];
    data[3] = data_[0] * other[3] + data_[1] * other[2] + data_[2] * other[1] +
              data_[3] * other[0];

    data_[0] = data[0];
    data_[1] = data[1];
    data_[2] = data[2];
    data_[3] = data[3];

    return *this;
  }

  std::array<T, 4> data_;
};

template <typename T>
  requires flatflow::arithmetic<T>
constexpr T evaluate_polynomial_impl(const polynomial<T> &poly,
                                     T value) noexcept {
  return poly[0] + value * (poly[1] + value * (poly[2] + value * poly[3]));
}

// evaluate_polynomial()
//
// Based on Horner's rule, evaluates a given polynomial of degree three with
// only three multiplications and three additions, applying Horner's method.
//
// This is optimal, since there are polynomials of degree three that cannot be
// evaluated with fewer arithmetic operations.
// See https://doi.org/10.1070%2Frm1966v021n01abeh004147.
template <typename T, typename U>
  requires(flatflow::arithmetic<T> && flatflow::arithmetic<U>)
constexpr T evaluate_polynomial(const polynomial<T> &poly, U value) noexcept {
  return evaluate_polynomial_impl<T>(poly, value);
}

}  // namespace internal
}  // namespace flatflow

#endif  // FLATFLOW_OPS_INTERNAL_POLYNOMIAL_H_
