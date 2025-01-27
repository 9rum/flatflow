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

#include "flatflow/data/internal/types.h"

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
  requires Numerical<T>
class polynomial {
 public:
  using container_type = std::array<T, 4>;
  using value_type = typename container_type::value_type;
  using size_type = typename container_type::size_type;

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

  polynomial(const container_type &data) : data_(data) {}

  polynomial(const polynomial &other) : data_(other.data()) {}

  polynomial &operator=(const polynomial &other) {
    data_ = other.data();
    return *this;
  }

  polynomial(container_type &&data) : data_(std::move(data)) {}

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

  container_type &data() { return data_; }

  const container_type &data() const { return data_; }

  value_type &operator[](size_type index) { return data_[index]; }

  const value_type &operator[](size_type index) const { return data_[index]; }

  // Operators
  //
  // `polynomial<>` supports basic polynomial arithmetic, as its Boost
  // counterpart do. Advanced manipulations such as fast Fourier transform (FFT)
  // and factorization are not supported for now.
  template <typename V>
    requires Numerical<V>
  T operator()(const V &value) const noexcept {
    return evaluate_polynomial(*this, value);
  }

  polynomial operator+(const T &value) const {
    auto p = *this;
    p += value;
    return p;
  }

  polynomial &operator+=(const T &value) { return addition(value); }

  polynomial operator+(const polynomial &other) const {
    auto p = *this;
    p += other;
    return p;
  }

  polynomial &operator+=(const polynomial &other) { return addition(other); }

  polynomial operator-(const T &value) const {
    auto p = *this;
    p -= value;
    return p;
  }

  polynomial &operator-=(const T &value) { return subtraction(value); }

  polynomial operator-(const polynomial &other) const {
    auto p = *this;
    p -= other;
    return p;
  }

  polynomial &operator-=(const polynomial &other) { return subtraction(other); }

  polynomial operator*(const T &value) const {
    auto p = *this;
    p *= value;
    return p;
  }

  polynomial &operator*=(const T &value) { return multiplication(value); }

  polynomial operator*(const polynomial &other) const {
    auto p = *this;
    p *= other;
    return p;
  }

  polynomial &operator*=(const polynomial &other) {
    return multiplication(other);
  }

  polynomial operator/(const T &value) const {
    auto p = *this;
    p /= value;
    return p;
  }

  polynomial &operator/=(const T &value) { return division(value); }

  polynomial operator<<(const T &value) const {
    auto p = *this;
    p <<= value;
    return p;
  }

  polynomial &operator<<=(const T &value) {
    data_[0] <<= value;
    data_[1] <<= value;
    data_[2] <<= value;
    data_[3] <<= value;
    return *this;
  }

  polynomial operator>>(const T &value) const {
    auto p = *this;
    p >>= value;
    return p;
  }

  polynomial &operator>>=(const T &value) {
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

 private:
  template <typename BinaryOp>
  polynomial &addition(const T &value, BinaryOp op) {
    data_[0] = op(data_[0], value);
    return *this;
  }

  polynomial &addition(const T &value) { return addition(value, std::plus()); }

  polynomial &subtraction(const T &value) {
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
  polynomial &multiplication(const T &value, BinaryOp op) {
    data_[0] = op(data_[0], value);
    data_[1] = op(data_[1], value);
    data_[2] = op(data_[2], value);
    data_[3] = op(data_[3], value);
    return *this;
  }

  polynomial &multiplication(const T &value) {
    return multiplication(value, std::multiplies());
  }

  polynomial &division(const T &value) {
    return multiplication(value, std::divides());
  }

  polynomial &multiplication(const polynomial &other) {
    auto data = container_type();

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

  container_type data_;
};

template <typename T>
  requires Numerical<T>
constexpr T evaluate_polynomial_impl(const polynomial<T> &p,
                                     const T &value) noexcept {
  return p[0] + value * (p[1] + value * (p[2] + value * p[3]));
}

// evaluate_polynomial()
//
// Based on Horner's rule, evaluates a given polynomial of degree three with
// only three multiplications and three additions, applying Horner's method.
//
// This is optimal, since there are polynomials of degree three that cannot be
// evaluated with fewer arithmetic operations.
// See https://doi.org/10.1070%2Frm1966v021n01abeh004147.
template <typename T, typename V>
  requires(Numerical<T> && Numerical<V>)
constexpr T evaluate_polynomial(const polynomial<T> &p,
                                const V &value) noexcept {
  return evaluate_polynomial_impl<T>(p, value);
}

}  // namespace internal
}  // namespace flatflow

#endif  // FLATFLOW_OPS_INTERNAL_POLYNOMIAL_H_
