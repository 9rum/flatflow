// Copyright 2024 The FlatFlow Authors
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

#ifndef FLATFLOW_DATA_INTERNAL_TYPES_H_
#define FLATFLOW_DATA_INTERNAL_TYPES_H_

#include <concepts>

namespace flatflow {
namespace internal {

// The concept `Unsigned<T>` is satisfied if and only if `T` is an unsigned
// integral type and `std::same_as<T, bool>` is `false`. This is intended to be
// a drop-in replacement for `std::is_unsigned` and `std::unsigned_integral`, as
// they treat boolean types as integers.
template <typename T>
concept Unsigned = std::unsigned_integral<T> && !std::same_as<T, bool>;

// The concept `Integral<T>` is satisfied if and only if `T` is an integral type
// and both `std::same_as<T, char>` and `std::same_as<T, bool>` are `false`.
// This is intended to be a drop-in replacement for `std::is_integral` and
// `std::integral`, as they treat character and boolean types as integers.
template <typename T>
concept Integral =
    std::integral<T> && !std::same_as<T, char> && !std::same_as<T, bool>;

// The concept `Numerical<T>` is satisfied if and only if `T` is a numerical
// type (i.e., integers and floating-points) and both `std::same_as<T, char>`
// and `std::same_as<T, bool>` are `false`.
template <typename T>
concept Numerical = Integral<T> || std::floating_point<T>;

// Returns the given argument unchanged. This is intended to be a drop-in
// replacement for `std::forward` when template argument deduction fails if
// `std::forward` is used as a template parameter.
template <typename T>
constexpr decltype(auto) forward(T operand) noexcept {
  return operand;
}

}  // namespace internal
}  // namespace flatflow

#endif  // FLATFLOW_DATA_INTERNAL_TYPES_H_
