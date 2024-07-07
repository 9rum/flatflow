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
#include <cstdint>
#include <limits>

namespace flatflow {
namespace data {
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

// Casts the given value under 32 bits to the corresponding 32-bit unsigned
// integer to prevent overflow during scheduling.
template <typename T>
  requires(Unsigned<T> && std::numeric_limits<T>::digits <
                              std::numeric_limits<uint32_t>::digits)
constexpr uint32_t OverflowSafeCast(T operand) noexcept {
  return static_cast<uint32_t>(operand);
}

// Casts the given value of 32 bits or more to the corresponding 64-bit unsigned
// integer to prevent overflow during scheduling.
template <typename T>
  requires(Unsigned<T> && std::numeric_limits<uint32_t>::digits <=
                              std::numeric_limits<T>::digits)
constexpr uint64_t OverflowSafeCast(T operand) noexcept {
  return static_cast<uint64_t>(operand);
}

}  // namespace internal
}  // namespace data
}  // namespace flatflow

#endif  // FLATFLOW_DATA_INTERNAL_TYPES_H_
