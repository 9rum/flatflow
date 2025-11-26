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

#ifndef FLATFLOW_OPS_PROMOTE_TYPES_H_
#define FLATFLOW_OPS_PROMOTE_TYPES_H_

#include <array>
#include <cstddef>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"

#include "flatflow/ops/scalar_type_generated.h"

namespace flatflow {

constexpr bool is_float8(ScalarType dtype) noexcept {
  return dtype == ScalarType::FLOAT8_E4M3FN ||
         dtype == ScalarType::FLOAT8_E4M3FNUZ ||
         dtype == ScalarType::FLOAT8_E5M2 ||
         dtype == ScalarType::FLOAT8_E5M2FNUZ;
}

constexpr bool is_reduced_floating_point(ScalarType dtype) noexcept {
  return dtype == ScalarType::FLOAT16 || dtype == ScalarType::BFLOAT16 ||
         is_float8(dtype);
}

constexpr bool is_floating_point(ScalarType dtype) noexcept {
  return dtype == ScalarType::FLOAT32 || dtype == ScalarType::FLOAT64 ||
         is_reduced_floating_point(dtype);
}

constexpr bool is_barebones_unsigned(ScalarType dtype) noexcept {
  return dtype == ScalarType::UINT16 || dtype == ScalarType::UINT32 ||
         dtype == ScalarType::UINT64;
}

// flatflow::promote_types()
//
// Returns the smallest size and scalar kind that is not smaller nor of lower
// kind than either `lhs` or `rhs`. See
// https://docs.pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc
// for more information on the type promotion logic.
ScalarType promote_types(ScalarType lhs, ScalarType rhs) {
  // If the two types are equal, return that type.
  if (lhs == rhs) {
    return lhs;
  }

  CHECK(!(is_float8(lhs) || is_float8(rhs))) << absl::StrFormat(
      "Promotion for float8 types is not supported, attempted to promote %s "
      "and %s",
      EnumNameScalarType(lhs), EnumNameScalarType(rhs));

  if (is_barebones_unsigned(lhs) || is_barebones_unsigned(rhs)) {
    if (is_floating_point(lhs)) {
      return lhs;
    }
    if (is_floating_point(rhs)) {
      return rhs;
    }
    LOG(FATAL) << absl::StrFormat(
        "Promotion for uint16, uint32, uint64 types is not supported, "
        "attempted to promote %s and %s",
        EnumNameScalarType(lhs), EnumNameScalarType(rhs));
  }

  static constexpr auto f4 = ScalarType::FLOAT32;
  static constexpr auto f8 = ScalarType::FLOAT64;
  static constexpr auto f2 = ScalarType::FLOAT16;
  static constexpr auto bf = ScalarType::BFLOAT16;
  static constexpr auto b1 = ScalarType::BOOL;
  static constexpr auto i1 = ScalarType::INT8;
  static constexpr auto i2 = ScalarType::INT16;
  static constexpr auto i4 = ScalarType::INT32;
  static constexpr auto i8 = ScalarType::INT64;
  static constexpr auto u1 = ScalarType::UINT8;

  static constexpr auto lookup = std::to_array({
      /*                      f4  f8  f2  bf  b1  i1  i2  i4  i8  u1 */
      std::to_array(/* f4 */ {f4, f8, f4, f4, f4, f4, f4, f4, f4, f4}),
      std::to_array(/* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, f8, f8}),
      std::to_array(/* f2 */ {f4, f8, f2, f4, f2, f2, f2, f2, f2, f2}),
      std::to_array(/* bf */ {f4, f8, f4, bf, bf, bf, bf, bf, bf, bf}),
      std::to_array(/* b1 */ {f4, f8, f2, bf, b1, i1, i2, i4, i8, u1}),
      std::to_array(/* i1 */ {f4, f8, f2, bf, i1, i1, i2, i4, i8, i2}),
      std::to_array(/* i2 */ {f4, f8, f2, bf, i2, i2, i2, i4, i8, i2}),
      std::to_array(/* i4 */ {f4, f8, f2, bf, i4, i4, i4, i4, i8, i4}),
      std::to_array(/* i8 */ {f4, f8, f2, bf, i8, i8, i8, i8, i8, i8}),
      std::to_array(/* u1 */ {f4, f8, f2, bf, u1, i2, i2, i4, i8, u1}),
  });

  return lookup[static_cast<std::size_t>(lhs)][static_cast<std::size_t>(rhs)];
}

}  // namespace flatflow

#endif  // FLATFLOW_OPS_PROMOTE_TYPES_H_
