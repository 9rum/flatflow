// SPDX-License-Identifier: Apache-2.0

#ifndef FLATFLOW_OPS_SCALAR_TYPE_H_
#define FLATFLOW_OPS_SCALAR_TYPE_H_

#include <array>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"

#include "flatflow/ops/scalar_type_generated.h"

namespace flatflow {

namespace internal {

inline constexpr auto f4 = ScalarType::float32;
inline constexpr auto f8 = ScalarType::float64;
inline constexpr auto f2 = ScalarType::float16;
inline constexpr auto bf = ScalarType::bfloat16;
inline constexpr auto b1 = ScalarType::bool_;
inline constexpr auto i1 = ScalarType::int8;
inline constexpr auto i2 = ScalarType::int16;
inline constexpr auto i4 = ScalarType::int32;
inline constexpr auto i8 = ScalarType::int64;
inline constexpr auto u1 = ScalarType::uint8;

inline constexpr auto lookup = std::to_array({
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

}  // namespace internal

constexpr bool is_float8_type(ScalarType dtype) noexcept {
  return dtype == ScalarType::float8_e4m3fn ||
         dtype == ScalarType::float8_e4m3fnuz ||
         dtype == ScalarType::float8_e5m2 ||
         dtype == ScalarType::float8_e5m2fnuz;
}

constexpr bool is_reduced_floating_type(ScalarType dtype) noexcept {
  return dtype == ScalarType::float16 || dtype == ScalarType::bfloat16 ||
         is_float8_type(dtype);
}

constexpr bool is_floating_type(ScalarType dtype) noexcept {
  return dtype == ScalarType::float32 || dtype == ScalarType::float64 ||
         is_reduced_floating_type(dtype);
}

constexpr bool is_barebones_unsigned_type(ScalarType dtype) noexcept {
  return dtype == ScalarType::uint16 || dtype == ScalarType::uint32 ||
         dtype == ScalarType::uint64;
}

// Returns the data type with the smallest size and scalar kind that is not
// smaller nor of lower kind than either `lhs` or `rhs`.
//
// See https://docs.pytorch.org/docs/stable/tensor_attributes.html for more
// information on the type promotion logic.
constexpr ScalarType promote_types(ScalarType lhs, ScalarType rhs) noexcept {
  // If the two types are equal, return that type.
  if (lhs == rhs) {
    return lhs;
  }

  CHECK(!(is_float8_type(lhs) || is_float8_type(rhs))) << absl::StrFormat(
      "Promotion for float8 types is not supported, attempted to promote %s "
      "and %s",
      EnumNameScalarType(lhs), EnumNameScalarType(rhs));

  if (is_barebones_unsigned_type(lhs) || is_barebones_unsigned_type(rhs)) {
    if (is_floating_type(lhs)) {
      return lhs;
    }
    if (is_floating_type(rhs)) {
      return rhs;
    }
    CHECK(false) << absl::StrFormat(
        "Promotion for uint16, uint32, uint64 types is not supported, "
        "attempted to promote %s and %s",
        EnumNameScalarType(lhs), EnumNameScalarType(rhs));
  }

  const auto lix = static_cast<decltype(internal::lookup)::size_type>(lhs);
  const auto rix = static_cast<decltype(internal::lookup)::size_type>(rhs);

  return internal::lookup[lix][rix];
}

}  // namespace flatflow

#endif  // FLATFLOW_OPS_SCALAR_TYPE_H_
