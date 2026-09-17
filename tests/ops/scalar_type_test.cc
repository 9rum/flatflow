// SPDX-License-Identifier: Apache-2.0

#include "flatflow/ops/scalar_type.h"

#include "gtest/gtest.h"

#include "flatflow/ops/scalar_type_generated.h"

namespace {

TEST(PromoteTypesTest, HasDiagonalIdentity) {
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::float32) ==
                flatflow::ScalarType::float32);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::float64) ==
                flatflow::ScalarType::float64);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::float16) ==
                flatflow::ScalarType::float16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::bfloat16) ==
                flatflow::ScalarType::bfloat16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::bool_,
                                        flatflow::ScalarType::bool_) ==
                flatflow::ScalarType::bool_);
  static_assert(flatflow::promote_types(flatflow::ScalarType::int8,
                                        flatflow::ScalarType::int8) ==
                flatflow::ScalarType::int8);
  static_assert(flatflow::promote_types(flatflow::ScalarType::int16,
                                        flatflow::ScalarType::int16) ==
                flatflow::ScalarType::int16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::int32,
                                        flatflow::ScalarType::int32) ==
                flatflow::ScalarType::int32);
  static_assert(flatflow::promote_types(flatflow::ScalarType::int64,
                                        flatflow::ScalarType::int64) ==
                flatflow::ScalarType::int64);
  static_assert(flatflow::promote_types(flatflow::ScalarType::uint8,
                                        flatflow::ScalarType::uint8) ==
                flatflow::ScalarType::uint8);
  static_assert(flatflow::promote_types(flatflow::ScalarType::uint16,
                                        flatflow::ScalarType::uint16) ==
                flatflow::ScalarType::uint16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::uint32,
                                        flatflow::ScalarType::uint32) ==
                flatflow::ScalarType::uint32);
  static_assert(flatflow::promote_types(flatflow::ScalarType::uint64,
                                        flatflow::ScalarType::uint64) ==
                flatflow::ScalarType::uint64);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float8_e4m3fn,
                                        flatflow::ScalarType::float8_e4m3fn) ==
                flatflow::ScalarType::float8_e4m3fn);
  static_assert(
      flatflow::promote_types(flatflow::ScalarType::float8_e4m3fnuz,
                              flatflow::ScalarType::float8_e4m3fnuz) ==
      flatflow::ScalarType::float8_e4m3fnuz);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float8_e5m2,
                                        flatflow::ScalarType::float8_e5m2) ==
                flatflow::ScalarType::float8_e5m2);
  static_assert(
      flatflow::promote_types(flatflow::ScalarType::float8_e5m2fnuz,
                              flatflow::ScalarType::float8_e5m2fnuz) ==
      flatflow::ScalarType::float8_e5m2fnuz);
}

TEST(PromoteTypesTest, IsCommutative) {
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::float64) ==
                flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::float32));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::float16) ==
                flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::float32));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::bfloat16) ==
                flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::float32));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::bool_) ==
                flatflow::promote_types(flatflow::ScalarType::bool_,
                                        flatflow::ScalarType::float32));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::int8) ==
                flatflow::promote_types(flatflow::ScalarType::int8,
                                        flatflow::ScalarType::float32));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::int16) ==
                flatflow::promote_types(flatflow::ScalarType::int16,
                                        flatflow::ScalarType::float32));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::int32) ==
                flatflow::promote_types(flatflow::ScalarType::int32,
                                        flatflow::ScalarType::float32));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::int64) ==
                flatflow::promote_types(flatflow::ScalarType::int64,
                                        flatflow::ScalarType::float32));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::uint8) ==
                flatflow::promote_types(flatflow::ScalarType::uint8,
                                        flatflow::ScalarType::float32));

  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::float16) ==
                flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::float64));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::bfloat16) ==
                flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::float64));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::bool_) ==
                flatflow::promote_types(flatflow::ScalarType::bool_,
                                        flatflow::ScalarType::float64));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::int8) ==
                flatflow::promote_types(flatflow::ScalarType::int8,
                                        flatflow::ScalarType::float64));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::int16) ==
                flatflow::promote_types(flatflow::ScalarType::int16,
                                        flatflow::ScalarType::float64));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::int32) ==
                flatflow::promote_types(flatflow::ScalarType::int32,
                                        flatflow::ScalarType::float64));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::int64) ==
                flatflow::promote_types(flatflow::ScalarType::int64,
                                        flatflow::ScalarType::float64));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::uint8) ==
                flatflow::promote_types(flatflow::ScalarType::uint8,
                                        flatflow::ScalarType::float64));

  static_assert(flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::bfloat16) ==
                flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::float16));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::bool_) ==
                flatflow::promote_types(flatflow::ScalarType::bool_,
                                        flatflow::ScalarType::float16));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::int8) ==
                flatflow::promote_types(flatflow::ScalarType::int8,
                                        flatflow::ScalarType::float16));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::int16) ==
                flatflow::promote_types(flatflow::ScalarType::int16,
                                        flatflow::ScalarType::float16));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::int32) ==
                flatflow::promote_types(flatflow::ScalarType::int32,
                                        flatflow::ScalarType::float16));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::int64) ==
                flatflow::promote_types(flatflow::ScalarType::int64,
                                        flatflow::ScalarType::float16));
  static_assert(flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::uint8) ==
                flatflow::promote_types(flatflow::ScalarType::uint8,
                                        flatflow::ScalarType::float16));

  static_assert(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::bool_) ==
                flatflow::promote_types(flatflow::ScalarType::bool_,
                                        flatflow::ScalarType::bfloat16));
  static_assert(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::int8) ==
                flatflow::promote_types(flatflow::ScalarType::int8,
                                        flatflow::ScalarType::bfloat16));
  static_assert(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::int16) ==
                flatflow::promote_types(flatflow::ScalarType::int16,
                                        flatflow::ScalarType::bfloat16));
  static_assert(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::int32) ==
                flatflow::promote_types(flatflow::ScalarType::int32,
                                        flatflow::ScalarType::bfloat16));
  static_assert(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::int64) ==
                flatflow::promote_types(flatflow::ScalarType::int64,
                                        flatflow::ScalarType::bfloat16));
  static_assert(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::uint8) ==
                flatflow::promote_types(flatflow::ScalarType::uint8,
                                        flatflow::ScalarType::bfloat16));

  static_assert(flatflow::promote_types(flatflow::ScalarType::bool_,
                                        flatflow::ScalarType::int8) ==
                flatflow::promote_types(flatflow::ScalarType::int8,
                                        flatflow::ScalarType::bool_));
  static_assert(flatflow::promote_types(flatflow::ScalarType::bool_,
                                        flatflow::ScalarType::int16) ==
                flatflow::promote_types(flatflow::ScalarType::int16,
                                        flatflow::ScalarType::bool_));
  static_assert(flatflow::promote_types(flatflow::ScalarType::bool_,
                                        flatflow::ScalarType::int32) ==
                flatflow::promote_types(flatflow::ScalarType::int32,
                                        flatflow::ScalarType::bool_));
  static_assert(flatflow::promote_types(flatflow::ScalarType::bool_,
                                        flatflow::ScalarType::int64) ==
                flatflow::promote_types(flatflow::ScalarType::int64,
                                        flatflow::ScalarType::bool_));
  static_assert(flatflow::promote_types(flatflow::ScalarType::bool_,
                                        flatflow::ScalarType::uint8) ==
                flatflow::promote_types(flatflow::ScalarType::uint8,
                                        flatflow::ScalarType::bool_));

  static_assert(flatflow::promote_types(flatflow::ScalarType::int8,
                                        flatflow::ScalarType::int16) ==
                flatflow::promote_types(flatflow::ScalarType::int16,
                                        flatflow::ScalarType::int8));
  static_assert(flatflow::promote_types(flatflow::ScalarType::int8,
                                        flatflow::ScalarType::int32) ==
                flatflow::promote_types(flatflow::ScalarType::int32,
                                        flatflow::ScalarType::int8));
  static_assert(flatflow::promote_types(flatflow::ScalarType::int8,
                                        flatflow::ScalarType::int64) ==
                flatflow::promote_types(flatflow::ScalarType::int64,
                                        flatflow::ScalarType::int8));
  static_assert(flatflow::promote_types(flatflow::ScalarType::int8,
                                        flatflow::ScalarType::uint8) ==
                flatflow::promote_types(flatflow::ScalarType::uint8,
                                        flatflow::ScalarType::int8));

  static_assert(flatflow::promote_types(flatflow::ScalarType::int16,
                                        flatflow::ScalarType::int32) ==
                flatflow::promote_types(flatflow::ScalarType::int32,
                                        flatflow::ScalarType::int16));
  static_assert(flatflow::promote_types(flatflow::ScalarType::int16,
                                        flatflow::ScalarType::int64) ==
                flatflow::promote_types(flatflow::ScalarType::int64,
                                        flatflow::ScalarType::int16));
  static_assert(flatflow::promote_types(flatflow::ScalarType::int16,
                                        flatflow::ScalarType::uint8) ==
                flatflow::promote_types(flatflow::ScalarType::uint8,
                                        flatflow::ScalarType::int16));

  static_assert(flatflow::promote_types(flatflow::ScalarType::int32,
                                        flatflow::ScalarType::int64) ==
                flatflow::promote_types(flatflow::ScalarType::int64,
                                        flatflow::ScalarType::int32));
  static_assert(flatflow::promote_types(flatflow::ScalarType::int32,
                                        flatflow::ScalarType::uint8) ==
                flatflow::promote_types(flatflow::ScalarType::uint8,
                                        flatflow::ScalarType::int32));

  static_assert(flatflow::promote_types(flatflow::ScalarType::int64,
                                        flatflow::ScalarType::uint8) ==
                flatflow::promote_types(flatflow::ScalarType::uint8,
                                        flatflow::ScalarType::int64));
}

TEST(PromoteTypesTest, MatchesC10Reference) {
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::float64) ==
                flatflow::ScalarType::float64);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::float16) ==
                flatflow::ScalarType::float32);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::bfloat16) ==
                flatflow::ScalarType::float32);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::bool_) ==
                flatflow::ScalarType::float32);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::int8) ==
                flatflow::ScalarType::float32);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::int16) ==
                flatflow::ScalarType::float32);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::int32) ==
                flatflow::ScalarType::float32);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::int64) ==
                flatflow::ScalarType::float32);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float32,
                                        flatflow::ScalarType::uint8) ==
                flatflow::ScalarType::float32);

  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::float16) ==
                flatflow::ScalarType::float64);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::bfloat16) ==
                flatflow::ScalarType::float64);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::bool_) ==
                flatflow::ScalarType::float64);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::int8) ==
                flatflow::ScalarType::float64);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::int16) ==
                flatflow::ScalarType::float64);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::int32) ==
                flatflow::ScalarType::float64);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::int64) ==
                flatflow::ScalarType::float64);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float64,
                                        flatflow::ScalarType::uint8) ==
                flatflow::ScalarType::float64);

  static_assert(flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::bfloat16) ==
                flatflow::ScalarType::float32);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::bool_) ==
                flatflow::ScalarType::float16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::int8) ==
                flatflow::ScalarType::float16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::int16) ==
                flatflow::ScalarType::float16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::int32) ==
                flatflow::ScalarType::float16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::int64) ==
                flatflow::ScalarType::float16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::float16,
                                        flatflow::ScalarType::uint8) ==
                flatflow::ScalarType::float16);

  static_assert(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::bool_) ==
                flatflow::ScalarType::bfloat16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::int8) ==
                flatflow::ScalarType::bfloat16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::int16) ==
                flatflow::ScalarType::bfloat16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::int32) ==
                flatflow::ScalarType::bfloat16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::int64) ==
                flatflow::ScalarType::bfloat16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                        flatflow::ScalarType::uint8) ==
                flatflow::ScalarType::bfloat16);

  static_assert(flatflow::promote_types(flatflow::ScalarType::bool_,
                                        flatflow::ScalarType::int8) ==
                flatflow::ScalarType::int8);
  static_assert(flatflow::promote_types(flatflow::ScalarType::bool_,
                                        flatflow::ScalarType::int16) ==
                flatflow::ScalarType::int16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::bool_,
                                        flatflow::ScalarType::int32) ==
                flatflow::ScalarType::int32);
  static_assert(flatflow::promote_types(flatflow::ScalarType::bool_,
                                        flatflow::ScalarType::int64) ==
                flatflow::ScalarType::int64);
  static_assert(flatflow::promote_types(flatflow::ScalarType::bool_,
                                        flatflow::ScalarType::uint8) ==
                flatflow::ScalarType::uint8);

  static_assert(flatflow::promote_types(flatflow::ScalarType::int8,
                                        flatflow::ScalarType::int16) ==
                flatflow::ScalarType::int16);
  static_assert(flatflow::promote_types(flatflow::ScalarType::int8,
                                        flatflow::ScalarType::int32) ==
                flatflow::ScalarType::int32);
  static_assert(flatflow::promote_types(flatflow::ScalarType::int8,
                                        flatflow::ScalarType::int64) ==
                flatflow::ScalarType::int64);
  static_assert(flatflow::promote_types(flatflow::ScalarType::int8,
                                        flatflow::ScalarType::uint8) ==
                flatflow::ScalarType::int16);

  static_assert(flatflow::promote_types(flatflow::ScalarType::int16,
                                        flatflow::ScalarType::int32) ==
                flatflow::ScalarType::int32);
  static_assert(flatflow::promote_types(flatflow::ScalarType::int16,
                                        flatflow::ScalarType::int64) ==
                flatflow::ScalarType::int64);
  static_assert(flatflow::promote_types(flatflow::ScalarType::int16,
                                        flatflow::ScalarType::uint8) ==
                flatflow::ScalarType::int16);

  static_assert(flatflow::promote_types(flatflow::ScalarType::int32,
                                        flatflow::ScalarType::int64) ==
                flatflow::ScalarType::int64);
  static_assert(flatflow::promote_types(flatflow::ScalarType::int32,
                                        flatflow::ScalarType::uint8) ==
                flatflow::ScalarType::int32);

  static_assert(flatflow::promote_types(flatflow::ScalarType::int64,
                                        flatflow::ScalarType::uint8) ==
                flatflow::ScalarType::int64);
}

}  // namespace
