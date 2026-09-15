// SPDX-License-Identifier: Apache-2.0

#include "flatflow/ops/scalar_type.h"

#include "gtest/gtest.h"

#include "flatflow/ops/scalar_type_generated.h"

namespace {

TEST(PromoteTypesTest, HasDiagonalIdentity) {
  for (const auto dtype : flatflow::EnumValuesScalarType()) {
    EXPECT_EQ(flatflow::promote_types(dtype, dtype), dtype);
  }
}

TEST(PromoteTypesTest, IsCommutative) {
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::float64),
            flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::float32));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::float16),
            flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::float32));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::bfloat16),
            flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                    flatflow::ScalarType::float32));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::bool_),
            flatflow::promote_types(flatflow::ScalarType::bool_,
                                    flatflow::ScalarType::float32));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::int8),
            flatflow::promote_types(flatflow::ScalarType::int8,
                                    flatflow::ScalarType::float32));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::int16),
            flatflow::promote_types(flatflow::ScalarType::int16,
                                    flatflow::ScalarType::float32));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::int32),
            flatflow::promote_types(flatflow::ScalarType::int32,
                                    flatflow::ScalarType::float32));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::int64),
            flatflow::promote_types(flatflow::ScalarType::int64,
                                    flatflow::ScalarType::float32));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::uint8),
            flatflow::promote_types(flatflow::ScalarType::uint8,
                                    flatflow::ScalarType::float32));

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::float16),
            flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::float64));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::bfloat16),
            flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                    flatflow::ScalarType::float64));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::bool_),
            flatflow::promote_types(flatflow::ScalarType::bool_,
                                    flatflow::ScalarType::float64));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::int8),
            flatflow::promote_types(flatflow::ScalarType::int8,
                                    flatflow::ScalarType::float64));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::int16),
            flatflow::promote_types(flatflow::ScalarType::int16,
                                    flatflow::ScalarType::float64));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::int32),
            flatflow::promote_types(flatflow::ScalarType::int32,
                                    flatflow::ScalarType::float64));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::int64),
            flatflow::promote_types(flatflow::ScalarType::int64,
                                    flatflow::ScalarType::float64));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::uint8),
            flatflow::promote_types(flatflow::ScalarType::uint8,
                                    flatflow::ScalarType::float64));

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::bfloat16),
            flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                    flatflow::ScalarType::float16));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::bool_),
            flatflow::promote_types(flatflow::ScalarType::bool_,
                                    flatflow::ScalarType::float16));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::int8),
            flatflow::promote_types(flatflow::ScalarType::int8,
                                    flatflow::ScalarType::float16));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::int16),
            flatflow::promote_types(flatflow::ScalarType::int16,
                                    flatflow::ScalarType::float16));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::int32),
            flatflow::promote_types(flatflow::ScalarType::int32,
                                    flatflow::ScalarType::float16));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::int64),
            flatflow::promote_types(flatflow::ScalarType::int64,
                                    flatflow::ScalarType::float16));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::uint8),
            flatflow::promote_types(flatflow::ScalarType::uint8,
                                    flatflow::ScalarType::float16));

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                    flatflow::ScalarType::bool_),
            flatflow::promote_types(flatflow::ScalarType::bool_,
                                    flatflow::ScalarType::bfloat16));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                    flatflow::ScalarType::int8),
            flatflow::promote_types(flatflow::ScalarType::int8,
                                    flatflow::ScalarType::bfloat16));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                    flatflow::ScalarType::int16),
            flatflow::promote_types(flatflow::ScalarType::int16,
                                    flatflow::ScalarType::bfloat16));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                    flatflow::ScalarType::int32),
            flatflow::promote_types(flatflow::ScalarType::int32,
                                    flatflow::ScalarType::bfloat16));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                    flatflow::ScalarType::int64),
            flatflow::promote_types(flatflow::ScalarType::int64,
                                    flatflow::ScalarType::bfloat16));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                    flatflow::ScalarType::uint8),
            flatflow::promote_types(flatflow::ScalarType::uint8,
                                    flatflow::ScalarType::bfloat16));

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bool_,
                                    flatflow::ScalarType::int8),
            flatflow::promote_types(flatflow::ScalarType::int8,
                                    flatflow::ScalarType::bool_));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bool_,
                                    flatflow::ScalarType::int16),
            flatflow::promote_types(flatflow::ScalarType::int16,
                                    flatflow::ScalarType::bool_));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bool_,
                                    flatflow::ScalarType::int32),
            flatflow::promote_types(flatflow::ScalarType::int32,
                                    flatflow::ScalarType::bool_));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bool_,
                                    flatflow::ScalarType::int64),
            flatflow::promote_types(flatflow::ScalarType::int64,
                                    flatflow::ScalarType::bool_));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bool_,
                                    flatflow::ScalarType::uint8),
            flatflow::promote_types(flatflow::ScalarType::uint8,
                                    flatflow::ScalarType::bool_));

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int8,
                                    flatflow::ScalarType::int16),
            flatflow::promote_types(flatflow::ScalarType::int16,
                                    flatflow::ScalarType::int8));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int8,
                                    flatflow::ScalarType::int32),
            flatflow::promote_types(flatflow::ScalarType::int32,
                                    flatflow::ScalarType::int8));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int8,
                                    flatflow::ScalarType::int64),
            flatflow::promote_types(flatflow::ScalarType::int64,
                                    flatflow::ScalarType::int8));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int8,
                                    flatflow::ScalarType::uint8),
            flatflow::promote_types(flatflow::ScalarType::uint8,
                                    flatflow::ScalarType::int8));

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int16,
                                    flatflow::ScalarType::int32),
            flatflow::promote_types(flatflow::ScalarType::int32,
                                    flatflow::ScalarType::int16));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int16,
                                    flatflow::ScalarType::int64),
            flatflow::promote_types(flatflow::ScalarType::int64,
                                    flatflow::ScalarType::int16));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int16,
                                    flatflow::ScalarType::uint8),
            flatflow::promote_types(flatflow::ScalarType::uint8,
                                    flatflow::ScalarType::int16));

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int32,
                                    flatflow::ScalarType::int64),
            flatflow::promote_types(flatflow::ScalarType::int64,
                                    flatflow::ScalarType::int32));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int32,
                                    flatflow::ScalarType::uint8),
            flatflow::promote_types(flatflow::ScalarType::uint8,
                                    flatflow::ScalarType::int32));

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int64,
                                    flatflow::ScalarType::uint8),
            flatflow::promote_types(flatflow::ScalarType::uint8,
                                    flatflow::ScalarType::int64));
}

TEST(PromoteTypesTest, MatchesC10Reference) {
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::float64),
            flatflow::ScalarType::float64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::float16),
            flatflow::ScalarType::float32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::bfloat16),
            flatflow::ScalarType::float32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::bool_),
            flatflow::ScalarType::float32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::int8),
            flatflow::ScalarType::float32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::int16),
            flatflow::ScalarType::float32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::int32),
            flatflow::ScalarType::float32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::int64),
            flatflow::ScalarType::float32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float32,
                                    flatflow::ScalarType::uint8),
            flatflow::ScalarType::float32);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::float16),
            flatflow::ScalarType::float64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::bfloat16),
            flatflow::ScalarType::float64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::bool_),
            flatflow::ScalarType::float64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::int8),
            flatflow::ScalarType::float64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::int16),
            flatflow::ScalarType::float64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::int32),
            flatflow::ScalarType::float64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::int64),
            flatflow::ScalarType::float64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float64,
                                    flatflow::ScalarType::uint8),
            flatflow::ScalarType::float64);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::bfloat16),
            flatflow::ScalarType::float32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::bool_),
            flatflow::ScalarType::float16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::int8),
            flatflow::ScalarType::float16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::int16),
            flatflow::ScalarType::float16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::int32),
            flatflow::ScalarType::float16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::int64),
            flatflow::ScalarType::float16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::float16,
                                    flatflow::ScalarType::uint8),
            flatflow::ScalarType::float16);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                    flatflow::ScalarType::bool_),
            flatflow::ScalarType::bfloat16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                    flatflow::ScalarType::int8),
            flatflow::ScalarType::bfloat16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                    flatflow::ScalarType::int16),
            flatflow::ScalarType::bfloat16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                    flatflow::ScalarType::int32),
            flatflow::ScalarType::bfloat16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                    flatflow::ScalarType::int64),
            flatflow::ScalarType::bfloat16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bfloat16,
                                    flatflow::ScalarType::uint8),
            flatflow::ScalarType::bfloat16);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bool_,
                                    flatflow::ScalarType::int8),
            flatflow::ScalarType::int8);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bool_,
                                    flatflow::ScalarType::int16),
            flatflow::ScalarType::int16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bool_,
                                    flatflow::ScalarType::int32),
            flatflow::ScalarType::int32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bool_,
                                    flatflow::ScalarType::int64),
            flatflow::ScalarType::int64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::bool_,
                                    flatflow::ScalarType::uint8),
            flatflow::ScalarType::uint8);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int8,
                                    flatflow::ScalarType::int16),
            flatflow::ScalarType::int16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int8,
                                    flatflow::ScalarType::int32),
            flatflow::ScalarType::int32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int8,
                                    flatflow::ScalarType::int64),
            flatflow::ScalarType::int64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int8,
                                    flatflow::ScalarType::uint8),
            flatflow::ScalarType::int16);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int16,
                                    flatflow::ScalarType::int32),
            flatflow::ScalarType::int32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int16,
                                    flatflow::ScalarType::int64),
            flatflow::ScalarType::int64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int16,
                                    flatflow::ScalarType::uint8),
            flatflow::ScalarType::int16);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int32,
                                    flatflow::ScalarType::int64),
            flatflow::ScalarType::int64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int32,
                                    flatflow::ScalarType::uint8),
            flatflow::ScalarType::int32);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::int64,
                                    flatflow::ScalarType::uint8),
            flatflow::ScalarType::int64);
}

}  // namespace
