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

#include "flatflow/ops/promote_types.h"

#include "gtest/gtest.h"

#include "flatflow/ops/scalar_type_generated.h"

namespace {

TEST(PromoteTypesTest, Identity) {
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT32,
                                    flatflow::ScalarType::FLOAT32),
            flatflow::ScalarType::FLOAT32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT64,
                                    flatflow::ScalarType::FLOAT64),
            flatflow::ScalarType::FLOAT64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT16,
                                    flatflow::ScalarType::FLOAT16),
            flatflow::ScalarType::FLOAT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::BFLOAT16,
                                    flatflow::ScalarType::BFLOAT16),
            flatflow::ScalarType::BFLOAT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::BOOL,
                                    flatflow::ScalarType::BOOL),
            flatflow::ScalarType::BOOL);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT8,
                                    flatflow::ScalarType::INT8),
            flatflow::ScalarType::INT8);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT16,
                                    flatflow::ScalarType::INT16),
            flatflow::ScalarType::INT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT32,
                                    flatflow::ScalarType::INT32),
            flatflow::ScalarType::INT32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT64,
                                    flatflow::ScalarType::INT64),
            flatflow::ScalarType::INT64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::UINT8,
                                    flatflow::ScalarType::UINT8),
            flatflow::ScalarType::UINT8);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::UINT16,
                                    flatflow::ScalarType::UINT16),
            flatflow::ScalarType::UINT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::UINT32,
                                    flatflow::ScalarType::UINT32),
            flatflow::ScalarType::UINT32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::UINT64,
                                    flatflow::ScalarType::UINT64),
            flatflow::ScalarType::UINT64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT8_E4M3FN,
                                    flatflow::ScalarType::FLOAT8_E4M3FN),
            flatflow::ScalarType::FLOAT8_E4M3FN);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT8_E4M3FNUZ,
                                    flatflow::ScalarType::FLOAT8_E4M3FNUZ),
            flatflow::ScalarType::FLOAT8_E4M3FNUZ);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT8_E5M2,
                                    flatflow::ScalarType::FLOAT8_E5M2),
            flatflow::ScalarType::FLOAT8_E5M2);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT8_E5M2FNUZ,
                                    flatflow::ScalarType::FLOAT8_E5M2FNUZ),
            flatflow::ScalarType::FLOAT8_E5M2FNUZ);
}

TEST(PromoteTypesTest, MatchesC10Reference) {
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT32,
                                    flatflow::ScalarType::FLOAT64),
            flatflow::ScalarType::FLOAT64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT32,
                                    flatflow::ScalarType::FLOAT16),
            flatflow::ScalarType::FLOAT32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT32,
                                    flatflow::ScalarType::BFLOAT16),
            flatflow::ScalarType::FLOAT32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT32,
                                    flatflow::ScalarType::BOOL),
            flatflow::ScalarType::FLOAT32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT32,
                                    flatflow::ScalarType::INT8),
            flatflow::ScalarType::FLOAT32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT32,
                                    flatflow::ScalarType::INT16),
            flatflow::ScalarType::FLOAT32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT32,
                                    flatflow::ScalarType::INT32),
            flatflow::ScalarType::FLOAT32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT32,
                                    flatflow::ScalarType::INT64),
            flatflow::ScalarType::FLOAT32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT32,
                                    flatflow::ScalarType::UINT8),
            flatflow::ScalarType::FLOAT32);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT64,
                                    flatflow::ScalarType::FLOAT16),
            flatflow::ScalarType::FLOAT64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT64,
                                    flatflow::ScalarType::BFLOAT16),
            flatflow::ScalarType::FLOAT64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT64,
                                    flatflow::ScalarType::BOOL),
            flatflow::ScalarType::FLOAT64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT64,
                                    flatflow::ScalarType::INT8),
            flatflow::ScalarType::FLOAT64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT64,
                                    flatflow::ScalarType::INT16),
            flatflow::ScalarType::FLOAT64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT64,
                                    flatflow::ScalarType::INT32),
            flatflow::ScalarType::FLOAT64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT64,
                                    flatflow::ScalarType::INT64),
            flatflow::ScalarType::FLOAT64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT64,
                                    flatflow::ScalarType::UINT8),
            flatflow::ScalarType::FLOAT64);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT16,
                                    flatflow::ScalarType::BFLOAT16),
            flatflow::ScalarType::FLOAT32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT16,
                                    flatflow::ScalarType::BOOL),
            flatflow::ScalarType::FLOAT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT16,
                                    flatflow::ScalarType::INT8),
            flatflow::ScalarType::FLOAT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT16,
                                    flatflow::ScalarType::INT16),
            flatflow::ScalarType::FLOAT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT16,
                                    flatflow::ScalarType::INT32),
            flatflow::ScalarType::FLOAT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT16,
                                    flatflow::ScalarType::INT64),
            flatflow::ScalarType::FLOAT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT16,
                                    flatflow::ScalarType::UINT8),
            flatflow::ScalarType::FLOAT16);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::BFLOAT16,
                                    flatflow::ScalarType::BOOL),
            flatflow::ScalarType::BFLOAT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::BFLOAT16,
                                    flatflow::ScalarType::INT8),
            flatflow::ScalarType::BFLOAT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::BFLOAT16,
                                    flatflow::ScalarType::INT16),
            flatflow::ScalarType::BFLOAT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::BFLOAT16,
                                    flatflow::ScalarType::INT32),
            flatflow::ScalarType::BFLOAT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::BFLOAT16,
                                    flatflow::ScalarType::INT64),
            flatflow::ScalarType::BFLOAT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::BFLOAT16,
                                    flatflow::ScalarType::UINT8),
            flatflow::ScalarType::BFLOAT16);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::BOOL,
                                    flatflow::ScalarType::INT8),
            flatflow::ScalarType::INT8);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::BOOL,
                                    flatflow::ScalarType::INT16),
            flatflow::ScalarType::INT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::BOOL,
                                    flatflow::ScalarType::INT32),
            flatflow::ScalarType::INT32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::BOOL,
                                    flatflow::ScalarType::INT64),
            flatflow::ScalarType::INT64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::BOOL,
                                    flatflow::ScalarType::UINT8),
            flatflow::ScalarType::UINT8);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT8,
                                    flatflow::ScalarType::INT16),
            flatflow::ScalarType::INT16);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT8,
                                    flatflow::ScalarType::INT32),
            flatflow::ScalarType::INT32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT8,
                                    flatflow::ScalarType::INT64),
            flatflow::ScalarType::INT64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT8,
                                    flatflow::ScalarType::UINT8),
            flatflow::ScalarType::INT16);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT16,
                                    flatflow::ScalarType::INT32),
            flatflow::ScalarType::INT32);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT16,
                                    flatflow::ScalarType::INT64),
            flatflow::ScalarType::INT64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT16,
                                    flatflow::ScalarType::UINT8),
            flatflow::ScalarType::INT16);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT32,
                                    flatflow::ScalarType::INT64),
            flatflow::ScalarType::INT64);
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT32,
                                    flatflow::ScalarType::UINT8),
            flatflow::ScalarType::INT32);

  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT64,
                                    flatflow::ScalarType::UINT8),
            flatflow::ScalarType::INT64);
}

TEST(PromoteTypesTest, Commutativity) {
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT32,
                                    flatflow::ScalarType::FLOAT64),
            flatflow::promote_types(flatflow::ScalarType::FLOAT64,
                                    flatflow::ScalarType::FLOAT32));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::FLOAT16,
                                    flatflow::ScalarType::BFLOAT16),
            flatflow::promote_types(flatflow::ScalarType::BFLOAT16,
                                    flatflow::ScalarType::FLOAT16));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::BOOL,
                                    flatflow::ScalarType::INT8),
            flatflow::promote_types(flatflow::ScalarType::INT8,
                                    flatflow::ScalarType::BOOL));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT16,
                                    flatflow::ScalarType::INT32),
            flatflow::promote_types(flatflow::ScalarType::INT32,
                                    flatflow::ScalarType::INT16));
  EXPECT_EQ(flatflow::promote_types(flatflow::ScalarType::INT64,
                                    flatflow::ScalarType::UINT8),
            flatflow::promote_types(flatflow::ScalarType::UINT8,
                                    flatflow::ScalarType::INT64));
}

}  // namespace
