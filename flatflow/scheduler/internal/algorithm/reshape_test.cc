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

#include "flatflow/scheduler/internal/algorithm/reshape.h"

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

namespace {

TEST(ReshapeTest, RegularTensor) {
  const std::vector<std::vector<std::vector<uint64_t>>> tensor = {
      {
          {0, 1, 2, 3},
          {4, 5, 6, 7},
      },
      {
          {8, 9, 10, 11},
          {12, 13, 14, 15},
      },
      {
          {16, 17, 18, 19},
          {20, 21, 22, 23},
      },
      {
          {24, 25, 26, 27},
          {28, 29, 30, 31},
      },
      {
          {32, 33, 34, 35},
          {36, 37, 38, 39},
      },
      {
          {40, 41, 42, 43},
          {44, 45, 46, 47},
      },
      {
          {48, 49, 50, 51},
          {52, 53, 54, 55},
      },
      {
          {56, 57, 58, 59},
          {60, 61, 62, 63},
      },
  };

  const std::vector<std::vector<uint64_t>> matrix = {
      {0,  1,  2,  3,  8,  9,  10, 11, 16, 17, 18, 19, 24, 25, 26, 27,
       32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59},
      {4,  5,  6,  7,  12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31,
       36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63},
  };

  const auto reshaped =
      flatflow::scheduler::internal::algorithm::reshape(tensor);

  EXPECT_EQ(matrix.size(), reshaped.size());
  EXPECT_EQ(matrix.at(0).size(), reshaped.at(0).size());
  EXPECT_EQ(matrix.at(1).size(), reshaped.at(1).size());

  EXPECT_TRUE(std::equal(matrix.at(0).cbegin(), matrix.at(0).cend(),
                         reshaped.at(0).cbegin()));
  EXPECT_TRUE(std::equal(matrix.at(1).cbegin(), matrix.at(1).cend(),
                         reshaped.at(1).cbegin()));
}

TEST(ReshapeTest, IrregularTensor) {
  const std::vector<std::vector<std::vector<uint64_t>>> tensor = {
      {
          {0, 1, 2, 3},
          {4, 5, 6, 7},
      },
      {
          {8, 9, 10, 11},
          {12, 13, 14, 15},
      },
      {
          {16, 17, 18, 19},
          {20, 21, 22, 23},
      },
      {
          {24, 25, 26, 27},
          {28, 29, 30, 31},
      },
      {
          {32, 33, 34, 35},
          {36, 37, 38, 39},
      },
      {
          {40, 41, 42, 43},
          {44, 45, 46, 47},
      },
      {
          {48, 49, 50, 51},
          {52, 53, 54, 55},
      },
      {
          {56, 57, 58},
          {59, 60, 61},
      },
  };

  const std::vector<std::vector<uint64_t>> matrix = {
      {0,  1,  2,  3,  8,  9,  10, 11, 16, 17, 18, 19, 24, 25, 26, 27,
       32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58},
      {4,  5,  6,  7,  12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31,
       36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 59, 60, 61},
  };

  const auto reshaped =
      flatflow::scheduler::internal::algorithm::reshape(tensor);

  EXPECT_EQ(matrix.size(), reshaped.size());
  EXPECT_EQ(matrix.at(0).size(), reshaped.at(0).size());
  EXPECT_EQ(matrix.at(1).size(), reshaped.at(1).size());

  EXPECT_TRUE(std::equal(matrix.at(0).cbegin(), matrix.at(0).cend(),
                         reshaped.at(0).cbegin()));
  EXPECT_TRUE(std::equal(matrix.at(1).cbegin(), matrix.at(1).cend(),
                         reshaped.at(1).cbegin()));
}

}  // namespace
