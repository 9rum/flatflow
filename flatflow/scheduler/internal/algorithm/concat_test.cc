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

#include "flatflow/scheduler/internal/algorithm/concat.h"

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

namespace {

TEST(ConcatTest, HandleRegularMatrices) {
  auto lhs = std::vector<std::vector<std::size_t>>(
      {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
       {10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
       {20, 21, 22, 23, 24, 25, 26, 27, 28, 29},
       {30, 31, 32, 33, 34, 35, 36, 37, 38, 39}});

  const auto rhs =
      std::vector<std::vector<std::size_t>>({{40, 41, 42, 43, 44},
                                             {45, 46, 47, 48, 49},
                                             {50, 51, 52, 53, 54},
                                             {55, 56, 57, 58, 59}});

  const auto concatenated = std::vector<std::vector<std::size_t>>(
      {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 41, 42, 43, 44},
       {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 45, 46, 47, 48, 49},
       {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 50, 51, 52, 53, 54},
       {30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 55, 56, 57, 58, 59}});

  flatflow::scheduler::internal::algorithm::concat(lhs, rhs);

  EXPECT_EQ(lhs.size(), concatenated.size());
  for (std::size_t rank = 0; rank < concatenated.size(); ++rank) {
    EXPECT_EQ(lhs[rank].size(), concatenated[rank].size());
    EXPECT_TRUE(std::equal(lhs[rank].cbegin(), lhs[rank].cend(),
                           concatenated[rank].cbegin()));
  }
}

TEST(ConcatTest, HandleRegularMatrixWithIrregularMatrix) {
  auto lhs = std::vector<std::vector<std::size_t>>(
      {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
       {10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
       {20, 21, 22, 23, 24, 25, 26, 27, 28, 29},
       {30, 31, 32, 33, 34, 35, 36, 37, 38, 39}});

  const auto rhs =
      std::vector<std::vector<std::size_t>>({{40, 41, 42, 43, 44, 45},
                                             {46, 47, 48, 49, 50},
                                             {51, 52, 53, 54, 55},
                                             {56, 57, 58, 59}});

  const auto concatenated = std::vector<std::vector<std::size_t>>(
      {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 41, 42, 43, 44, 45},
       {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 46, 47, 48, 49, 50},
       {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 51, 52, 53, 54, 55},
       {30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 56, 57, 58, 59}});

  flatflow::scheduler::internal::algorithm::concat(lhs, rhs);

  EXPECT_EQ(lhs.size(), concatenated.size());
  for (std::size_t rank = 0; rank < concatenated.size(); ++rank) {
    EXPECT_EQ(lhs[rank].size(), concatenated[rank].size());
    EXPECT_TRUE(std::equal(lhs[rank].cbegin(), lhs[rank].cend(),
                           concatenated[rank].cbegin()));
  }
}

TEST(ConcatTest, HandleIrregularMatrixWithRegularMatrix) {
  auto lhs = std::vector<std::vector<std::size_t>>(
      {{0, 1, 2, 3, 4, 5, 6, 7},
       {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
       {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30},
       {31, 32, 33, 34, 35, 36, 37, 38, 39}});

  const auto rhs =
      std::vector<std::vector<std::size_t>>({{40, 41, 42, 43, 44},
                                             {45, 46, 47, 48, 49},
                                             {50, 51, 52, 53, 54},
                                             {55, 56, 57, 58, 59}});

  const auto concatenated = std::vector<std::vector<std::size_t>>(
      {{0, 1, 2, 3, 4, 5, 6, 7, 40, 41, 42, 43, 44},
       {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 45, 46, 47, 48, 49},
       {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 50, 51, 52, 53, 54},
       {31, 32, 33, 34, 35, 36, 37, 38, 39, 55, 56, 57, 58, 59}});

  flatflow::scheduler::internal::algorithm::concat(lhs, rhs);

  EXPECT_EQ(lhs.size(), concatenated.size());
  for (std::size_t rank = 0; rank < concatenated.size(); ++rank) {
    EXPECT_EQ(lhs[rank].size(), concatenated[rank].size());
    EXPECT_TRUE(std::equal(lhs[rank].cbegin(), lhs[rank].cend(),
                           concatenated[rank].cbegin()));
  }
}

TEST(ConcatTest, HandleIrregularMatrices) {
  auto lhs = std::vector<std::vector<std::size_t>>(
      {{0, 1, 2, 3, 4, 5, 6, 7},
       {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
       {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30},
       {31, 32, 33, 34, 35, 36, 37, 38, 39}});

  const auto rhs =
      std::vector<std::vector<std::size_t>>({{40, 41, 42, 43, 44, 45},
                                             {46, 47, 48, 49, 50},
                                             {51, 52, 53, 54, 55},
                                             {56, 57, 58, 59}});

  const auto concatenated = std::vector<std::vector<std::size_t>>(
      {{0, 1, 2, 3, 4, 5, 6, 7, 40, 41, 42, 43, 44, 45},
       {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 46, 47, 48, 49, 50},
       {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 51, 52, 53, 54, 55},
       {31, 32, 33, 34, 35, 36, 37, 38, 39, 56, 57, 58, 59}});

  flatflow::scheduler::internal::algorithm::concat(lhs, rhs);

  EXPECT_EQ(lhs.size(), concatenated.size());
  for (std::size_t rank = 0; rank < concatenated.size(); ++rank) {
    EXPECT_EQ(lhs[rank].size(), concatenated[rank].size());
    EXPECT_TRUE(std::equal(lhs[rank].cbegin(), lhs[rank].cend(),
                           concatenated[rank].cbegin()));
  }
}

}  // namespace
