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

#include "flatflow/scheduler/internal/scatter.h"

#include <cstddef>
#include <iterator>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

namespace {

TEST(ScatterTest, ScatterWithoutRemainder) {
  constexpr auto kN = static_cast<size_t>(1 << 3);
  constexpr auto kRank = static_cast<size_t>(1 << 0);
  constexpr auto kStride = static_cast<size_t>(1 << 6);
  constexpr auto kTotalSize = static_cast<size_t>(1 << 8);

  auto from = std::vector<size_t>(kTotalSize);
  std::iota(from.begin(), from.end(), 0);

  auto to = std::vector<size_t>(kTotalSize / kN);
  const auto result = flatflow::internal::Scatter(
      from.begin(), from.end(), to.begin(), kN, kRank, kStride);
  EXPECT_EQ(std::distance(result, to.end()), 0);

  EXPECT_EQ(to, std::vector<size_t>({8,   9,   10,  11,  12,  13,  14,  15,
                                     72,  73,  74,  75,  76,  77,  78,  79,
                                     136, 137, 138, 139, 140, 141, 142, 143,
                                     200, 201, 202, 203, 204, 205, 206, 207}));
}

TEST(ScatterTest, ScatterWithRemainder) {
  constexpr auto kN = static_cast<size_t>(1 << 3);
  constexpr auto kRank = static_cast<size_t>(1 << 1);
  constexpr auto kStride = static_cast<size_t>(1 << 6);
  constexpr auto kTotalSize = static_cast<size_t>(17 << 4);

  auto from = std::vector<size_t>(kTotalSize);
  std::iota(from.begin(), from.end(), 0);

  auto to = std::vector<size_t>(kTotalSize / kN);
  const auto result = flatflow::internal::Scatter(
      from.begin(), from.end(), to.begin(), kN, kRank, kStride);
  EXPECT_EQ(std::distance(result, to.end()), 0);

  EXPECT_EQ(to, std::vector<size_t>(
                    {16,  17,  18,  19,  20,  21,  22,  23,  80,  81,  82,  83,
                     84,  85,  86,  87,  144, 145, 146, 147, 148, 149, 150, 151,
                     208, 209, 210, 211, 212, 213, 214, 215, 260, 261}));
}

TEST(ScatterTest, ScatterWithRemainderOnly) {
  constexpr auto kN = static_cast<size_t>(1 << 3);
  constexpr auto kRank = static_cast<size_t>(1 << 1);
  constexpr auto kStride = static_cast<size_t>(1 << 5);
  constexpr auto kTotalSize = static_cast<size_t>(1 << 4);

  auto from = std::vector<size_t>(kTotalSize);
  std::iota(from.begin(), from.end(), 0);

  auto to = std::vector<size_t>(kTotalSize / kN);
  const auto result = flatflow::internal::Scatter(
      from.begin(), from.end(), to.begin(), kN, kRank, kStride);
  EXPECT_EQ(std::distance(result, to.end()), 0);

  EXPECT_EQ(to, std::vector<size_t>({4, 5}));
}

}  // namespace
