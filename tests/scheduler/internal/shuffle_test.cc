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

#include "flatflow/scheduler/internal/shuffle.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "flatflow/scheduler/internal/partition.h"

namespace {

TEST(ShuffleTest, InterBatchShufflingWithIntegerWorkloads) {
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 3);
  constexpr auto kNumMicroBatches = static_cast<std::size_t>(1 << 12);
  constexpr auto kSeed = static_cast<uint_fast32_t>(0);

  auto subsets = std::vector<flatflow::internal::Subset<uint32_t, size_t>>();
  subsets.reserve(kNumMicroBatches);

  for (std::size_t step = 0; step < kNumMicroBatches; ++step) {
    const auto sum = std::lround(std::log2(step + 2));
    auto items = std::vector<size_t>(kMicroBatchSize);
    std::iota(items.begin(), items.end(), step * kMicroBatchSize);
    subsets.emplace_back(sum, std::move(items));
  }

  flatflow::internal::Shuffle(subsets.begin(), subsets.end(), kSeed);

  EXPECT_EQ(subsets.size(), kNumMicroBatches);
  EXPECT_TRUE(std::is_sorted(subsets.cbegin(), subsets.cend()));

  std::for_each(subsets.cbegin(), subsets.cend(), [&](const auto &subset) {
    EXPECT_EQ(subset.items().size(), kMicroBatchSize);
  });
}

TEST(ShuffleTest, InterBatchShufflingWithOneIntegerWorkload) {
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1);
  constexpr auto kNumMicroBatches = static_cast<std::size_t>(1);
  constexpr auto kSeed = static_cast<uint_fast32_t>(0);

  auto subsets = std::vector<flatflow::internal::Subset<uint32_t, size_t>>();
  subsets.reserve(kNumMicroBatches);

  const auto sum = 0;
  auto items = std::vector<size_t>();
  items.reserve(kMicroBatchSize);
  items.emplace_back(0);
  subsets.emplace_back(sum, std::move(items));

  flatflow::internal::Shuffle(subsets.begin(), subsets.end(), kSeed);

  EXPECT_EQ(subsets.size(), kNumMicroBatches);
  EXPECT_EQ(subsets.front().items().size(), kMicroBatchSize);
}

TEST(ShuffleTest, InterBatchShufflingWithRealWorkloads) {
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 3);
  constexpr auto kNumMicroBatches = static_cast<std::size_t>(1 << 12);
  constexpr auto kSeed = static_cast<uint_fast32_t>(0);

  auto subsets = std::vector<flatflow::internal::Subset<double, size_t>>();
  subsets.reserve(kNumMicroBatches);

  for (std::size_t step = 0; step < kNumMicroBatches; ++step) {
    const auto sum = std::round(std::log2(step + 2));
    auto items = std::vector<size_t>(kMicroBatchSize);
    std::iota(items.begin(), items.end(), step * kMicroBatchSize);
    subsets.emplace_back(sum, std::move(items));
  }

  flatflow::internal::Shuffle(subsets.begin(), subsets.end(), kSeed);

  EXPECT_EQ(subsets.size(), kNumMicroBatches);
  EXPECT_TRUE(std::is_sorted(subsets.cbegin(), subsets.cend()));

  std::for_each(subsets.cbegin(), subsets.cend(), [&](const auto &subset) {
    EXPECT_EQ(subset.items().size(), kMicroBatchSize);
  });
}

TEST(ShuffleTest, InterBatchShufflingWithOneRealWorkload) {
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1);
  constexpr auto kNumMicroBatches = static_cast<std::size_t>(1);
  constexpr auto kSeed = static_cast<uint_fast32_t>(0);

  auto subsets = std::vector<flatflow::internal::Subset<double, size_t>>();
  subsets.reserve(kNumMicroBatches);

  const auto sum = 0.0;
  auto items = std::vector<size_t>();
  items.reserve(kMicroBatchSize);
  items.emplace_back(0);
  subsets.emplace_back(sum, std::move(items));

  flatflow::internal::Shuffle(subsets.begin(), subsets.end(), kSeed);

  EXPECT_EQ(subsets.size(), kNumMicroBatches);
  EXPECT_EQ(subsets.front().items().size(), kMicroBatchSize);
}

TEST(ShuffleTest, InterBatchShufflingWithZeroWorkloads) {
  constexpr auto kSeed = static_cast<uint_fast32_t>(0);

  auto subsets = std::vector<flatflow::internal::Subset<uint32_t, size_t>>();

  flatflow::internal::Shuffle(subsets.begin(), subsets.end(), kSeed);

  EXPECT_TRUE(subsets.empty());
}

}  // namespace
