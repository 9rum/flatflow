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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "flatflow/scheduler/internal/partition.h"

namespace {

TEST(ScatterTest, ScatterWithEmptySubsets) {
  constexpr auto kGlobalBatchSize = static_cast<std::size_t>(1 << 2);
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 2);
  constexpr auto kWorldSize = static_cast<std::size_t>(1 << 2);

  auto subsets = std::vector<flatflow::internal::Subset<uint32_t, size_t>>();
  auto batches = std::vector<flatflow::internal::Subset<
      uint32_t, flatflow::internal::Subset<uint32_t, size_t>>>();
  const auto result = flatflow::internal::Scatter(
      subsets.begin(), subsets.end(), batches.begin(),
      [](auto subset) { return subset; },
      [](auto subset) { return subset.sum(); }, kWorldSize,
      kGlobalBatchSize / kMicroBatchSize);
  EXPECT_TRUE(batches.empty());
  EXPECT_EQ(std::distance(result, batches.end()), 0);
}

TEST(ScatterTest, ScatterWithoutRemainder) {
  constexpr auto kGlobalBatchSize = static_cast<std::size_t>(1 << 9);
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 3);
  constexpr auto kNumBatches = static_cast<std::size_t>(1 << 8);
  constexpr auto kNumMicroBatches = static_cast<std::size_t>(1 << 12);
  constexpr auto kWorldSize = static_cast<std::size_t>(1 << 2);

  auto subsets = std::vector<flatflow::internal::Subset<uint32_t, size_t>>();
  subsets.reserve(kNumMicroBatches);

  for (std::size_t step = 0; step < kNumMicroBatches; ++step) {
    const auto sum = std::lround(std::log2(step + 2));
    auto items = std::vector<size_t>(kMicroBatchSize);
    std::iota(items.begin(), items.end(), step * kMicroBatchSize);
    subsets.emplace_back(sum, std::move(items));
  }

  auto batches = std::vector<flatflow::internal::Subset<
      uint32_t, flatflow::internal::Subset<uint32_t, size_t>>>(kNumBatches);
  const auto result = flatflow::internal::Scatter(
      subsets.begin(), subsets.end(), batches.begin(),
      [](auto subset) { return subset; },
      [](auto subset) { return subset.sum(); }, kWorldSize,
      kGlobalBatchSize / kMicroBatchSize);
  EXPECT_EQ(batches.size(), kNumBatches);
  EXPECT_EQ(std::distance(result, batches.end()), 0);

  std::for_each(batches.cbegin(), batches.cend(), [&](const auto &batch) {
    EXPECT_EQ(batch.items().size(),
              kGlobalBatchSize / kMicroBatchSize / kWorldSize);
    EXPECT_TRUE(std::is_sorted(batch.items().cbegin(), batch.items().cend()));
  });
}

TEST(ScatterTest, ScatterWithRemainder) {
  constexpr auto kGlobalBatchSize = static_cast<std::size_t>(3 << 8);
  constexpr auto kLastGlobalBatchSize = static_cast<std::size_t>(1 << 9);
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 3);
  constexpr auto kNumBatches = static_cast<std::size_t>(43 << 2);
  constexpr auto kNumMicroBatches = static_cast<std::size_t>(1 << 12);
  constexpr auto kWorldSize = static_cast<std::size_t>(1 << 2);

  auto subsets = std::vector<flatflow::internal::Subset<uint32_t, size_t>>();
  subsets.reserve(kNumMicroBatches);

  for (std::size_t step = 0; step < kNumMicroBatches; ++step) {
    const auto sum = std::lround(std::log2(step + 2));
    auto items = std::vector<size_t>(kMicroBatchSize);
    std::iota(items.begin(), items.end(), step * kMicroBatchSize);
    subsets.emplace_back(sum, std::move(items));
  }

  auto batches = std::vector<flatflow::internal::Subset<
      uint32_t, flatflow::internal::Subset<uint32_t, size_t>>>(kNumBatches);
  const auto result = flatflow::internal::Scatter(
      subsets.begin(), subsets.end(), batches.begin(),
      [](auto subset) { return subset; },
      [](auto subset) { return subset.sum(); }, kWorldSize,
      kGlobalBatchSize / kMicroBatchSize);
  EXPECT_EQ(batches.size(), kNumBatches);
  EXPECT_EQ(std::distance(result, batches.end()), 0);

  std::for_each(batches.cbegin(), std::prev(batches.cend(), kWorldSize),
                [&](const auto &batch) {
                  EXPECT_EQ(batch.items().size(),
                            kGlobalBatchSize / kMicroBatchSize / kWorldSize);
                  EXPECT_TRUE(std::is_sorted(batch.items().cbegin(),
                                             batch.items().cend()));
                });
  std::for_each(
      std::prev(batches.cend(), kWorldSize), batches.cend(),
      [&](const auto &batch) {
        EXPECT_EQ(batch.items().size(),
                  kLastGlobalBatchSize / kMicroBatchSize / kWorldSize);
        EXPECT_TRUE(
            std::is_sorted(batch.items().cbegin(), batch.items().cend()));
      });
}

}  // namespace
