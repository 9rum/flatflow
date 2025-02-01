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
#include <cstdint>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "flatflow/scheduler/internal/partition.h"

namespace {

TEST(ShuffleTest, InterBatchShufflingWithIntegerWorkloads) {
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 3);
  constexpr auto kNumMicroBatches = static_cast<std::size_t>(1 << 12);
  constexpr auto kSeed = static_cast<uint64_t>(0);

  auto micro_batches =
      std::vector<flatflow::internal::Subset<uint64_t, uint32_t>>();
  micro_batches.reserve(kNumMicroBatches);

  for (std::size_t step = 0; step < kNumMicroBatches; ++step) {
    const auto sum = std::lround(std::log2(step + 2));
    auto data = std::vector<uint64_t>();
    data.reserve(kMicroBatchSize);
    for (std::size_t index = 0; index < kMicroBatchSize; ++index) {
      data.emplace_back(step * kMicroBatchSize + index);
    }
    micro_batches.emplace_back(sum, std::move(data));
  }

  flatflow::internal::shuffle(micro_batches, kSeed);

  EXPECT_EQ(micro_batches.size(), kNumMicroBatches);
  EXPECT_TRUE(std::is_sorted(
      micro_batches.cbegin(), micro_batches.cend(),
      [](const auto &lhs, const auto &rhs) { return lhs.sum() < rhs.sum(); }));

  std::for_each(micro_batches.cbegin(), micro_batches.cend(),
                [&](const auto &micro_batch) {
                  EXPECT_EQ(micro_batch.data().size(), kMicroBatchSize);
                });
}

TEST(ShuffleTest, InterBatchShufflingWithOneIntegerWorkload) {
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1);
  constexpr auto kNumMicroBatches = static_cast<std::size_t>(1);
  constexpr auto kSeed = static_cast<uint64_t>(0);

  auto micro_batches =
      std::vector<flatflow::internal::Subset<uint64_t, uint32_t>>();
  micro_batches.reserve(kNumMicroBatches);

  const auto sum = 0;
  auto data = std::vector<uint64_t>();
  data.reserve(kMicroBatchSize);
  data.emplace_back(0);
  micro_batches.emplace_back(sum, std::move(data));

  flatflow::internal::shuffle(micro_batches, kSeed);

  EXPECT_EQ(micro_batches.size(), kNumMicroBatches);
  EXPECT_EQ(micro_batches.front().data().size(), kMicroBatchSize);
}

TEST(ShuffleTest, InterBatchShufflingWithRealWorkloads) {
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 3);
  constexpr auto kNumMicroBatches = static_cast<std::size_t>(1 << 12);
  constexpr auto kSeed = static_cast<uint64_t>(0);

  auto micro_batches =
      std::vector<flatflow::internal::Subset<uint64_t, double>>();
  micro_batches.reserve(kNumMicroBatches);

  for (std::size_t step = 0; step < kNumMicroBatches; ++step) {
    const auto sum = std::round(std::log2(step + 2));
    auto data = std::vector<uint64_t>();
    data.reserve(kMicroBatchSize);
    for (std::size_t index = 0; index < kMicroBatchSize; ++index) {
      data.emplace_back(step * kMicroBatchSize + index);
    }
    micro_batches.emplace_back(sum, std::move(data));
  }

  flatflow::internal::shuffle(micro_batches, kSeed);

  EXPECT_EQ(micro_batches.size(), kNumMicroBatches);
  EXPECT_TRUE(std::is_sorted(
      micro_batches.cbegin(), micro_batches.cend(),
      [](const auto &lhs, const auto &rhs) { return lhs.sum() < rhs.sum(); }));

  std::for_each(micro_batches.cbegin(), micro_batches.cend(),
                [&](const auto &micro_batch) {
                  EXPECT_EQ(micro_batch.data().size(), kMicroBatchSize);
                });
}

TEST(ShuffleTest, InterBatchShufflingWithOneRealWorkload) {
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1);
  constexpr auto kNumMicroBatches = static_cast<std::size_t>(1);
  constexpr auto kSeed = static_cast<uint64_t>(0);

  auto micro_batches =
      std::vector<flatflow::internal::Subset<uint64_t, double>>();
  micro_batches.reserve(kNumMicroBatches);

  const auto sum = 0.0;
  auto data = std::vector<uint64_t>();
  data.reserve(kMicroBatchSize);
  data.emplace_back(0);
  micro_batches.emplace_back(sum, std::move(data));

  flatflow::internal::shuffle(micro_batches, kSeed);

  EXPECT_EQ(micro_batches.size(), kNumMicroBatches);
  EXPECT_EQ(micro_batches.front().data().size(), kMicroBatchSize);
}

TEST(ShuffleTest, InterBatchShufflingWithZeroWorkloads) {
  constexpr auto kSeed = static_cast<uint64_t>(0);

  auto micro_batches =
      std::vector<flatflow::internal::Subset<uint64_t, int64_t>>();

  flatflow::internal::shuffle(micro_batches, kSeed);

  EXPECT_TRUE(micro_batches.empty());
}

}  // namespace
