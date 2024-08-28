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

#include "flatflow/scheduler/internal/algorithm/shuffle.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace {

TEST(ShuffleTest, InterBatchShufflingWithIntegerMakespans) {
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 3);
  constexpr auto kNumMicroBatches = static_cast<std::size_t>(1 << 12);

  auto micro_batches = std::vector<
      std::pair<uint32_t, std::vector<std::pair<uint16_t, uint64_t>>>>();
  micro_batches.reserve(kNumMicroBatches);

  for (std::size_t step = 0; step < kNumMicroBatches; ++step) {
    auto micro_batch =
        std::pair<uint32_t, std::vector<std::pair<uint16_t, uint64_t>>>();
    micro_batch.first = static_cast<uint32_t>(std::lround(std::log2(step + 2)));
    micro_batch.second.reserve(kMicroBatchSize);
    for (std::size_t index = 0; index < kMicroBatchSize; ++index) {
      micro_batch.second.emplace_back(
          static_cast<uint16_t>(std::lround(std::log2(step + 2))),
          static_cast<uint64_t>(step * kMicroBatchSize + index));
    }
    micro_batches.emplace_back(std::move(micro_batch));
  }

  auto shuffled = flatflow::scheduler::internal::algorithm::shuffle(
      micro_batches, 0UL, false);

  EXPECT_EQ(shuffled.size(), kNumMicroBatches);

  for (const auto &micro_batch : shuffled) {
    EXPECT_EQ(micro_batch.size(), kMicroBatchSize);
  }
}

TEST(ShuffleTest, InterBatchShufflingWithOneIntegerMakespan) {
  auto micro_batch =
      std::vector<std::pair<uint16_t, uint64_t>>(1, std::make_pair(1, 0));
  auto micro_batches = std::vector<
      std::pair<uint32_t, std::vector<std::pair<uint16_t, uint64_t>>>>();
  micro_batches.emplace_back(0, std::move(micro_batch));

  auto shuffled = flatflow::scheduler::internal::algorithm::shuffle(
      micro_batches, 0UL, false);

  EXPECT_EQ(shuffled.size(), 1);
  EXPECT_EQ(shuffled.front().size(), 1);
}

TEST(ShuffleTest, InterBatchShufflingWithRealMakespans) {
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 3);
  constexpr auto kNumMicroBatches = static_cast<std::size_t>(1 << 12);

  auto micro_batches = std::vector<
      std::pair<double, std::vector<std::pair<uint16_t, uint64_t>>>>();
  micro_batches.reserve(kNumMicroBatches);

  for (std::size_t step = 0; step < kNumMicroBatches; ++step) {
    auto micro_batch =
        std::pair<double, std::vector<std::pair<uint16_t, uint64_t>>>();
    micro_batch.first = std::round(std::log2(step + 2));
    micro_batch.second.reserve(kMicroBatchSize);
    for (std::size_t index = 0; index < kMicroBatchSize; ++index) {
      micro_batch.second.emplace_back(
          static_cast<uint16_t>(std::lround(std::log2(step + 2))),
          static_cast<uint64_t>(step * kMicroBatchSize + index));
    }
    micro_batches.emplace_back(std::move(micro_batch));
  }

  auto shuffled = flatflow::scheduler::internal::algorithm::shuffle(
      micro_batches, 0UL, false);

  EXPECT_EQ(shuffled.size(), kNumMicroBatches);

  for (const auto &micro_batch : shuffled) {
    EXPECT_EQ(micro_batch.size(), kMicroBatchSize);
  }
}

TEST(ShuffleTest, InterBatchShufflingWithOneRealMakespan) {
  auto micro_batch =
      std::vector<std::pair<uint16_t, uint64_t>>(1, std::make_pair(1, 0));
  auto micro_batches = std::vector<
      std::pair<double, std::vector<std::pair<uint16_t, uint64_t>>>>();
  micro_batches.emplace_back(0.0, std::move(micro_batch));

  auto shuffled = flatflow::scheduler::internal::algorithm::shuffle(
      micro_batches, 0UL, false);

  EXPECT_EQ(shuffled.size(), 1);
  EXPECT_EQ(shuffled.front().size(), 1);
}

TEST(ShuffleTest, InterBatchShufflingWithFlatShuffle) {
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 3);
  constexpr auto kNumMicroBatches = static_cast<std::size_t>(1 << 12);

  auto micro_batches = std::vector<
      std::pair<double, std::vector<std::pair<uint16_t, uint64_t>>>>();
  micro_batches.reserve(kNumMicroBatches);

  for (std::size_t step = 0; step < kNumMicroBatches; ++step) {
    auto micro_batch =
        std::pair<double, std::vector<std::pair<uint16_t, uint64_t>>>();
    micro_batch.first = std::round(std::log2(step + 2));
    micro_batch.second.reserve(kMicroBatchSize);
    for (std::size_t index = 0; index < kMicroBatchSize; ++index) {
      micro_batch.second.emplace_back(
          static_cast<uint16_t>(std::lround(std::log2(step + 2))),
          static_cast<uint64_t>(step * kMicroBatchSize + index));
    }
    micro_batches.emplace_back(std::move(micro_batch));
  }

  auto shuffled = flatflow::scheduler::internal::algorithm::shuffle(
      micro_batches, 0UL, true);

  EXPECT_EQ(shuffled.size(), kNumMicroBatches);

  for (const auto &micro_batch : shuffled) {
    EXPECT_EQ(micro_batch.size(), kMicroBatchSize);
  }
}

}  // namespace
