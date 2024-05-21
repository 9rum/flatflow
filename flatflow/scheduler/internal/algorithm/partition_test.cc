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

#include "flatflow/scheduler/internal/algorithm/partition.h"

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace {

TEST(PartitionTest, KarmarkarKarpWithUniformIntegerDistribution) {
  constexpr auto kMinSize = static_cast<uint16_t>(1);
  constexpr auto kMaxSize = static_cast<uint16_t>(1 << 12);
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 2);
  constexpr auto kNumMicroBatches = static_cast<uint64_t>(1 << 8);

  auto distribution = std::uniform_int_distribution(kMinSize, kMaxSize);
  auto generator = std::default_random_engine();

  auto items = std::vector<std::pair<uint16_t, uint64_t>>();
  items.reserve(kMicroBatchSize * static_cast<std::size_t>(kNumMicroBatches));

  for (std::size_t index = 0; index < items.capacity(); ++index) {
    items.emplace_back(distribution(generator), index);
  }
  std::sort(items.begin(), items.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  const auto micro_batches =
      flatflow::scheduler::internal::algorithm::KarmarkarKarp(
          items, kNumMicroBatches,
          [](const auto &size) { return static_cast<uint32_t>(size); });
  EXPECT_EQ(micro_batches.size(), static_cast<std::size_t>(kNumMicroBatches));

  std::cout << "makespans: ";
  std::for_each(micro_batches.cbegin(), micro_batches.cend(),
                [&](const auto &micro_batch) {
                  EXPECT_EQ(micro_batch.second.size(), kMicroBatchSize);
                  std::cout << micro_batch.first << ' ';
                });
  std::cout << '\n';
}

}  // namespace
