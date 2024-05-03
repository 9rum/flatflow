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

TEST(ShuffleTest, RegularTensor) {
  constexpr auto kInterval = static_cast<std::size_t>(1 << 10);
  constexpr auto kWorldSize = static_cast<std::size_t>(1 << 6);
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 4);

  auto tensor = std::vector<std::vector<std::vector<uint64_t>>>();
  tensor.reserve(kInterval);

  for (std::size_t step = 0; step < kInterval; ++step) {
    auto batch = std::vector<std::vector<uint64_t>>();
    batch.reserve(kWorldSize);
    for (std::size_t rank = 0; rank < kWorldSize; ++rank) {
      batch.emplace_back(std::move(std::vector<uint64_t>(kMicroBatchSize)));
    }
    tensor.emplace_back(std::move(batch));
  }

  flatflow::scheduler::internal::algorithm::shuffle(tensor, 0UL);

  EXPECT_EQ(tensor.size(), kInterval);

  for (std::size_t step = 0; step < kInterval; ++step) {
    EXPECT_EQ(tensor.at(step).size(), kWorldSize);
    for (std::size_t rank = 0; rank < kWorldSize; ++rank) {
      EXPECT_EQ(tensor.at(step).at(rank).size(), kMicroBatchSize);
    }
  }
}

TEST(ShuffleTest, IrregularTensor) {
  constexpr auto kInterval = static_cast<std::size_t>(1 << 10);
  constexpr auto kWorldSize = static_cast<std::size_t>(1 << 6);
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 4);
  constexpr auto kLastMicroBatchSize = static_cast<std::size_t>(1 << 3);

  auto tensor = std::vector<std::vector<std::vector<uint64_t>>>();
  tensor.reserve(kInterval);

  for (std::size_t step = 0; step < kInterval - 1; ++step) {
    auto batch = std::vector<std::vector<uint64_t>>();
    batch.reserve(kWorldSize);
    for (std::size_t rank = 0; rank < kWorldSize; ++rank) {
      batch.emplace_back(std::move(std::vector<uint64_t>(kMicroBatchSize)));
    }
    tensor.emplace_back(std::move(batch));
  }

  auto batch = std::vector<std::vector<uint64_t>>();
  batch.reserve(kWorldSize);
  for (std::size_t rank = 0; rank < kWorldSize; ++rank) {
    batch.emplace_back(std::move(std::vector<uint64_t>(kLastMicroBatchSize)));
  }
  tensor.emplace_back(std::move(batch));

  flatflow::scheduler::internal::algorithm::shuffle(tensor, 0UL);

  EXPECT_EQ(tensor.size(), kInterval);

  for (std::size_t step = 0; step < kInterval - 1; ++step) {
    EXPECT_EQ(tensor.at(step).size(), kWorldSize);
    for (std::size_t rank = 0; rank < kWorldSize; ++rank) {
      EXPECT_EQ(tensor.at(step).at(rank).size(), kMicroBatchSize);
    }
  }

  EXPECT_EQ(tensor.at(kInterval - 1).size(), kWorldSize);
  for (std::size_t rank = 0; rank < kWorldSize; ++rank) {
    EXPECT_EQ(tensor.at(kInterval - 1).at(rank).size(), kLastMicroBatchSize);
  }
}

}  // namespace
