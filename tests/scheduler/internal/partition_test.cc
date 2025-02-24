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

#include "flatflow/scheduler/internal/partition.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <random>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/internal/globals.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "gtest/gtest.h"

namespace {

class PartitionTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!absl::log_internal::IsInitialized()) {
      absl::InitializeLog();
      absl::SetStderrThreshold(absl::LogSeverity::kInfo);
    }
  }

  static constexpr auto kMicroBatchSize = static_cast<size_t>(1 << 3);
  static constexpr auto kNumMicrobatches = static_cast<size_t>(1 << 12);
};

TEST_F(PartitionTest, BLDMWithEmptyItems) {
  auto items = std::vector<std::pair<uint32_t, size_t>>();
  auto subsets = std::vector<flatflow::internal::Subset<uint32_t, size_t>>();
  const auto result = flatflow::internal::Partition(
      items.begin(), items.end(), subsets.begin(),
      [](const auto &item) { return item.first; },
      [](const auto &item) { return item.second; }, 0);
  EXPECT_TRUE(subsets.empty());
  EXPECT_EQ(std::distance(result, subsets.end()), 0);
}

TEST_F(PartitionTest, BLDMWithGaltonIntegerDistribution) {
  auto distribution = std::lognormal_distribution(5.252, 0.293);
  auto generator = std::default_random_engine();

  auto items = std::vector<std::pair<uint32_t, size_t>>();
  items.reserve(kMicroBatchSize * kNumMicrobatches);

  while (items.size() < items.capacity()) {
    const auto size = distribution(generator);
    if (0.5 <= size && size < 8192.5) {
      const auto workload = std::lround(size);
      const auto index = items.size();
      items.emplace_back(workload, index);
    }
  }

  std::sort(items.begin(), items.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  auto subsets = std::vector<flatflow::internal::Subset<uint32_t, size_t>>(
      kNumMicrobatches);
  const auto result = flatflow::internal::Partition(
      items.begin(), items.end(), subsets.begin(),
      [](const auto &item) { return item.first; },
      [](const auto &item) { return item.second; }, kNumMicrobatches);
  EXPECT_EQ(subsets.size(), kNumMicrobatches);
  EXPECT_EQ(std::distance(result, subsets.end()), 0);

  EXPECT_TRUE(std::is_sorted(subsets.cbegin(), subsets.cend()));

  auto workloads = std::vector<uint32_t>();
  workloads.reserve(kNumMicrobatches);

  std::for_each(subsets.cbegin(), subsets.cend(), [&](const auto &subset) {
    EXPECT_EQ(subset.items().size(), kMicroBatchSize);
    workloads.emplace_back(subset.sum());
  });

  LOG(INFO) << absl::StrFormat("Workloads: %s", absl::StrJoin(workloads, " "));
}

TEST_F(PartitionTest, BLDMWithGaltonRealDistribution) {
  auto distribution = std::lognormal_distribution(5.252, 0.293);
  auto generator = std::default_random_engine();

  auto items = std::vector<std::pair<uint32_t, size_t>>();
  items.reserve(kMicroBatchSize * kNumMicrobatches);

  while (items.size() < items.capacity()) {
    const auto size = distribution(generator);
    if (0.5 <= size && size < 8192.5) {
      const auto workload = std::lround(size);
      const auto index = items.size();
      items.emplace_back(workload, index);
    }
  }

  std::sort(items.begin(), items.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  auto subsets =
      std::vector<flatflow::internal::Subset<double, size_t>>(kNumMicrobatches);
  const auto result = flatflow::internal::Partition(
      items.begin(), items.end(), subsets.begin(),
      [](const auto &item) { return static_cast<double>(item.first); },
      [](const auto &item) { return item.second; }, kNumMicrobatches);
  EXPECT_EQ(subsets.size(), kNumMicrobatches);
  EXPECT_EQ(std::distance(result, subsets.end()), 0);

  EXPECT_TRUE(std::is_sorted(subsets.cbegin(), subsets.cend()));

  auto workloads = std::vector<double>();
  workloads.reserve(kNumMicrobatches);

  std::for_each(subsets.cbegin(), subsets.cend(), [&](const auto &subset) {
    EXPECT_EQ(subset.items().size(), kMicroBatchSize);
    workloads.emplace_back(subset.sum());
  });

  LOG(INFO) << absl::StrFormat("Workloads: %s", absl::StrJoin(workloads, " "));
}

}  // namespace
