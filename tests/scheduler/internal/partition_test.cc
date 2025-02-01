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
#include <cstdint>
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

  static constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 3);
  static constexpr auto kNumMicroBatches = static_cast<std::size_t>(1 << 12);
};

TEST_F(PartitionTest, BLDMWithEmptyItems) {
  const auto items = std::vector<std::pair<const uint32_t, uint64_t>>();

  const auto micro_batches = flatflow::internal::BLDM(
      items, items.size(), [](const auto &size) { return size; });
  EXPECT_TRUE(micro_batches.empty());
}

TEST_F(PartitionTest, BLDMWithGaltonIntegerDistribution) {
  auto distribution = std::lognormal_distribution(5.252, 0.293);
  auto generator = std::default_random_engine();

  auto pairs = std::vector<std::pair<uint32_t, uint64_t>>();
  pairs.reserve(kMicroBatchSize * kNumMicroBatches);

  while (pairs.size() < pairs.capacity()) {
    const auto size = distribution(generator);
    if (0.5 <= size && size < 8192.5) {
      const auto workload = std::lround(size);
      const auto index = pairs.size();
      pairs.emplace_back(workload, index);
    }
  }

  std::sort(pairs.begin(), pairs.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  auto items = std::vector<std::pair<const uint32_t, uint64_t>>();
  items.reserve(pairs.size());

  std::for_each(
      std::execution::seq, pairs.cbegin(), pairs.cend(),
      [&](const auto &pair) { items.emplace_back(pair.first, pair.second); });

  const auto micro_batches = flatflow::internal::BLDM(
      items, kNumMicroBatches, [](const auto &size) { return size; });
  EXPECT_TRUE(std::is_sorted(
      micro_batches.cbegin(), micro_batches.cend(),
      [](const auto &lhs, const auto &rhs) { return lhs.sum() < rhs.sum(); }));

  auto workloads = std::vector<uint32_t>();
  workloads.reserve(kNumMicroBatches);

  EXPECT_EQ(micro_batches.size(), kNumMicroBatches);
  std::for_each(micro_batches.cbegin(), micro_batches.cend(),
                [&](const auto &micro_batch) {
                  EXPECT_EQ(micro_batch.data().size(), kMicroBatchSize);
                  workloads.emplace_back(micro_batch.sum());
                });

  LOG(INFO) << absl::StrFormat("Workloads: %s", absl::StrJoin(workloads, " "));
}

TEST_F(PartitionTest, BLDMWithGaltonRealDistribution) {
  auto distribution = std::lognormal_distribution(5.252, 0.293);
  auto generator = std::default_random_engine();

  auto pairs = std::vector<std::pair<uint32_t, uint64_t>>();
  pairs.reserve(kMicroBatchSize * kNumMicroBatches);

  while (pairs.size() < pairs.capacity()) {
    const auto size = distribution(generator);
    if (0.5 <= size && size < 8192.5) {
      const auto workload = std::lround(size);
      const auto index = pairs.size();
      pairs.emplace_back(workload, index);
    }
  }

  std::sort(pairs.begin(), pairs.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  auto items = std::vector<std::pair<const uint32_t, uint64_t>>();
  items.reserve(pairs.size());

  std::for_each(
      std::execution::seq, pairs.cbegin(), pairs.cend(),
      [&](const auto &pair) { items.emplace_back(pair.first, pair.second); });

  const auto micro_batches = flatflow::internal::BLDM(
      items, kNumMicroBatches,
      [](const auto &size) { return static_cast<double>(size); });
  EXPECT_TRUE(std::is_sorted(
      micro_batches.cbegin(), micro_batches.cend(),
      [](const auto &lhs, const auto &rhs) { return lhs.sum() < rhs.sum(); }));

  auto workloads = std::vector<double>();
  workloads.reserve(kNumMicroBatches);

  EXPECT_EQ(micro_batches.size(), kNumMicroBatches);
  std::for_each(micro_batches.cbegin(), micro_batches.cend(),
                [&](const auto &micro_batch) {
                  EXPECT_EQ(micro_batch.data().size(), kMicroBatchSize);
                  workloads.emplace_back(micro_batch.sum());
                });

  LOG(INFO) << absl::StrFormat("Workloads: %s", absl::StrJoin(workloads, " "));
}

}  // namespace
