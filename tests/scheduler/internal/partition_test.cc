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
#include <numeric>
#include <random>
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
  static constexpr auto kNumMicrobatches = static_cast<std::size_t>(1 << 12);
};

TEST_F(PartitionTest, BLDMWithEmptyItems) {
  auto indices = std::vector<std::size_t>();

  auto projs = std::vector<std::uint32_t>();

  auto subsets =
      std::vector<flatflow::internal::Subset<std::size_t, std::uint32_t>>();
  const auto result = flatflow::internal::Partition(
      indices.begin(), indices.end(), subsets.begin(), 0,
      [&](auto index) { return projs[index]; });

  EXPECT_TRUE(subsets.empty());
  EXPECT_EQ(std::distance(result, subsets.end()), 0);
}

TEST_F(PartitionTest, BLDMWithGaltonIntegerDistribution) {
  auto distribution = std::lognormal_distribution(5.252, 0.293);
  auto generator = std::default_random_engine();

  auto indices = std::vector<std::size_t>(kMicroBatchSize * kNumMicrobatches);
  std::iota(indices.begin(), indices.end(), 0);

  auto projs = std::vector<std::uint32_t>();
  projs.reserve(kMicroBatchSize * kNumMicrobatches);

  while (projs.size() < projs.capacity()) {
    const auto proj = distribution(generator);
    if (0.5 <= proj && proj < 8192.5) {
      projs.emplace_back(std::lround(proj));
    }
  }

  std::sort(projs.begin(), projs.end());

  auto subsets =
      std::vector<flatflow::internal::Subset<std::size_t, std::uint32_t>>(
          kNumMicrobatches);
  const auto result = flatflow::internal::Partition(
      indices.begin(), indices.end(), subsets.begin(), kNumMicrobatches,
      [&](auto index) { return projs[index]; });

  EXPECT_EQ(subsets.size(), kNumMicrobatches);
  EXPECT_EQ(std::distance(result, subsets.end()), 0);

  EXPECT_TRUE(std::is_sorted(subsets.cbegin(), subsets.cend()));

  projs = std::vector<std::uint32_t>();
  projs.reserve(kNumMicrobatches);

  std::for_each(subsets.cbegin(), subsets.cend(), [&](const auto &subset) {
    EXPECT_EQ(subset.items().size(), kMicroBatchSize);
    projs.emplace_back(subset.sum());
  });

  LOG(INFO) << absl::StrFormat("Subset sums: %s", absl::StrJoin(projs, " "));
}

TEST_F(PartitionTest, BLDMWithGaltonRealDistribution) {
  auto distribution = std::lognormal_distribution(5.252, 0.293);
  auto generator = std::default_random_engine();

  auto indices = std::vector<std::size_t>(kMicroBatchSize * kNumMicrobatches);
  std::iota(indices.begin(), indices.end(), 0);

  auto projs = std::vector<double>();
  projs.reserve(kMicroBatchSize * kNumMicrobatches);

  while (projs.size() < projs.capacity()) {
    const auto proj = distribution(generator);
    if (0.5 <= proj && proj < 8192.5) {
      projs.emplace_back(std::round(proj));
    }
  }

  std::sort(projs.begin(), projs.end());

  auto subsets = std::vector<flatflow::internal::Subset<std::size_t, double>>(
      kNumMicrobatches);
  const auto result = flatflow::internal::Partition(
      indices.begin(), indices.end(), subsets.begin(), kNumMicrobatches,
      [&](auto index) { return projs[index]; });

  EXPECT_EQ(subsets.size(), kNumMicrobatches);
  EXPECT_EQ(std::distance(result, subsets.end()), 0);

  EXPECT_TRUE(std::is_sorted(subsets.cbegin(), subsets.cend()));

  projs = std::vector<double>();
  projs.reserve(kNumMicrobatches);

  std::for_each(subsets.cbegin(), subsets.cend(), [&](const auto &subset) {
    EXPECT_EQ(subset.items().size(), kMicroBatchSize);
    projs.emplace_back(subset.sum());
  });

  LOG(INFO) << absl::StrFormat("Subset sums: %s", absl::StrJoin(projs, " "));
}

}  // namespace
