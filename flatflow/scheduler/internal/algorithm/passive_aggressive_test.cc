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

#include "flatflow/scheduler/internal/algorithm/passive_aggressive.h"

#include <functional>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace {

class RegressorTest : public testing::Test {
 protected:
  void SetUp() override {
    auto distribution = std::lognormal_distribution(5.252, 0.293);
    auto generator = std::default_random_engine();

    workloads.reserve(kNumMicroBatches);
    while (workloads.size() < workloads.capacity()) {
      auto workload = std::vector<double>();
      workload.reserve(kMicroBatchSize);
      while (workload.size() < workload.capacity()) {
        workload.emplace_back(std::round(distribution(generator)));
      }
      workloads.emplace_back(std::move(workload));
    }
  }

  static constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 2);
  static constexpr auto kNumMicroBatches = static_cast<std::size_t>(1 << 5);
  static constexpr auto kEpsilon = 1e-4;
  static constexpr auto kThreshold = 1e-2;
  static constexpr auto kHiddenSize = 4e2;
  std::vector<std::vector<double>> workloads;
};

TEST_F(RegressorTest, Linear) {
  constexpr auto kCoefficient = 1e2;
  constexpr auto kIntercept = 2e2;

  auto sums = std::vector<double>();
  sums.reserve(kNumMicroBatches);

  auto costs = std::vector<double>();
  costs.reserve(kNumMicroBatches);

  for (const auto &workload : workloads) {
    const auto sum = std::accumulate(workload.cbegin(), workload.cend(), 0.0);
    sums.emplace_back(sum);
    costs.emplace_back(kCoefficient * sum + kIntercept);
  }

  auto regressor =
      flatflow::scheduler::internal::algorithm::PassiveAggressiveRegressor<1>(
          kEpsilon);
  EXPECT_FALSE(regressor.fit(sums, costs));

  for (std::size_t index = 0; index < kNumMicroBatches; ++index) {
    const auto sum = sums[index];
    const auto cost = costs[index];
    const auto prediction = regressor.predict(sum) + regressor.intercept();
    EXPECT_LE(std::abs(cost - prediction) / cost, kThreshold);
  }
}

TEST_F(RegressorTest, Quadratic) {
  constexpr auto kIntercept = 3e2;

  auto costs = std::vector<double>();
  costs.reserve(kNumMicroBatches);

  for (const auto &workload : workloads) {
    const auto sum = std::accumulate(workload.cbegin(), workload.cend(), 0.0);
    const auto sqsum = std::inner_product(workload.cbegin(), workload.cend(),
                                          workload.cbegin(), 0.0);
    costs.emplace_back(sqsum + kHiddenSize * sum + kIntercept);
  }

  auto regressor =
      flatflow::scheduler::internal::algorithm::PassiveAggressiveRegressor<2>(
          kHiddenSize, kEpsilon);
  EXPECT_FALSE(regressor.fit(workloads, costs));

  for (std::size_t index = 0; index < kNumMicroBatches; ++index) {
    const auto &workload = workloads[index];
    const auto cost = costs[index];
    const auto prediction = std::transform_reduce(
        workload.cbegin(), workload.cend(), regressor.intercept(), std::plus(),
        [&](const auto size) { return regressor.predict(size); });
    EXPECT_LE(std::abs(cost - prediction) / cost, kThreshold);
  }
}

}  // namespace
