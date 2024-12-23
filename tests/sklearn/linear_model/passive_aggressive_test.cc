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

#include "flatflow/sklearn/linear_model/passive_aggressive.h"

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

    sizes.reserve(kNumMicroBatches);
    while (sizes.size() < sizes.capacity()) {
      auto _sizes = std::vector<double>();
      _sizes.reserve(kMicroBatchSize);
      while (_sizes.size() < _sizes.capacity()) {
        _sizes.emplace_back(std::round(distribution(generator)));
      }
      sizes.emplace_back(std::move(_sizes));
    }
  }

  static constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 2);
  static constexpr auto kNumMicroBatches = static_cast<std::size_t>(1 << 5);
  static constexpr auto kEpsilon = 1e-4;
  static constexpr auto kThreshold = 1e-2;
  static constexpr auto kHiddenSize = 4e2;
  std::vector<std::vector<double>> sizes;
};

TEST_F(RegressorTest, Linear) {
  constexpr auto kCoefficient = 1e2;
  constexpr auto kIntercept = 2e2;

  auto sums = std::vector<double>();
  sums.reserve(kNumMicroBatches);

  auto costs = std::vector<double>();
  costs.reserve(kNumMicroBatches);

  for (const auto &_sizes : sizes) {
    const auto sum = std::accumulate(_sizes.cbegin(), _sizes.cend(), 0.0);
    sums.emplace_back(sum);
    costs.emplace_back(kCoefficient * sum + kIntercept);
  }

  auto regressor =
      flatflow::sklearn::linear_model::PassiveAggressiveRegressor<1>(1.0, 1000,
                                                                     kEpsilon);
  regressor.fit(sums, costs);
  EXPECT_FALSE(regressor.converged());

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

  for (const auto &_sizes : sizes) {
    const auto sum = std::accumulate(_sizes.cbegin(), _sizes.cend(), 0.0);
    const auto sqsum = std::inner_product(_sizes.cbegin(), _sizes.cend(),
                                          _sizes.cbegin(), 0.0);
    costs.emplace_back(sqsum + kHiddenSize * sum + kIntercept);
  }

  auto regressor =
      flatflow::sklearn::linear_model::PassiveAggressiveRegressor<2>(
          kHiddenSize, 1.0, 1000, kEpsilon);
  regressor.fit(sizes, costs);
  EXPECT_FALSE(regressor.converged());

  for (std::size_t index = 0; index < kNumMicroBatches; ++index) {
    const auto &_sizes = sizes[index];
    const auto cost = costs[index];
    const auto prediction = std::transform_reduce(
        _sizes.cbegin(), _sizes.cend(), regressor.intercept(), std::plus(),
        [&](const auto size) { return regressor.predict(size); });
    EXPECT_LE(std::abs(cost - prediction) / cost, kThreshold);
  }
}

}  // namespace
