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

#include "flatflow/scheduler/internal/algorithm/regression.h"

#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace {
template <std::size_t Order>
class Regressor : public flatflow::scheduler::internal::algorithm::
                      PassiveAggressiveRegressor<Order> {
 public:
  using flatflow::scheduler::internal::algorithm::PassiveAggressiveRegressor<
      Order>::PassiveAggressiveRegressor;

  void check_coefficients(const std::vector<double> &coeff, double threshold) {
    const auto &model = this->coef_;
    const auto predict = model[model.size() - 1];
    const auto label = coeff[coeff.size() - 1];
    const auto result = std::abs(predict - label) / label;
    EXPECT_LE(std::floor((result * 100) + .5) / 100, threshold);
  }
};

class RegressionTest : public testing::Test {
 protected:
  void SetUp() override { runtimes_.reserve(kDatasetSize); }

  double generate_linear(double x, const std::vector<double> &coef) {
    double sum = coef[0];
    for (std::size_t i = 0; i < coef.size(); i++) {
      sum += coef[i] * std::pow(x, i);
    }
    return sum;
  }

  double generate_quadratic(const std::vector<double> &x,
                            const std::vector<double> &coef) {
    double sum = coef[0];
    for (std::size_t i = 1; i < coef.size(); i++) {
      double temp = 0.0;
      for (const auto &xi : x) {
        temp += std::pow(xi, i);
      }
      sum += coef[i] * temp;
    }
    return sum;
  }

  std::vector<double> generate_coefficients(std::size_t degree) {
    std::vector<double> coeffs;
    std::random_device random_device;
    std::mt19937 random_generator(random_device());
    std::uniform_real_distribution<double> uniform_dist(kMinCoef, kMaxCoef);
    for (std::size_t coefIdx = 0; coefIdx <= degree; ++coefIdx) {
      coeffs.push_back(uniform_dist(random_generator));
    }
    return coeffs;
  }

  std::vector<double> random_partition(int x, int node) {
    if (node == 0) {
      return {};
    }
    std::vector<double> result;
    result.reserve(node);
    const auto average = static_cast<double>(x) / node;
    const auto remainder = x % node;
    for (std::size_t index = 0; index < node; ++index) {
      result.push_back(average);
    }
    result.back() += remainder;
    return result;
  }

  static constexpr auto kLinear = static_cast<std::size_t>(1);
  static constexpr auto kQuadratic = static_cast<std::size_t>(2);
  static constexpr auto kEpsilon = static_cast<double>(0.0001);
  static constexpr auto kThreshold = static_cast<double>(0.01);
  static constexpr auto kNode = static_cast<int>(4);
  static constexpr auto kMinRange = static_cast<int>(100);
  static constexpr auto kMaxRange = static_cast<int>(200);
  static constexpr auto kMinCoef = static_cast<int>(300);
  static constexpr auto kMaxCoef = static_cast<int>(400);
  static constexpr std::size_t kDatasetSize = 100;
  std::vector<double> runtimes_;
};

TEST_F(RegressionTest, LinearRegression) {
  std::vector<double> workloads;
  Regressor<kLinear> regressor(kEpsilon);
  const auto coeff = generate_coefficients(kLinear);
  std::random_device random_device;
  std::mt19937 random_generator(random_device());
  std::uniform_real_distribution<double> uniform_dist(kMinRange, kMaxRange);
  for (std::size_t i = 0; i < kDatasetSize; i++) {
    const auto x = uniform_dist(random_generator);
    workloads.push_back(x);
    double predict = generate_linear(x, coeff);
    runtimes_.push_back(predict);
  }
  regressor.fit(workloads, runtimes_);
  regressor.check_coefficients(coeff, kThreshold);
}

TEST_F(RegressionTest, PolynomialRegression) {
  std::vector<std::vector<double>> workloads;
  Regressor<kQuadratic> regressor(kEpsilon);
  const auto coeff = generate_coefficients(kQuadratic);
  std::random_device random_device;
  std::mt19937 random_generator(random_device());
  std::uniform_real_distribution<double> uniform_dist(kMinRange, kMaxRange);
  for (std::size_t i = 0; i < kDatasetSize; i++) {
    std::vector<double> partition;
    const auto x = random_partition(uniform_dist(random_generator), kNode);
    for (std::size_t index = 0; index < kNode; index++) {
      partition.push_back(x[index]);
    }
    workloads.push_back(partition);
    double predict = generate_quadratic(partition, coeff);
    runtimes_.push_back(predict);
  }
  regressor.fit(workloads, runtimes_);
  regressor.check_coefficients(coeff, kThreshold);
}

}  // namespace
