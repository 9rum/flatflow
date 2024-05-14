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
template <std::size_t degree>
class Regressor : public flatflow::scheduler::internal::algorithm::
                      PassiveAggressiveRegressor<degree> {
 public:
  using flatflow::scheduler::internal::algorithm::PassiveAggressiveRegressor<
      degree>::PassiveAggressiveRegressor;

  void checkCoefficients(const std::vector<double> &coefficients,
                         double threshold) {
    const auto &modelCoefficients = this->coef_;
    const auto modelCoefficient =
        modelCoefficients[modelCoefficients.size() - 1];
    const auto labelCoefficient = coefficients[coefficients.size() - 1];
    const auto result =
        std::abs(modelCoefficient - labelCoefficient) / labelCoefficient;
    EXPECT_LE(std::floor((result * 100) + .5) / 100, threshold);
  }
};

class RegressionTest : public testing::Test {
 protected:
  void SetUp() override { runtimes_.reserve(kDatasetSize); }

  double generateLinear(const double &x, const std::vector<double> &coef) {
    double sum = coef[0];
    for (std::size_t i = 0; i < coef.size(); i++) {
      sum += coef[i] * std::pow(x, i);
    }
    return sum;
  }

  double generateQuadratic(const std::vector<double> &x,
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

  std::vector<double> generateCoefficients(std::size_t degree) {
    std::vector<double> coeffs;
    std::random_device randomDevice;
    std::mt19937 randomGenerator(randomDevice());
    std::uniform_real_distribution<double> uniformDist(kMinCoef, kMaxCoef);
    for (std::size_t coefIdx = 0; coefIdx <= degree; ++coefIdx) {
      coeffs.push_back(uniformDist(randomGenerator));
    }
    return coeffs;
  }

  std::vector<double> randomPartition(const int &x, const int &node) {
    if (node == 0) {
      return {};
    }
    std::vector<double> result;
    result.reserve(node);
    const auto avg = static_cast<double>(x) / node;
    const auto remainder = x % node;
    for (std::size_t index = 0; index < node; ++index) {
      result.push_back(avg);
    }
    result.back() += remainder;
    return result;
  }

  static constexpr std::size_t kLinear = 1;
  static constexpr std::size_t kQuadratic = 2;
  static constexpr auto kEpsilon = 0.0001;
  static constexpr auto kThreshold = 0.01;
  static constexpr auto kNode = 4;
  static constexpr auto kMinRange = 100.0;
  static constexpr auto kMaxRange = 200.0;
  static constexpr auto kMinCoef = 300.0;
  static constexpr auto kMaxCoef = 400.0;
  static constexpr std::size_t kDatasetSize = 100;
  std::vector<double> runtimes_;
};

TEST_F(RegressionTest, LinearRegression) {
  std::vector<double> workloads;
  Regressor<kLinear> regressor(kEpsilon);
  const auto coefficients = generateCoefficients(kLinear);
  std::random_device randomDevice;
  std::mt19937 randomGenerator(randomDevice());
  std::uniform_real_distribution<double> uniformDist(kMinRange, kMaxRange);
  for (std::size_t i = 0; i < kDatasetSize; i++) {
    const auto x = uniformDist(randomGenerator);
    workloads.push_back(x);
    double predict = generateLinear(x, coefficients);
    runtimes_.push_back(predict);
  }
  regressor.fit(workloads, runtimes_);
  regressor.checkCoefficients(coefficients, kThreshold);
}

TEST_F(RegressionTest, PolynomialRegression) {
  std::vector<std::vector<double>> workloads;
  Regressor<kQuadratic> regressor(kEpsilon);
  const auto coefficients = generateCoefficients(kQuadratic);
  std::random_device randomDevice;
  std::mt19937 randomGenerator(randomDevice());
  std::uniform_real_distribution<double> uniformDist(kMinRange, kMaxRange);
  for (std::size_t i = 0; i < kDatasetSize; i++) {
    std::vector<double> partition;
    const auto x = randomPartition(uniformDist(randomGenerator), kNode);
    for (std::size_t nodeIndex = 0; nodeIndex < kNode; nodeIndex++) {
      partition.push_back(x[nodeIndex]);
    }
    workloads.push_back(partition);
    double predict = generateQuadratic(partition, coefficients);
    runtimes_.push_back(predict);
  }
  regressor.fit(workloads, runtimes_);
  regressor.checkCoefficients(coefficients, kThreshold);
}

}  // namespace
