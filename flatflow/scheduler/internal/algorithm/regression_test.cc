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
class Regression : public flatflow::scheduler::internal::algorithm::
                       PassiveAggressiveRegressor<degree> {
 public:
  Regression(const double epsilon)
      : flatflow::scheduler::internal::algorithm::PassiveAggressiveRegressor<
            degree>(epsilon) {}
  void fit(const std::vector<double> &X, const std::vector<double> &y) {
    flatflow::scheduler::internal::algorithm::PassiveAggressiveRegressor<
        degree>::fit(X, y);
  }

  inline std::vector<double> &getCoef() { return this->coef_; }
};

class RegressionTest : public testing::Test {
 protected:
  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(100.0, 120.0);
    for (std::size_t idx = 0; idx < kDatasetSize; ++idx) {
      X[idx] = dis(gen);
    }
  }

  double generateDummy(double x, std::vector<double> coef) {
    double sum = 0;
    for (std::size_t coefIdx = 0; coefIdx < coef.size(); ++coefIdx) {
      sum += coef[coefIdx] * std::pow(x, coefIdx);
    }
    return sum;
  }

  std::vector<double> generateCoefficients(std::size_t degree) {
    std::vector<double> coeffs;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(300.0, 400.0);
    for (std::size_t coefIdx = 0; coefIdx <= degree; ++coefIdx) {
      coeffs.push_back(dis(gen));
    }
    return coeffs;
  }

  void checkThreshold(const std::vector<double> &modelCofficients,
                      const std::vector<double> &labelCofficients) {
    const auto idx = modelCofficients.size() - 1;
    EXPECT_TRUE(std::abs(modelCofficients[idx] - labelCofficients[idx]) /
                    labelCofficients[idx] <
                kThreshold);
  }

  static constexpr auto kDatasetSize = static_cast<std::size_t>((1 << 10));
  static constexpr auto kEpsilon = 1e-4;
  static constexpr auto kThreshold = 0.01;
  std::vector<double> X = std::vector<double>(kDatasetSize, 0);
  std::vector<double> y = std::vector<double>(kDatasetSize, 0);
};

TEST_F(RegressionTest, LinearRegression) {
  constexpr int degree(1);

  Regression<degree> regressor(kEpsilon);

  std::vector<double> coefficients = generateCoefficients(degree);

  for (std::size_t idx = 0; idx < kDatasetSize; ++idx) {
    y[idx] = generateDummy(X[idx], coefficients);
  }
  regressor.fit(X, y);
  const auto &modelCofficients = regressor.getCoef();
  checkThreshold(modelCofficients, coefficients);
}

TEST_F(RegressionTest, PolynomialRegression) {
  constexpr int degree(2);

  Regression<degree> regressor(kEpsilon);

  std::vector<double> coefficients = generateCoefficients(degree);

  for (std::size_t idx = 0; idx < kDatasetSize; ++idx) {
    y[idx] = generateDummy(X[idx], coefficients);
  }
  regressor.fit(X, y);
  const auto &modelCofficients = regressor.getCoef();
  checkThreshold(modelCofficients, coefficients);
}

}  // namespace
