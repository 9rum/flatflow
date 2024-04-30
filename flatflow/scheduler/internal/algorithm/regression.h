// Adapted from https://github.com/scikit-learn/scikit-learn/blob/1.4.2/sklearn/linear_model/_sgd_fast.pyx.tp
// Copyright (c) 2007-2024 The scikit-learn developers. All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_REGRESSION_H_
#define FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_REGRESSION_H_

#include <cmath>
#include <execution>
#include <numeric>
#include <vector>

#include "absl/log/check.h"

namespace flatflow {
namespace scheduler {
namespace internal {
namespace algorithm {

// PassiveAggressiveRegressor<>
//
// A`flatflow::scheduler::internal::algorithm::PassiveAggressiveRegressor` is
// pure cpp version of sklearn.linear_model.PassiveAggressiveRegressor.
//
// coef_ is initialized in to 1.0. And bias term is initialized to 0.0.
// This initialization method is based on flatflow static scheduler.
// This Regressor is currently supporting basic PassiveAggressiveRegressor.
// C_ and max_iter are each fixed to 1.0 and 1000.
// Epsilon is the threshold for the update. Original PassiveAggressiveRegressor
// assigns epsilon value to 0.1. Smaller the epsilon is, loss will be updated
// more. See _sgd_fast.pyx.tp for detail implementation.
//
// References:
// https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html#sklearn.linear_model.PassiveAggressiveRegressor
template <std::size_t degree>
class PassiveAggressiveRegressor {
 public:
  inline explicit PassiveAggressiveRegressor(const double &epsilon)
      : epsilon_(epsilon) {
    coef_ = std::vector<double>(degree + 1, 1.0);
    power_ = std::vector<double>(degree + 1, 0.0);
    coef_[0] = 0.0;
  }

  // PassiveAggressiveRegressor::fit()
  //
  // Fits the model to the given vector data X and y.
  // This implementation is based on Epsilon-Insensitive Passive Aggressive
  // Regressor. This method is implementation of partialfit method in order to
  // use in online learning.
  inline void fit(const std::vector<double> &X, const std::vector<double> &y) {
    CHECK_EQ(X.size(), y.size());
    for (std::size_t iter = 0; iter < kMaxIter_; ++iter) {
      for (std::size_t dataIdx = 0; dataIdx < X.size(); ++dataIdx) {
        const auto prediction = predict(X[dataIdx]);
        const auto loss =
            std::max(0.0, std::abs(y[dataIdx] - prediction) - epsilon_);

        for (std::size_t coefIdx = 0; coefIdx < coef_.size(); ++coefIdx) {
          power_[coefIdx] = std::pow(X[dataIdx], std::pow(2.0, coefIdx));
        }

        const auto xqnorm = std::accumulate(power_.begin(), power_.end(), 0.0);
        const auto update = (std::min(kC_, loss / xqnorm)) *
                            ((y[dataIdx] > prediction) ? 1 : -1);

        for (std::size_t coefIdx = 0; coefIdx < coef_.size(); ++coefIdx) {
          coef_[coefIdx] += update * std::pow(X[dataIdx], coefIdx);
        }
      }
    }
  }

  // PassiveAggressiveRegressor::predict()
  //
  // Predicts value based on given scalar x and current model coef_.
  inline double predict(const double &x) const {
    auto prediction = 0.0;
    for (std::size_t coefIdx = 0; coefIdx < coef_.size(); ++coefIdx) {
      prediction += coef_[coefIdx] * std::pow(x, coefIdx);
    }
    return prediction;
  }

 protected:
  double epsilon_;
  std::vector<double> coef_;
  std::vector<double> power_;
  static constexpr auto kMaxIter_ = 1000;
  static constexpr auto kC_ = 1.0;
};

}  // namespace algorithm
}  // namespace internal
}  // namespace scheduler
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_REGRESSION_H_
