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
template <std::size_t Order>
class PassiveAggressiveRegressor {
 public:
  inline explicit PassiveAggressiveRegressor(double epsilon)
      : epsilon_(epsilon) {
    coef_ = std::vector<double>(Order, 1.0);
    power_ = std::vector<double>(Order + 1, 0.0);
  }

  // PassiveAggressiveRegressor::fit()
  //
  // Fits the model to the given vector workloads and runtimes.
  // This implementation is based on Epsilon-Insensitive Passive Aggressive
  // Regressor. Unlike Polynomial regressor, current fit is used in
  // heterogeneous environment.
  inline bool fit(const std::vector<double> &workloads,
                  const std::vector<double> &runtimes) {
    CHECK_EQ(workloads.size(), runtimes.size());
    bool early_stop = true;
    for (std::size_t iter = 0; iter < max_iter_; ++iter) {
      for (std::size_t idx = 0; idx < workloads.size(); ++idx) {
        const auto prediction = predict(workloads[idx]);
        const auto loss =
            std::max(0.0, std::abs(runtimes[idx] - prediction) - epsilon_);
        if (loss != 0.0) {
          early_stop = false;
          power_[0] = std::pow(workloads[idx], std::pow(2.0, 0));
          power_[1] = std::pow(workloads[idx], std::pow(2.0, 1));

          const auto xqnorm =
              std::accumulate(power_.begin(), power_.end(), 0.0);
          const auto update = (std::min(C_, loss / xqnorm)) *
                              ((runtimes[idx] > prediction) ? 1 : -1);

          intercept += update;
          coef_[0] += update * workloads[idx];
        }
      }
    }
    return early_stop;
  }
  // PassiveAggressiveRegressor::predict()
  //
  // Predicts value based on given scalar worload and current model coef_.
  inline const double predict(double workload) const {
    return coef_[0] * workload;
  }

  double intercept = 0.0;

 protected:
  double epsilon_;
  std::vector<double> coef_;
  std::vector<double> power_;
  static constexpr auto C_ = 1.0;
  static constexpr auto max_iter_ = 1000;
};

// PassiveAggressiveRegressor<2>
//
// An explicit template specialization of
// `flatflow::scheduler::internal::algorithm::PassiveAggressiveRegressor` for
// online learning.
//
// Initialization is based quadratic polynomial regression.
// This Regressor is currently supporting basic PassiveAggressiveRegressor.
// Unlike Linear regressor, this regressor is effective for models with
// quadratic complexity.
template <>
class PassiveAggressiveRegressor<2> {
 public:
  inline explicit PassiveAggressiveRegressor(double epsilon)
      : epsilon_(epsilon) {
    coef_ = {1.0, 1.0};
    power_ = {1.0, 0.0, 0.0};
  }
  // PassiveAggressiveRegressor<2>::fit()
  //
  // Fits the model to the given vector workloads and runtimes.
  // This implementation is based on Epsilon-Insensitive Passive Aggressive
  // Regressor. Template specialization enables regressor to fit under
  // homogeneous environment.
  inline bool fit(const std::vector<std::vector<double>> &workloads,
                  const std::vector<double> &runtimes) {
    CHECK_EQ(workloads.size(), runtimes.size());
    bool early_stop = true;
    std::vector<std::vector<double>> workload(workloads.size());
    for (std::size_t widx = 0; widx < workloads.size(); widx++) {
      std::vector<double> X(2, 0.0);
      for (std::size_t idx = 0; idx < workloads[idx].size(); idx++) {
        X[0] += workloads[widx][idx];
        X[1] += workloads[widx][idx] * workloads[widx][idx];
      }
      workload[widx] = X;
    }
    for (std::size_t iter = 0; iter < max_iter_; iter++) {
      for (std::size_t widx = 0; widx < workloads.size(); widx++) {
        const auto prediction = predict(workload[widx]);
        const auto loss =
            std::max(0.0, std::abs(runtimes[widx] - prediction) - epsilon_);
        if (loss != 0.0) {
          early_stop = false;
          power_[0] = std::pow(workload[widx][0], 1);
          power_[1] = std::pow(workload[widx][0], 2.0);
          power_[2] = std::pow(workload[widx][1], 2.0);

          const auto xqnorm =
              std::accumulate(power_.begin(), power_.end(), 0.0);
          const auto update = (std::min(C_, loss / xqnorm)) *
                              ((runtimes[widx] > prediction) ? 1 : -1);

          intercept_ += update;
          coef_[0] += update * workload[widx][0];
          coef_[1] += update * workload[widx][1];
        }
      }
    }
    return early_stop;
  }
  // PassiveAggressiveRegressor::predict()
  //
  // Predicts value based on given scalar workloads and current model coef_.
  inline const double predict(const std::vector<double> &workloads) const {
    return coef_[0] * workloads[0] + coef_[1] * workloads[1];
  }

 protected:
  double epsilon_;
  double intercept_;
  std::vector<double> coef_;
  std::vector<double> power_;
  static constexpr auto C_ = 1.0;
  static constexpr auto max_iter_ = 1000;
};

}  // namespace algorithm
}  // namespace internal
}  // namespace scheduler
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_REGRESSION_H_
