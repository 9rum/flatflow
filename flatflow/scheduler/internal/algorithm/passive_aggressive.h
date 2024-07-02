// Adapted from https://github.com/scikit-learn/scikit-learn/blob/1.5.0/sklearn/linear_model/_passive_aggressive.py
// Copyright (c) 2007-2024 The scikit-learn developers. All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_PASSIVE_AGGRESSIVE_H_
#define FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_PASSIVE_AGGRESSIVE_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <vector>

namespace flatflow {
namespace scheduler {
namespace internal {
namespace algorithm {

// PassiveAggressiveRegressor<>
//
// The passive-aggressive algorithms are a family of algorithms
// for large-scale learning. They are similar to the perceptron
// in that they do not require a learning rate. However, contrary
// to the perceptron, they include a regularization parameter `C`.
// See https://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf
// for reference.
//
// This is a port of `sklearn.linear_model.PassiveAggressiveRegressor` for
// internal usage. Note that it is not intended to be used for an arbitrary
// number of features; this regressor can be used only for linear models.
// A separate template specialization exists for quadratic models, and each
// template specialization is optimized for the given order.
template <int Order>
class PassiveAggressiveRegressor {
 public:
  explicit PassiveAggressiveRegressor() {}

  // `epsilon` denotes the threshold for prediction loss, which defaults to 0.1.
  // If the difference between the current prediction and the correct label is
  // below this threshold, the model is not updated.
  explicit PassiveAggressiveRegressor(double epsilon = 0.1, double C = 1.0,
                                      std::size_t max_iter = 1000)
      : epsilon_(epsilon), C_(C), max_iter_(max_iter) {
    coef_ = 1.0;
    intercept_ = 0.0;
  }

  explicit PassiveAggressiveRegressor(const PassiveAggressiveRegressor &other) = default;

  PassiveAggressiveRegressor &operator=(const PassiveAggressiveRegressor &other) = default;

  explicit PassiveAggressiveRegressor(PassiveAggressiveRegressor &&other) = default;

  PassiveAggressiveRegressor &operator=(PassiveAggressiveRegressor &&other) = default;

  // PassiveAggressiveRegressor::fit()
  //
  // Fits the model with passive-aggressive algorithm.
  // This uses epsilon-insensitive loss, which is equivalent to PA-I
  // in the reference paper.
  bool fit(const std::vector<double> &workloads,
           const std::vector<double> &costs) {
    assert(workloads.size() == costs.size());

    auto converged = true;

    for (std::size_t epoch = 0; epoch < max_iter_; ++epoch) {
      for (std::size_t index = 0; index < workloads.size(); ++index) {
        const auto workload = workloads[index];
        const auto cost = costs[index];

        const auto prediction = coef_ * workload + intercept_;
        const auto loss = std::abs(cost - prediction) - epsilon_;

        if (0.0 < loss) {
          const auto sqnorm = workload * (workload + 1.0);
          const auto update = std::min(loss / sqnorm, C_);

          if (prediction < cost) {
            coef_ += update * workload;
            intercept_ += update;
          } else {
            coef_ -= update * workload;
            intercept_ -= update;
          }

          converged = false;
        }
      }
    }

    return converged;
  }

  // PassiveAggressiveRegressor::predict()
  //
  // Predicts cost for the given workload.
  inline double predict(double workload) const noexcept {
    return coef_ * workload;
  }

  // PassiveAggressiveRegressor::intercept()
  //
  // Returns the current model intercept.
  inline double intercept() const noexcept { return intercept_; }

 protected:
  std::size_t max_iter_;
  double C_;
  double coef_;
  double epsilon_;
  double intercept_;
};

// PassiveAggressiveRegressor<>
//
// A template specialization of `PassiveAggressiveRegressor` for models with
// quadratic complexity.
template <>
class PassiveAggressiveRegressor</*Order=*/2> {
 public:
  explicit PassiveAggressiveRegressor() {}

  // Unlike its linear counterpart, this regressor requires `hidden_size`
  // to initialize the coefficients since the complexity of Transformers is
  // `O(n^2 d + n d^2)`, where `n` and `d` denote the sequence length and
  // hidden size, respectively.
  explicit PassiveAggressiveRegressor(double hidden_size, double epsilon = 0.1,
                                      double C = 1.0,
                                      std::size_t max_iter = 1000)
      : epsilon_(epsilon), C_(C), max_iter_(max_iter) {
    coef_ = std::to_array({1.0, hidden_size});
    intercept_ = 0.0;
  }

  explicit PassiveAggressiveRegressor(const PassiveAggressiveRegressor &other) = default;

  PassiveAggressiveRegressor &operator=(const PassiveAggressiveRegressor &other) = default;

  explicit PassiveAggressiveRegressor(PassiveAggressiveRegressor &&other) = default;

  PassiveAggressiveRegressor &operator=(PassiveAggressiveRegressor &&other) = default;

  // PassiveAggressiveRegressor::fit()
  //
  // Fits the model with passive-aggressive algorithm. Unlike the linear model,
  // this regressor takes one or more workloads for each cost since multiple
  // workloads are mapped to a single cost in the profile. For this reason, a
  // dot product-based regression is used instead of the canonical regression.
  bool fit(const std::vector<std::vector<double>> &workloads,
           const std::vector<double> &costs) {
    assert(workloads.size() == costs.size());

    auto converged = true;

    for (std::size_t epoch = 0; epoch < max_iter_; ++epoch) {
      for (std::size_t index = 0; index < workloads.size(); ++index) {
        const auto &workload = workloads[index];
        const auto cost = costs[index];

        const auto sum =
            std::accumulate(workload.cbegin(), workload.cend(), 0.0);
        const auto sqsum = std::inner_product(
            workload.cbegin(), workload.cend(), workload.cbegin(), 0.0);

        const auto prediction = coef_[0] * sqsum + coef_[1] * sum + intercept_;
        const auto loss = std::abs(cost - prediction) - epsilon_;

        if (0.0 < loss) {
          const auto sqnorm = sum * sum + sqsum * sqsum;
          const auto update = std::min(loss / sqnorm, C_);

          if (prediction < cost) {
            coef_[0] += update * sqsum;
            coef_[1] += update * sum;
            intercept_ += update;
          } else {
            coef_[0] -= update * sqsum;
            coef_[1] -= update * sum;
            intercept_ -= update;
          }

          converged = false;
        }
      }
    }

    return converged;
  }

  // PassiveAggressiveRegressor::predict()
  //
  // Predicts cost for the given workload.
  inline double predict(double workload) const noexcept {
    return workload * (coef_[0] * workload + coef_[1]);
  }

  // PassiveAggressiveRegressor::intercept()
  //
  // Returns the current model intercept.
  inline double intercept() const noexcept { return intercept_; }

 protected:
  std::size_t max_iter_;
  double C_;
  double epsilon_;
  double intercept_;
  std::array<double, 2> coef_;
};

}  // namespace algorithm
}  // namespace internal
}  // namespace scheduler
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_PASSIVE_AGGRESSIVE_H_
