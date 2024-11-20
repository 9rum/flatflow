// Adapted from https://github.com/scikit-learn/scikit-learn/blob/1.5.0/sklearn/linear_model/_passive_aggressive.py
// Copyright (c) 2007-2024 The scikit-learn developers. All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef FLATFLOW_SKLEARN_LINEAR_MODEL_PASSIVE_AGGRESSIVE_H_
#define FLATFLOW_SKLEARN_LINEAR_MODEL_PASSIVE_AGGRESSIVE_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <vector>

#include "flatflow/data/internal/types.h"

namespace flatflow {
namespace sklearn {
namespace linear_model {

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

  // Constructors and assignment operators
  //
  // `epsilon` denotes the threshold for prediction loss, which defaults to 0.1.
  // If the difference between the current prediction and the correct label is
  // below this threshold, the model is not updated.
  explicit PassiveAggressiveRegressor(double epsilon = 0.1, double C = 1.0,
                                      std::size_t max_iter = 1000)
      : epsilon_(epsilon), C_(C), max_iter_(max_iter) {
    coef_ = 1.0;
    intercept_ = 0.0;
    converged_ = false;
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
  template <typename T, typename U>
    requires(flatflow::internal::Numerical<T> &&
             flatflow::internal::Numerical<U>)
  void fit(const std::vector<T> &sizes, const std::vector<U> &costs) {
    assert(sizes.size() == costs.size());

    if (costs.empty()) {
      return;
    }

    converged_ = true;

    for (std::size_t epoch = 0; epoch < max_iter_; ++epoch) {
      for (std::size_t index = 0; index < sizes.size(); ++index) {
        const auto size = static_cast<double>(sizes[index]);
        const auto cost = static_cast<double>(costs[index]);

        const auto prediction = coef_ * size + intercept_;
        const auto loss = std::abs(cost - prediction) - epsilon_;

        if (0.0 < loss) {
          const auto sqnorm = size * (size + 1.0);
          const auto update = std::min(loss / sqnorm, C_);

          if (prediction < cost) {
            coef_ += update * size;
            intercept_ += update;
          } else {
            coef_ -= update * size;
            intercept_ -= update;
          }

          converged_ = false;
        }
      }
    }
  }

  // PassiveAggressiveRegressor::predict()
  //
  // Predicts cost for the given size.
  template <typename T>
    requires flatflow::internal::Numerical<T>
  inline double predict(T size) const noexcept {
    return coef_ * static_cast<double>(size);
  }

  // PassiveAggressiveRegressor::intercept()
  //
  // Returns the current model intercept.
  inline double intercept() const noexcept { return intercept_; }

  // PassiveAggressiveRegressor::converged()
  //
  // Returns whether the model has converged.
  inline bool converged() const noexcept { return converged_; }

 protected:
  std::size_t max_iter_;
  double C_;
  double coef_;
  double epsilon_;
  double intercept_;
  bool converged_;
};

// PassiveAggressiveRegressor<>
//
// A template specialization of `PassiveAggressiveRegressor` for models with
// quadratic complexity.
template <>
class PassiveAggressiveRegressor</*Order=*/2> {
 public:
  explicit PassiveAggressiveRegressor() {}

  // Constructors and assignment operators
  //
  // Unlike its linear counterpart, this regressor requires initial value
  // of the linear term.
  // For typical Transformers, this is eight times the hidden dimension size.
  template <typename T>
    requires flatflow::internal::Numerical<T>
  explicit PassiveAggressiveRegressor(T coefficient, double epsilon = 0.1,
                                      double C = 1.0,
                                      std::size_t max_iter = 1000)
      : epsilon_(epsilon), C_(C), max_iter_(max_iter) {
    intercept_ = 0.0;
    converged_ = false;
    coef_.front() = 1.0;
    coef_.back() = static_cast<double>(coefficient);
  }

  explicit PassiveAggressiveRegressor(const PassiveAggressiveRegressor &other) = default;

  PassiveAggressiveRegressor &operator=(const PassiveAggressiveRegressor &other) = default;

  explicit PassiveAggressiveRegressor(PassiveAggressiveRegressor &&other) = default;

  PassiveAggressiveRegressor &operator=(PassiveAggressiveRegressor &&other) = default;

  // PassiveAggressiveRegressor::fit()
  //
  // Fits the model with passive-aggressive algorithm.
  // Unlike the linear model, this regressor takes one or more sizes for each
  // cost since multiple sizes are mapped to a single cost in the profile.
  // For this reason, a dot product-based regression is used instead of the
  // canonical regression.
  template <typename T, typename U>
    requires(flatflow::internal::Numerical<T> &&
             flatflow::internal::Numerical<U>)
  void fit(const std::vector<std::vector<T>> &sizes,
           const std::vector<U> &costs) {
    assert(sizes.size() == costs.size());

    if (costs.empty()) {
      return;
    }

    converged_ = true;

    for (std::size_t epoch = 0; epoch < max_iter_; ++epoch) {
      for (std::size_t index = 0; index < sizes.size(); ++index) {
        const auto &_sizes = sizes[index];
        const auto cost = costs[index];

        const auto sum = std::accumulate(_sizes.cbegin(), _sizes.cend(), 0.0);
        const auto sqsum = std::inner_product(_sizes.cbegin(), _sizes.cend(),
                                              _sizes.cbegin(), 0.0);

        const auto prediction =
            coef_.front() * sqsum + coef_.back() * sum + intercept_;
        const auto loss = std::abs(cost - prediction) - epsilon_;

        if (0.0 < loss) {
          const auto sqnorm = sum * sum + sqsum * sqsum;
          const auto update = std::min(loss / sqnorm, C_);

          if (prediction < cost) {
            coef_.front() += update * sqsum;
            coef_.back() += update * sum;
            intercept_ += update;
          } else {
            coef_.front() -= update * sqsum;
            coef_.back() -= update * sum;
            intercept_ -= update;
          }

          converged_ = false;
        }
      }
    }
  }

  // PassiveAggressiveRegressor::predict()
  //
  // Predicts cost for the given size.
  template <typename T>
    requires flatflow::internal::Numerical<T>
  inline double predict(T size) const noexcept {
    return static_cast<double>(size) *
           (coef_.front() * static_cast<double>(size) + coef_.back());
  }

  // PassiveAggressiveRegressor::intercept()
  //
  // Returns the current model intercept.
  inline double intercept() const noexcept { return intercept_; }

  // PassiveAggressiveRegressor::converged()
  //
  // Returns whether the model has converged.
  inline bool converged() const noexcept { return converged_; }

 protected:
  std::size_t max_iter_;
  double C_;
  double epsilon_;
  double intercept_;
  bool converged_;
  std::array<double, 2> coef_;
};

}  // namespace linear_model
}  // namespace sklearn
}  // namespace flatflow

#endif  // FLATFLOW_SKLEARN_LINEAR_MODEL_PASSIVE_AGGRESSIVE_H_
