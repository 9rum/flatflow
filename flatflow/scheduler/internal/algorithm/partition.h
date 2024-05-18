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

#ifndef FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_PARTITION_H_
#define FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_PARTITION_H_

#include <algorithm>
#include <iterator>
#include <queue>
#include <span>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/internal/platform.h"

namespace flatflow {
namespace scheduler {
namespace internal {
namespace algorithm {

namespace {

// Subset<>
//
//
struct Subset {
  inline ABSL_ATTRIBUTE_ALWAYS_INLINE explicit Subset(
      const std::pair<const uint16_t, uint64_t> &item) {
    sum = item.first;
    indices = std::vector<uint64_t>(1, item.second);
  }

  inline ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator<(
      const Subset &other) const noexcept {
    return sum < other.sum;
  }

  inline ABSL_ATTRIBUTE_ALWAYS_INLINE void Join(const Subset &other) {
    sum += other.sum;
    indices.reserve(indices.size() + other.indices.size());
    indices.insert(indices.end(),
                   std::make_move_iterator(other.indices.begin()),
                   std::make_move_iterator(other.indices.end()));
  }

  uint_fast64_t sum;
  std::vector<uint64_t> indices;
};

// Solution<>
//
//
struct Solution {
  ABSL_ATTRIBUTE_NOINLINE explicit Solution(
      std::span<const std::pair<const uint16_t, uint64_t>> items) {
    subsets.reserve(items.size());
    for (const auto &item : items) {
      subsets.emplace_back(Subset(item));
    }
    difference = subsets.at(subsets.size() - 1).sum - subsets.at(0).sum;
  }

  inline ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator<(
      const Solution &other) const noexcept {
    return difference < other.difference;
  }

  ABSL_ATTRIBUTE_NOINLINE void Difference(const Solution &other) {
    for (std::size_t index = 0; index < subsets.size(); ++index) {
      subsets.at(index).Join(other.subsets.at(subsets.size() - index - 1));
    }
    std::sort(subsets.begin(), subsets.end());
    difference = subsets.at(subsets.size() - 1).sum - subsets.at(0).sum;
  }

  uint_fast64_t difference;
  std::vector<Subset> subsets;
};

}  // namespace

// KarmarkarKarp()
//
// Partitions the given items using the Balanced Largest Differencing Method
// (BLDM) of Michiels, Aarts, Korst, van Leeuwen and Spieksma from the paper
// `Computer-assisted proof of performance ratios for the Differencing Method
// <https://www.sciencedirect.com/science/article/pii/S1572528611000508>`,
// a variant of LDM for balanced number partitioning with larger cardinalities.
ABSL_ATTRIBUTE_NOINLINE
std::vector<std::pair<uint_fast64_t, std::vector<uint64_t>>> KarmarkarKarp(
    const std::vector<std::pair<const uint16_t, uint64_t>> &items,
    uint64_t micro_batch_size, uint64_t num_micro_batches) {
  auto solutions = std::priority_queue<Solution>();

  for (std::size_t index = 0; index < items.size();
       index += static_cast<std::size_t>(num_micro_batches)) {
    solutions.emplace(std::span(items).subspan(
        index, static_cast<std::size_t>(num_micro_batches)));
  }

  for (; 1 < solutions.size();) {
    auto solution = solutions.top();
    solutions.pop();

    auto other = solutions.top();
    solutions.pop();

    solution.Difference(other);
    solutions.emplace(solution);
  }

  const auto &solution = solutions.top();

  auto micro_batches =
      std::vector<std::pair<uint_fast64_t, std::vector<uint64_t>>>();
  micro_batches.reserve(solution.subsets.size());

  for (const auto &subset : solution.subsets) {
    micro_batches.emplace_back(subset.sum, std::move(subset.indices));
  }

  return micro_batches;
}

// Multifit()
//
// Partitions the given items using the multifit algorithm of Coffman, Garey and
// Johnson from the paper `An Application of Bin-Packing to Multiprocessor
// Scheduling <https://epubs.siam.org/doi/abs/10.1137/0207001>`.

}  // namespace algorithm
}  // namespace internal
}  // namespace scheduler
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_PARTITION_H_
