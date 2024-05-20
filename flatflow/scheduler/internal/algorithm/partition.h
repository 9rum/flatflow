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

#include <omp.h>

#include <algorithm>
#include <concepts>
#include <execution>
#include <iterator>
#include <queue>
#include <span>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/internal/platform.h"
#include "absl/strings/str_format.h"

#include "flatflow/data/internal/types.h"

namespace flatflow {
namespace scheduler {
namespace internal {
namespace algorithm {

namespace {

// Subset<>
//
// Structure that stores a partition created by the Karmarkar-Karp algorithm.
// Note that unlike in the original partition problem, this structure does not
// store the items but only the subset sum.
template <typename Index, typename Size, typename UnaryOp>
  requires(flatflow::data::internal::Unsigned<Index> &&
           flatflow::data::internal::Unsigned<Size> &&
           std::invocable<UnaryOp, Size>)
struct Subset {
  using key_type = Size;
  using mapped_type = Index;
  using result_type = std::invoke_result_t<UnaryOp, Size>;

  inline ABSL_ATTRIBUTE_ALWAYS_INLINE explicit Subset(
      const std::pair<const key_type, mapped_type> &ABSL_RANDOM_INTERNAL_RESTRICT item,
      UnaryOp op) {
    sum = op(item.first);
    indices = std::vector<mapped_type>(1, item.second);
  }

  inline ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator<(
      const Subset &ABSL_RANDOM_INTERNAL_RESTRICT other) const noexcept {
    return sum < other.sum;
  }

  inline ABSL_ATTRIBUTE_ALWAYS_INLINE void Join(
      const Subset &ABSL_RANDOM_INTERNAL_RESTRICT other) {
    sum += other.sum;
    indices.reserve(indices.size() + other.indices.size());
    indices.insert(indices.end(),
                   std::make_move_iterator(other.indices.begin()),
                   std::make_move_iterator(other.indices.end()));
  }

  result_type sum;
  std::vector<mapped_type> indices;
};

// Solution<>
//
// Structure that holds a partial solution used in the Karmarkar-Karp algorithm.
template <typename Index, typename Size, typename UnaryOp>
  requires(flatflow::data::internal::Unsigned<Index> &&
           flatflow::data::internal::Unsigned<Size> &&
           std::invocable<UnaryOp, Size>)
struct Solution {
  using key_type = Size;
  using mapped_type = Index;
  using result_type = std::invoke_result_t<UnaryOp, Size>;

  ABSL_ATTRIBUTE_NOINLINE explicit Solution(
      std::span<const std::pair<const key_type, mapped_type>> items,
      UnaryOp op) {
    subsets.reserve(items.size());
    for (const auto &item : items) {
      subsets.emplace_back(item, op);
    }
    difference = subsets.at(subsets.size() - 1).sum - subsets.at(0).sum;
  }

  inline ABSL_ATTRIBUTE_ALWAYS_INLINE bool operator<(
      const Solution &ABSL_RANDOM_INTERNAL_RESTRICT other) const noexcept {
    return difference < other.difference;
  }

  ABSL_ATTRIBUTE_NOINLINE void Combine(
      const Solution &ABSL_RANDOM_INTERNAL_RESTRICT other) {
    #pragma omp parallel for
    for (std::size_t index = 0; index < subsets.size(); ++index) {
      subsets.at(index).Join(other.subsets.at(subsets.size() - index - 1));
    }

    std::sort(std::execution::par, subsets.begin(), subsets.end());
    difference = subsets.at(subsets.size() - 1).sum - subsets.at(0).sum;
  }

  result_type difference;
  std::vector<Subset<mapped_type, key_type, UnaryOp>> subsets;
};

}  // namespace

// KarmarkarKarp()
//
// Partitions the given items using the Balanced Largest Differencing Method
// (BLDM) of Michiels, Aarts, Korst, van Leeuwen and Spieksma from the paper
// `Computer-assisted proof of performance ratios for the Differencing Method
// <https://www.sciencedirect.com/science/article/pii/S1572528611000508>`,
// a variant of LDM for balanced number partitioning with larger cardinalities.
template <typename Index, typename Size, typename UnaryOp>
  requires(flatflow::data::internal::Unsigned<Index> &&
           flatflow::data::internal::Unsigned<Size> &&
           std::invocable<UnaryOp, Size>)
ABSL_ATTRIBUTE_NOINLINE auto KarmarkarKarp(
    const std::vector<std::pair<const Size, Index>> &ABSL_RANDOM_INTERNAL_RESTRICT items,
    const Index &ABSL_RANDOM_INTERNAL_RESTRICT num_micro_batches, UnaryOp op)
    -> std::vector<
        std::pair<std::invoke_result_t<UnaryOp, Size>, std::vector<Index>>> {
  CHECK_NE(num_micro_batches, 0);

  const auto stride = static_cast<std::size_t>(num_micro_batches);
  CHECK_EQ(items.size() % stride, 0);

  const auto now = omp_get_wtime();

  // Initially, BLDM starts with a sequence of `k` partial solutions, where each
  // partial solution is obtained from the `m` smallest remaining items.
  auto solutions = std::priority_queue<Solution<Index, Size, UnaryOp>>();

  for (std::size_t index = 0; index < items.size(); index += stride) {
    solutions.emplace(std::span(items).subspan(index, stride), op);
  }

  // Next, the algorithm selects two partial solutions from the sequence, for
  // which the difference between the maximum and minimum subset sum is largest.
  // These two solutions are combined into a new partial solution by joining the
  // subset with the smallest sum in one solution with the subset with the
  // largest sum in another solution, the subset with the second smallest sum in
  // one solution with the subset with the second largest sum in another
  // solution, and so on. This process is called differencing the solutions.
  // The combined solution replaces the two solutions in the sequence, and we
  // iterate this differencing operation until only one solution in the sequence
  // remains, which is the balanced solution obtained by BLDM.
  while (1 < solutions.size()) {
    auto solution = solutions.top();
    solutions.pop();

    solution.Combine(solutions.top());
    solutions.pop();

    solutions.emplace(solution);
  }

  auto micro_batches = std::vector<
      std::pair<std::invoke_result_t<UnaryOp, Size>, std::vector<Index>>>();
  micro_batches.reserve(solutions.top().subsets.size());

  for (const auto &subset : solutions.top().subsets) {
    micro_batches.emplace_back(subset.sum, std::move(subset.indices));
  }

  LOG(INFO) << absl::StrFormat("Partitioning %u items into %u subsets took %fs", items.size(), num_micro_batches, omp_get_wtime() - now);

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
