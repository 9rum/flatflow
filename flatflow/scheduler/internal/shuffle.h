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

#ifndef FLATFLOW_SCHEDULER_INTERNAL_SHUFFLE_H_
#define FLATFLOW_SCHEDULER_INTERNAL_SHUFFLE_H_

#include <algorithm>
#include <execution>
#include <iterator>
#include <random>
#include <utility>
#include <vector>

#include "flatflow/data/internal/types.h"
#include "flatflow/scheduler/internal/partition.h"

namespace flatflow {
namespace internal {

// shuffle()
//
// After partitioning, a `flatflow::Scheduler<>` shuffles between batches, which
// we call inter-batch shuffling. This enables shuffling not only between data
// samples with the same size but also between scheduled batches. We use the
// same pseudo-random number generator and random seed as `flatflow::Dataset<>`
// for deterministic shuffling.
template <typename IndexType, typename R>
  requires(Unsigned<IndexType> && Numerical<R>)
void shuffle(std::vector<Subset<IndexType, R>> &subsets, IndexType seed) {
  using iterator = typename std::vector<Subset<IndexType, R>>::iterator;

  if (subsets.empty()) {
    return;
  }

  const auto num_subsets = subsets.size();

  // The inter-batch shuffling first groups subsets (i.e., micro-batches) with
  // the same subset sum to minimize computation stalls. This forms shuffling
  // ranges. Then, subsets within each shuffling range are shuffled.
  auto offsets = std::vector<std::size_t>();
  offsets.emplace_back(0);

  for (std::size_t offset = 1; offset < num_subsets; ++offset) {
    if (subsets[offset - 1].sum() != subsets[offset].sum()) {
      offsets.emplace_back(offset);
    }
  }

  auto ranges = std::vector<std::pair<iterator, iterator>>();
  ranges.reserve(offsets.size());

  for (std::size_t index = 0; index < offsets.size() - 1; ++index) {
    const auto begin = std::next(subsets.begin(), offsets[index]);
    const auto end = std::next(subsets.begin(), offsets[index + 1]);
    ranges.emplace_back(begin, end);
  }
  ranges.emplace_back(std::next(subsets.begin(), offsets.back()),
                      subsets.end());

  std::for_each(std::execution::par, ranges.cbegin(), ranges.cend(),
                [&](const auto &range) {
                  const auto begin = range.first;
                  const auto end = range.second;
                  auto generator = std::mt19937();
                  generator.seed(seed);
                  std::shuffle(begin, end, generator);
                });
}

}  // namespace internal
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_INTERNAL_SHUFFLE_H_
