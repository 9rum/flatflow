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
#include <cstddef>
#include <execution>
#include <iterator>
#include <random>
#include <utility>
#include <vector>

namespace flatflow {
namespace internal {

// Shuffle()
//
// After partitioning, a `flatflow::Scheduler<>` shuffles between batches, which
// we call inter-batch shuffling. This enables shuffling not only between data
// samples with the same size but also between scheduled batches. We use the
// same pseudo-random number generator and random seed as `flatflow::Dataset<>`
// for deterministic shuffling.
template <typename RandomIt>
void Shuffle(RandomIt first, RandomIt last, std::mt19937::result_type seed) {
  const auto m = std::distance(first, last);

  // The inter-batch shuffling first groups subsets (i.e., micro-batches) with
  // the same subset sum to minimize computation stalls. This forms shuffling
  // ranges. Then, subsets within each shuffling range are shuffled.
  auto offsets =
      std::vector<typename std::iterator_traits<RandomIt>::difference_type>();
  offsets.emplace_back(0);

  for (typename std::iterator_traits<RandomIt>::difference_type offset = 1;
       offset < m; ++offset) {
    if (*std::next(first, offset - 1) < *std::next(first, offset)) {
      offsets.emplace_back(offset);
    }
  }

  auto ranges = std::vector<std::pair<RandomIt, RandomIt>>();
  ranges.reserve(offsets.size());

  for (std::size_t index = 0; index < offsets.size() - 1; ++index) {
    ranges.emplace_back(std::next(first, offsets[index]),
                        std::next(first, offsets[index + 1]));
  }
  ranges.emplace_back(std::next(first, offsets.back()), last);

  std::for_each(std::execution::par, ranges.cbegin(), ranges.cend(),
                [&](const auto &range) {
                  auto generator = std::mt19937();
                  generator.seed(seed);
                  std::shuffle(range.first, range.second, generator);
                });
}

}  // namespace internal
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_INTERNAL_SHUFFLE_H_
