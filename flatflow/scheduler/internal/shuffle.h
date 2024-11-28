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
#include <cassert>
#include <execution>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

#include "flatflow/aten/generator.h"
#include "flatflow/data/internal/types.h"

namespace flatflow {
namespace internal {

// shuffle()
//
// After scheduling, a `flatflow::Scheduler<>` shuffles between batches, which
// we call inter-batch shuffling. This enables shuffling not only between data
// samples with the same size but also between scheduled batches. It uses the
// same pseudo-random number generator and random seed as `flatflow::Dataset<>`
// for deterministic shuffling.
template <typename R, typename Index, typename Size>
  requires(Numerical<R> && Unsigned<Index> && Unsigned<Size>)
std::vector<std::vector<std::pair<Size, Index>>> shuffle(
    const std::vector<std::pair<R, std::vector<std::pair<Size, Index>>>> &micro_batches,
    Index seed, bool use_flat_shuffle) {
  const auto num_micro_batches = micro_batches.size();
  assert(num_micro_batches != 0);

  // The inter-batch shuffling is carried out as follows:
  //
  // * First, group micro-batches with the same makespan to minimize computation
  //   stall. This forms shuffling ranges.
  // * Second, shuffle between these shuffling ranges if the user wants the data
  //   samples to be more shuffled; i.e., when `use_flat_shuffle` is set to
  //   `false`.
  // * Finally, shuffle the micro-batches within each shuffling range.
  //   To minimize memory movement, shuffle the indices first and then project
  //   the micro-batches onto the shuffled indices. Note that shuffling indices
  //   first and then mapping the data to the shuffled indices yields the same
  //   result as directly shuffling the data.
  auto offsets = std::vector<std::size_t>();
  offsets.emplace_back(0);

  for (std::size_t offset = 1; offset < num_micro_batches; ++offset) {
    if (micro_batches[offset - 1].first != micro_batches[offset].first) {
      offsets.emplace_back(offset);
    }
  }

  auto indices = std::vector<std::size_t>(num_micro_batches);
  std::iota(indices.begin(), indices.end(), 0);

  auto ranges = std::vector<std::pair<std::vector<std::size_t>::iterator,
                                      std::vector<std::size_t>::iterator>>();
  ranges.reserve(offsets.size());

  for (std::size_t index = 0; index < offsets.size(); ++index) {
    const auto begin = std::next(indices.begin(), offsets[index]);
    if (index < offsets.size() - 1) {
      const auto end = std::next(indices.begin(), offsets[index + 1]);
      ranges.emplace_back(begin, end);
    } else {
      const auto end = indices.end();
      ranges.emplace_back(begin, end);
    }
  }

  if (!use_flat_shuffle) {
    auto generator = flatflow::aten::Generator(seed);
    std::shuffle(ranges.begin(), ranges.end(), generator);
  }

  std::for_each(std::execution::par, ranges.cbegin(), ranges.cend(),
                [&](const auto &range) {
                  const auto begin = range.first;
                  const auto end = range.second;
                  auto generator = flatflow::aten::Generator(seed);
                  std::shuffle(begin, end, generator);
                });

  auto shuffled = std::vector<std::vector<std::pair<Size, Index>>>();
  shuffled.reserve(num_micro_batches);

  std::for_each(std::execution::seq, ranges.cbegin(), ranges.cend(),
                [&](const auto &range) {
                  std::for_each(std::execution::seq, range.first, range.second,
                                [&](const auto index) {
                                  shuffled.emplace_back(
                                      std::move(micro_batches[index].second));
                                });
                });

  return shuffled;
}

}  // namespace internal
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_INTERNAL_SHUFFLE_H_
