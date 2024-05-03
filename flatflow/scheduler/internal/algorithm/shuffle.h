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

#ifndef FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_SHUFFLE_H_
#define FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_SHUFFLE_H_

#include <algorithm>
#include <iterator>
#include <random>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/random/internal/platform.h"

namespace flatflow {
namespace scheduler {
namespace internal {
namespace algorithm {

// shuffle()
//
// After scheduling, a `flatflow::scheduler::Scheduler<>` shuffles between
// batches, which we call inter-batch shuffling. This enables shuffling not
// only between data samples with the same size but also between scheduled
// batches. It uses the same pseudorandom number generator and random seed
// as `flatflow::data::Dataset<>` for deterministic shuffling.
template <typename T>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline void shuffle(
    std::vector<std::vector<std::vector<T>>> &ABSL_RANDOM_INTERNAL_RESTRICT batches,
    const T &ABSL_RANDOM_INTERNAL_RESTRICT seed) {
  const auto interval = batches.size();
  CHECK_GT(interval, 0);

  const auto world_size = batches.at(0).size();
  CHECK_GT(world_size, 0);

  // When the batch size and last batch size are different from each other
  // (i.e., when remainder exists), the last batch should be excluded from
  // the shuffling range.
  auto end = batches.end();
  if (batches.at(0).at(0).size() != batches.at(interval - 1).at(0).size()) {
    std::advance(end, -1);
  }

  auto generator = std::ranlux48();
  generator.seed(static_cast<uint_fast64_t>(seed));
  std::shuffle(batches.begin(), end, generator);
}

}  // namespace algorithm
}  // namespace internal
}  // namespace scheduler
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_SHUFFLE_H_
