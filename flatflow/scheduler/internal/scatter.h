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

#ifndef FLATFLOW_SCHEDULER_INTERNAL_SCATTER_H_
#define FLATFLOW_SCHEDULER_INTERNAL_SCATTER_H_

#include <algorithm>
#include <execution>
#include <iterator>

#include "absl/log/check.h"

#include "flatflow/scheduler/internal/partition.h"

namespace flatflow {
namespace internal {

// Scatter()
//
// Distributes the given items in the range [`first`, `last`) into `m` subsets.
// The items can be grouped in a strided manner, and the resulting subsets are
// stored in another range beginning at `result`, also in a strided manner.
template <typename InputIt, typename OutputIt, typename F, typename Proj>
OutputIt Scatter(
    InputIt first, InputIt last, OutputIt result, F func, Proj proj,
    typename std::iterator_traits<InputIt>::difference_type m,
    typename std::iterator_traits<InputIt>::difference_type stride) {
  const auto n = std::distance(first, last);

  if (n == 0) {
    return result;
  }

  CHECK_NE(m, 0);
  CHECK_EQ(n % m, 0);
  CHECK_EQ(stride % m, 0);

  auto d_last = result;

  // clang-format off
  #pragma omp parallel for reduction(max : d_last)
  for (typename std::iterator_traits<InputIt>::difference_type offset = 0;
       offset < n; offset += stride) {
    const auto d_first = std::next(result, offset / stride * m);
    d_last = Partition(std::next(first, offset),
                       std::next(first, std::min(offset + stride, n)), d_first,
                       func, proj, m);

    // To minimize both computation stalls across pipeline stages and
    // synchronization latency between pipelines, we distribute the shuffled
    // micro-batches at the granularity of mini-batch:
    //
    // * In data parallelism, synchronization latency between pipelines hinders
    //   scalability (in both synchronous pipeline schedules such as GPipe and
    //   asynchronous pipeline schedules such as PipeDream), so micro-batches
    //   are re-partitioned into each of the pipelines.
    // * On the other hand, in pipeline parallelism, earlier pipeline stage
    //   should take less execution time than the subsequent one, so
    //   micro-batches are then sorted in the same pipeline.
    std::for_each(std::execution::par, d_first, d_last, [](auto &subset) {
      std::sort(std::execution::par, subset.items().begin(),
                subset.items().end());
    });
  }
  // clang-format on

  return d_last;
}

}  // namespace internal
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_INTERNAL_SCATTER_H_
