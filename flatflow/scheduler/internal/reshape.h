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

#ifndef FLATFLOW_SCHEDULER_INTERNAL_RESHAPE_H_
#define FLATFLOW_SCHEDULER_INTERNAL_RESHAPE_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

#include "flatflow/data/internal/types.h"

namespace flatflow {
namespace internal {

// reshape()
//
// Distributes the given shuffled micro-batches to each of the workers.
template <typename Index, typename Size>
  requires(Unsigned<Index> && Unsigned<Size>)
std::vector<std::vector<std::pair<Size, Index>>> reshape(
    const std::vector<std::vector<std::pair<Size, Index>>> &micro_batches,
    Index world_size, Index global_batch_size) {
  assert(world_size != 0);
  assert(global_batch_size != 0);
  assert(global_batch_size % world_size == 0);

  const auto num_micro_batches = micro_batches.size();
  assert(num_micro_batches != 0);
  assert(num_micro_batches % world_size == 0);

  const auto micro_batch_size = micro_batches.front().size();
  assert(micro_batch_size != 0);
  assert(global_batch_size / world_size % micro_batch_size == 0);

  // To minimize both computation stalls across pipeline stages and
  // synchronization latency between pipelines, we distribute the shuffled
  // micro-batches at the granularity of mini-batch:
  //
  // * In pipeline parallelism, all pipeline stages should have the same
  //   execution time, so micro-batches are first distributed to the same
  //   pipeline.
  // * On the other hand, in data parallelism, synchronization latency between
  //   pipelines hinders scalability (in both synchronous pipeline schedules
  //   such as GPipe and asynchronous pipeline schedules such as PipeDream),
  //   so micro-batches are then distributed to other pipelines.
  //
  // Such distribution policy that prioritizes pipeline parallelism is due to
  // the fact that computation stalls occur for each pipeline stage while
  // synchronization latency occurs only for each batch.
  const auto last_global_batch_size =
      (micro_batch_size * num_micro_batches - 1) % global_batch_size + 1;
  const auto stride = global_batch_size / world_size / micro_batch_size;
  const auto last_stride =
      last_global_batch_size / world_size / micro_batch_size;
  const auto last_batch_offset =
      num_micro_batches / stride / world_size * stride * world_size;
  const auto num_samples = num_micro_batches / world_size * micro_batch_size;

  auto reshaped = std::vector<std::vector<std::pair<Size, Index>>>();
  reshaped.reserve(world_size);

  while (reshaped.size() < reshaped.capacity()) {
    reshaped.emplace_back(
        std::move(std::vector<std::pair<Size, Index>>(num_samples)));
  }

  #pragma omp parallel for
  for (std::size_t offset = 0; offset < num_micro_batches; ++offset) {
    const auto &micro_batch = micro_batches[offset];

    if (offset < last_batch_offset) {
      const auto rank = offset / stride % world_size;
      const auto index =
          (offset / stride / world_size * stride + offset % stride) *
          micro_batch_size;
      std::move(micro_batch.begin(), micro_batch.end(),
                std::next(reshaped[rank].begin(), index));
    } else {
      const auto rank = (offset - last_batch_offset) / last_stride % world_size;
      const auto index = (last_batch_offset / world_size +
                          (offset - last_batch_offset) % last_stride) *
                         micro_batch_size;
      std::move(micro_batch.begin(), micro_batch.end(),
                std::next(reshaped[rank].begin(), index));
    }
  }

  return reshaped;
}

}  // namespace internal
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_INTERNAL_RESHAPE_H_
