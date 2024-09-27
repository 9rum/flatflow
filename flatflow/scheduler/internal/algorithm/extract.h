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

#ifndef FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_EXTRACT_H_
#define FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_EXTRACT_H_

#include <cassert>
#include <utility>
#include <vector>

namespace flatflow {
namespace scheduler {
namespace internal {
namespace algorithm {

// extract()
//
// Splits the given pairs of size and index.
template <typename Index, typename Size>
std::pair<std::vector<std::vector<Index>>, std::vector<std::vector<Size>>>
extract(const std::vector<std::vector<std::pair<Size, Index>>> &items) {
  const auto world_size = items.size();
  assert(world_size != 0);

  const auto num_samples = items.front().size();
  assert(num_samples != 0);

  const auto num_items = world_size * num_samples;

  auto indices = std::vector<std::vector<Index>>();
  indices.reserve(world_size);

  auto sizes = std::vector<std::vector<Size>>();
  sizes.reserve(world_size);

  for (std::size_t rank = 0; rank < world_size; ++rank) {
    indices.emplace_back(std::move(std::vector<Index>(num_samples)));
    sizes.emplace_back(std::move(std::vector<Size>(num_samples)));
  }

  #pragma omp parallel for
  for (std::size_t index = 0; index < num_items; ++index) {
    const auto rank = index / num_samples;
    const auto offset = index % num_samples;
    indices[rank][offset] = items[rank][offset].second;
    sizes[rank][offset] = items[rank][offset].first;
  }

  return std::make_pair(indices, sizes);
}

}  // namespace algorithm
}  // namespace internal
}  // namespace scheduler
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_EXTRACT_H_
