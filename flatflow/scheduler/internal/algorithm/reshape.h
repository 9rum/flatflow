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

#ifndef FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_RESHAPE_H_
#define FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_RESHAPE_H_

#include <algorithm>
#include <execution>
#include <utility>
#include <vector>

#include "absl/log/check.h"

namespace flatflow {
namespace scheduler {
namespace internal {
namespace algorithm {

// reshape()
//
// Converts the given three-dimensional tensor to a corresponding
// two-dimensional tensor or a matrix.
template <typename T>
inline auto reshape(const std::vector<std::vector<std::vector<T>>> &tensor)
    -> std::vector<std::vector<T>> {
  const auto interval = tensor.size();
  CHECK(0 < interval);

  const auto world_size = tensor.at(0).size();
  CHECK(0 < world_size);

  auto matrix = std::vector<std::vector<T>>();
  matrix.reserve(world_size);

  const auto row_size = tensor.at(0).at(0).size() * (interval - 1) +
                        tensor.at(interval - 1).at(0).size();

  for (; matrix.size() < matrix.capacity();) {
    matrix.emplace_back(std::move(std::vector<T>(row_size)));
  }

  // TODO: Flatten the below nested loops and parallelize it in a bulk
  // synchronous parallel manner.
  #pragma omp parallel for
  for (std::size_t rank = 0; rank < world_size; ++rank) {
    auto dest = matrix.at(rank).begin();
    std::for_each(std::execution::seq, tensor.cbegin(), tensor.cend(),
                  [&](const auto &batch) {
                    dest = std::copy(batch.at(rank).cbegin(),
                                     batch.at(rank).cend(), dest);
                  });
  }

  return matrix;
}

}  // namespace algorithm
}  // namespace internal
}  // namespace scheduler
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_RESHAPE_H_
