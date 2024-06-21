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

#ifndef FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_CONCAT_H_
#define FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_CONCAT_H_

#include <cassert>
#include <iterator>
#include <vector>

namespace flatflow {
namespace scheduler {
namespace internal {
namespace algorithm {

// concat()
//
// Concatenates each row of the given two matrices.
template <typename T>
inline void concat(std::vector<std::vector<T>> &lhs,
                   const std::vector<std::vector<T>> &rhs) {
  const auto data_parallel_size = lhs.size();
  assert(data_parallel_size == rhs.size());

  #pragma omp parallel for
  for (std::size_t rank = 0; rank < data_parallel_size; ++rank) {
    lhs[rank].reserve(lhs[rank].size() + rhs[rank].size());
    lhs[rank].insert(lhs[rank].cend(),
                     std::make_move_iterator(rhs[rank].begin()),
                     std::make_move_iterator(rhs[rank].end()));
  }
}

}  // namespace algorithm
}  // namespace internal
}  // namespace scheduler
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_CONCAT_H_
