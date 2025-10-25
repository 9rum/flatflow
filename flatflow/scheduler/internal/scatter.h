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

#include <omp.h>

#include <algorithm>
#include <iterator>

#include "absl/log/check.h"

namespace flatflow {
namespace internal {

// Scatter()
//
// Sends data in the strided range [`first`, `last`) to the calling process with
// rank `rank` in a group of size `n`, storing the result in an output range
// starting from `result`.
template <typename InputIterator, typename OutputIterator>
OutputIterator Scatter(InputIterator first, InputIterator last,
                       OutputIterator result,
                       std::iter_difference_t<InputIterator> n,
                       std::iter_difference_t<InputIterator> rank,
                       std::iter_difference_t<InputIterator> stride) {
  const auto total_size = std::distance(first, last);

  if (total_size == 0) {
    return result;
  }

  CHECK_NE(n, 0);
  CHECK_NE(stride, 0);

  const auto remainder = (total_size - 1) % stride + 1;

  // clang-format off
  #pragma omp parallel for num_threads(omp_get_num_procs())
  for (std::iter_difference_t<InputIterator> offset = 0; offset < total_size;
       offset += stride) {
    const auto step = (offset + stride < total_size ? stride : remainder) / n;
    const auto base = std::next(first, offset + step * rank);
    std::move(base, std::next(base, step), std::next(result, offset / n));
  }
  // clang-format on

  return std::next(result, total_size / n);
}

}  // namespace internal
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_INTERNAL_SCATTER_H_
