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

#ifndef FLATFLOW_SCHEDULER_SCHEDULER_H_
#define FLATFLOW_SCHEDULER_SCHEDULER_H_

#include <utility>
#include <vector>

#include <flatbuffers/vector.h>
#include <omp.h>

#include "flatflow/data/dataset.h"
#include "flatflow/data/internal/types.h"

namespace flatflow {
namespace scheduler {

/// \brief A `flatflow::scheduler::Schedule` is a scheduling policy on how to
/// distribute the given data. In addition to static scheduling that reduces
/// workload imbalance, it supports dynamic scheduling that adaptively adjusts
/// the workload on each worker. Guided scheduling, on the other hand, provides
/// profile-guided optimization for Transformer-based models.
enum class Schedule : uint8_t {
  kStatic,
  kDynamic,
  kGuided,
};

/// \brief A common base class for all scheduler implementations. Each schedule
/// kind has its own partial template specialization, and here static scheduler
/// is implemented. Note that static scheduling is only effective for models
/// with linear complexity in the size of each data sample: traditional
/// convolutional neural networks (CNNs) and state space models (SSMs) in Mamba
/// family that implement linear-time sequence modeling are of this kind.
/// \tparam Index The data type of the values in data set.
/// \tparam Size The data type of the keys in data set.
/// \tparam Kind The schedule kind on how to distribute the given data.
template <typename Index, typename Size, Schedule Kind = Schedule::kStatic>
  requires(flatflow::data::internal::Unsigned<Index> &&
           flatflow::data::internal::Unsigned<Size>)
class Scheduler {
 public:
  using key_type = Size;
  using value_type = Index;

  /// \brief Constructor to prepare for static scheduling.
  /// \param sizes A mapping from an index to the relative size of the
  /// corresponding data sample.
  /// \param world_size Total number of workers participating in the job.
  /// \param batch_size How many samples to train per batch.
  /// \param seed A random seed used for selective shuffling.
  inline explicit Scheduler(
      const flatbuffers::Vector<key_type, value_type> *sizes,
      value_type world_size, value_type batch_size, value_type seed)
      : world_size_(world_size), batch_size_(batch_size), seed_(seed) {
    // (x - 1) / y is always equal to x % y == 0 ? x / y - 1 : x / y without any
    // branch instructions.
    last_batch_ = (sizes->size() - 1) / batch_size;

    // The last batch size must be calculated since the total number of data
    // samples may not be a multiple of batch size, while both are multiples of
    // world size.
    //
    // (x - 1) % y + 1 is always equal to x % y == 0 ? y : x % y without any
    // branch instructions.
    last_batch_size_ = (sizes->size() - 1) % batch_size + 1;

    mean_ = 0.0;
    #pragma omp parallel for reduction(+ : mean_)
    for (value_type index = 0; index < sizes->size(); ++index) {
      mean_ += static_cast<double>(sizes->Get(index)) / sizes->size();
    }

    dataset_ = std::move(flatflow::data::Dataset(sizes, seed));
  }

  /// \brief Assigns the next mini-batch to each of the workers.
  /// It adopts first-fit-decreasing (FFD), an approximately-optimal heuristic
  /// for bin packing.
  /// * FFD paper: https://dspace.mit.edu/bitstream/handle/1721.1/57819/17595570-MIT.pdf
  /// * Python implementation: https://github.com/erelsgl/prtpy/blob/ebe54010513ea725f7a3221e4aa0258afa15d6fb/prtpy/packing/first_fit.py
  /// \return The next mini-batch.
  inline std::vector<std::vector<value_type>> schedule() {
    const auto local_batch_size = step_ == last_batch_
                                      ? last_batch_size_ / world_size_
                                      : batch_size_ / world_size_;
    const auto bin_size =
        static_cast<key_type>(std::round(mean_ * local_batch_size));

    auto bins = std::vector<key_type>(static_cast<std::size_t>(world_size_), 0);
    auto batches = std::vector<std::vector<value_type>>();
    batches.reserve(static_cast<std::size_t>(world_size_));

    // Pack the bins in a first-fit-decreasing fashion.
    for (; batches.size() < batches.capacity();) {
      auto batch = std::vector<value_type>();
      batch.reserve(static_cast<std::size_t>(local_batch_size));
      for (; batch.size() < batch.capacity();) {
        const auto [index, size] =
            dataset_.find(bin_size - bins.at(batches.size()));
        batch.emplace_back(index);
        bins.at(batches.size()) += size;
      }
      batches.emplace_back(std::move(batch));
    }

    ++step_;

    return batches;
  }

 protected:
  double mean_;
  value_type world_size_, batch_size_, last_batch_, last_batch_size_, step_,
      seed_;
  flatflow::data::Dataset<value_type, key_type> dataset_;
};

}  // namespace scheduler
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_SCHEDULER_H_
