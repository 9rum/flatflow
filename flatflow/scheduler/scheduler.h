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

#include <omp.h>

#include <algorithm>
#include <execution>
#include <random>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/vector.h"

#include "flatflow/data/dataset.h"
#include "flatflow/data/internal/types.h"

namespace flatflow {
namespace scheduler {

// flatflow::scheduler::Schedule
//
// A `flatflow::scheduler::Schedule` is a scheduling policy on how to distribute
// the given data. In addition to static scheduling that reduces workload
// imbalance, it supports dynamic scheduling that adaptively adjusts the
// workload on each worker. Guided scheduling, on the other hand, provides
// profile-guided optimization for Transformer-based models.
enum class Schedule : uint8_t {
  kStatic,
  kDynamic,
  kGuided,
};

// flatflow::scheduler::Scheduler<>
//
// A common base class for all scheduler implementations. Each schedule kind has
// its own partial template specialization, and here, a static scheduler is
// implemented. Note that static scheduling is only effective for models with
// linear complexity in the size of each data sample: traditional convolutional
// neural networks (CNNs) and state space models (SSMs) in the Mamba family that
// implement linear-time sequence modeling are of this kind.
template <typename Index, typename Size, Schedule Kind>
  requires(flatflow::data::internal::Unsigned<Index> &&
           flatflow::data::internal::Unsigned<Size>)
class Scheduler {
 public:
  using key_type = Size;
  using value_type = Index;

  // Constructors and assignment operators
  //
  // A `flatflow::scheduler::Scheduler<>` does not allow multiple schedulers to
  // exist at the same time. That is, copy constructor and copy assignment
  // operator cannot be used; the default constructor is also not available
  // since a scheduler is initialized using `std::variant` and `std::monostate`
  // to select one of several schedule kinds at runtime without dynamic dispatch
  // overhead.
  Scheduler() = delete;

  // Constructor to prepare for static scheduling.
  inline explicit Scheduler(
      const flatbuffers::Vector<key_type, value_type> *sizes,
      value_type world_size, value_type batch_size, value_type seed)
      : world_size_(world_size), batch_size_(batch_size), seed_(seed) {
    CHECK(sizes->size() % world_size == 0)
        << "Total number of data samples must be a multiple of world size";
    CHECK(batch_size % world_size == 0)
        << "Batch size must be a multiple of world size";

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

  Scheduler(const Scheduler &) = delete;

  Scheduler &operator=(const Scheduler &) = delete;

  inline explicit Scheduler(Scheduler &&) = default;

  Scheduler &operator=(Scheduler &&) = default;

  // Scheduler::schedule()
  //
  // Schedules and shuffles batches for the next training epoch to each of the
  // workers. This schedule kind discards the scheduling interval, as static
  // scheduling occurs at the granularity of epoch.
  inline auto schedule([[maybe_unused]] value_type interval)
      -> std::vector<std::vector<value_type>> {
    const auto now = omp_get_wtime();

    auto batches = std::vector<std::vector<std::vector<value_type>>>();
    batches.reserve(static_cast<std::size_t>(last_batch_ + 1));

    for (; batches.size() < batches.capacity();) {
      batches.emplace_back(schedule());
    }

    // After scheduling, a `flatflow::scheduler::Scheduler` shuffles between
    // batches, which we call inter-batch shuffling. This enables shuffling not
    // only between data samples with the same size but also between scheduled
    // batches. It uses the same pseudorandom number generator and random seed
    // as `flatflow::data::Dataset` for deterministic shuffling.
    auto generator = std::ranlux48();
    generator.seed(static_cast<uint_fast64_t>(seed_ + epoch_));
    std::shuffle(batches.begin(), batches.end(), generator);

    const auto indices = reshape(batches);

    LOG(INFO) << absl::StrFormat("Scheduling %u steps took %f seconds", last_batch_ + 1, omp_get_wtime() - now);

    return indices;
  }

  // Scheduler::on_batch_begin()
  //
  // A callback to be called at the beginning of a training batch.
  inline void on_batch_begin(value_type batch) const noexcept {
    dataset_.on_batch_begin(batch);
  }

  // Scheduler::on_batch_end()
  //
  // A callback to be called at the end of a training batch.
  inline void on_batch_end(value_type batch) const noexcept {
    dataset_.on_batch_end(batch);
  }

  // Scheduler::on_epoch_begin()
  //
  // A callback to be called at the beginning of an epoch.
  inline void on_epoch_begin(value_type epoch, value_type rank) {
    if (rank == 0) {
      step_ = 0;
      epoch_ = epoch;
      dataset_.on_epoch_begin(epoch);
    }
  }

  // Scheduler::on_epoch_end()
  //
  // A callback to be called at the end of an epoch.
  inline void on_epoch_end(value_type epoch, value_type rank) {
    if (rank == 0) {
      dataset_.on_epoch_end(epoch);
    }
  }

  // Scheduler::on_train_begin()
  //
  // A callback to be called at the beginning of training.
  inline void on_train_begin() const noexcept { dataset_.on_train_begin(); }

  // Scheduler::on_train_end()
  //
  // A callback to be called at the end of training.
  inline void on_train_end() const noexcept { dataset_.on_train_end(); }

 protected:
  // Scheduler::schedule()
  //
  // Assigns the next batch to each of the workers. It adopts
  // first-fit-decreasing (FFD), a near-optimal heuristic for bin packing.
  // * FFD paper: https://dspace.mit.edu/bitstream/handle/1721.1/57819/17595570-MIT.pdf
  // * Python implementation: https://github.com/erelsgl/prtpy/blob/ebe54010513ea725f7a3221e4aa0258afa15d6fb/prtpy/packing/first_fit.py
  inline std::vector<std::vector<value_type>> schedule() {
    const auto local_batch_size = step_++ == last_batch_
                                      ? last_batch_size_ / world_size_
                                      : batch_size_ / world_size_;
    const auto bin_size =
        static_cast<key_type>(std::round(mean_ * local_batch_size));

    auto bins = std::vector<key_type>(static_cast<std::size_t>(world_size_), 0);
    auto batches = std::vector<std::vector<value_type>>();
    batches.reserve(static_cast<std::size_t>(world_size_));

    // Pack the bins in a first-fit-decreasing fashion.
    //
    // NOTE:
    //
    // Assigning each worker an equal number of data samples with the same sum
    // of sizes per batch is an online variation of multiway number partitioning
    // with number constraints, so called balanced multiway number partitioning.
    // As this problem is NP-hard, there are several approximation algorithms
    // such as largest differencing method (LDM or Karmarkar–Karp algorithm).
    // For now, we model this as a bin packing problem and use the FFD solver,
    // an approximately-optimal heuristic for bin packing with approximation
    // ratio of 11/9. The scheduling behavior may change in the future.
    //
    // * Partition problem: https://en.wikipedia.org/wiki/Partition_problem
    // * Balanced number partitioning: https://en.wikipedia.org/wiki/Balanced_number_partitioning
    // * Largest differencing method: https://en.wikipedia.org/wiki/Largest_differencing_method
    // * Python implementation for Karmarkar–Karp algorithm: https://github.com/erelsgl/prtpy/blob/3ab0facffebc758c49bb3d06bd94a5f140a99863/prtpy/partitioning/karmarkar_karp.py
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

    return batches;
  }

  // Scheduler::reshape()
  //
  // Converts the given three-dimensional tensor to a corresponding
  // two-dimensional tensor or a matrix.
  inline std::vector<std::vector<value_type>> reshape(
      const std::vector<std::vector<std::vector<value_type>>> &tensor) const {
    auto matrix = std::vector<std::vector<value_type>>();
    matrix.reserve(static_cast<std::size_t>(world_size_));

    for (; matrix.size() < matrix.capacity();) {
      const auto row = std::vector<value_type>(static_cast<std::size_t>(
          (batch_size_ * last_batch_ + last_batch_size_) / world_size_));
      matrix.emplace_back(std::move(row));
    }

    #pragma omp parallel for
    for (std::size_t rank = 0; rank < matrix.size(); ++rank) {
      auto dest = matrix.at(rank).begin();
      std::for_each(std::execution::seq, tensor.cbegin(), tensor.cend(),
                    [&](const auto &batch) {
                      dest = std::copy(batch.at(rank).cbegin(),
                                       batch.at(rank).cend(), dest);
                    });
    }

    return matrix;
  }

  double mean_;
  value_type world_size_, batch_size_, last_batch_, last_batch_size_, step_,
      epoch_, seed_;
  flatflow::data::Dataset<value_type, key_type> dataset_;
};

}  // namespace scheduler
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_SCHEDULER_H_
