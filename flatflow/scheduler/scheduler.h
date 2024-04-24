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
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/vector.h"

#include "flatflow/data/dataset.h"
#include "flatflow/data/internal/types.h"
#include "flatflow/scheduler/internal/algorithm/reshape.h"

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
  // In addition to the below constructor to set up scheduling,
  // a `flatflow::scheduler::Scheduler<>` supports copy and move constructors
  // and assignment operators; the default constructor, on the other hand, is
  // not available since the scheduler is initialized using `std::variant` and
  // `std::monostate` to select one of several schedule kinds at runtime without
  // dynamic dispatch overhead.
  inline explicit Scheduler(
      const flatbuffers::Vector<key_type, value_type> *sizes,
      const value_type &world_size, const value_type &batch_size,
      const value_type &seed)
      : world_size_(world_size), batch_size_(batch_size), seed_(seed) {
    CHECK(batch_size != 0);
    CHECK(batch_size % world_size == 0);
    CHECK(sizes->size() != 0);
    CHECK(sizes->size() % world_size == 0);

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

    const auto now = omp_get_wtime();

    // Since the sum of sizes can exceed double precision range, one has to
    // compute the partial means and then reduce them to minimize precison loss.
    constexpr auto stride =
        static_cast<value_type>(/*1 << 53=*/0x20000000000000) /
        static_cast<value_type>(std::numeric_limits<key_type>::max());

    auto means = std::vector<long double>(
        static_cast<std::size_t>((sizes->size() - 1) / stride + 1));

    // CAVEATS
    //
    // The `omp_set_nested()` routine has been deprecated.
    // The nested loops below may or may not be fully parallelized.
    omp_set_nested(1);

    #pragma omp parallel for
    for (value_type index = 0; index < sizes->size(); index += stride) {
      auto sum = static_cast<uint_fast64_t>(0);

      #pragma omp parallel for reduction(+ : sum)
      for (value_type offset = index;
           offset < std::min(index + stride, sizes->size()); ++offset) {
        sum += static_cast<uint_fast64_t>(sizes->Get(offset));
      }
      // Since the sum of each block is guaranteed to be less than 1 << 53,
      // it is safe to cast it to long double without precision loss even if
      // long double is not implemented.
      means.at(static_cast<std::size_t>(index / stride)) =
          static_cast<long double>(sum) /
          static_cast<long double>(sizes->size());
    }
    mean_ = static_cast<double>(
        std::reduce(std::execution::par, means.cbegin(), means.cend()));

    LOG(INFO) << absl::StrFormat("Block averaging took %fs", omp_get_wtime() - now);

    // TODO: Initialize data set and scheduler concurrently.
    //
    // The below copy assignment is actually not copied but direct-initialized
    // by copy elision.
    dataset_ = flatflow::data::Dataset(sizes, seed);
  }

  Scheduler() = delete;

  inline Scheduler(const Scheduler &other) = default;

  inline Scheduler &operator=(const Scheduler &other) = default;

  inline Scheduler(Scheduler &&other) = default;

  inline Scheduler &operator=(Scheduler &&other) = default;

  // Scheduler::schedule()
  //
  // Schedules and shuffles batches for the next training epoch to each of the
  // workers. This schedule kind discards the scheduling interval, as static
  // scheduling occurs at the granularity of epoch.
  inline auto schedule([[maybe_unused]] const value_type &interval)
      -> std::vector<std::vector<value_type>> {
    const auto now = omp_get_wtime();

    auto batches = std::vector<std::vector<std::vector<value_type>>>();
    batches.reserve(static_cast<std::size_t>(last_batch_ + 1));

    for (; batches.size() < batches.capacity();) {
      batches.emplace_back(std::move(schedule()));
    }

    // After scheduling, a `flatflow::scheduler::Scheduler<>` shuffles between
    // batches, which we call inter-batch shuffling. This enables shuffling not
    // only between data samples with the same size but also between scheduled
    // batches. It uses the same pseudorandom number generator and random seed
    // as `flatflow::data::Dataset` for deterministic shuffling.
    auto generator = std::ranlux48();
    generator.seed(static_cast<uint_fast64_t>(seed_ + epoch_));
    std::shuffle(batches.begin(), batches.end(), generator);

    LOG(INFO) << absl::StrFormat("Scheduling %u steps took %fs", last_batch_ + 1, omp_get_wtime() - now);

    return internal::algorithm::reshape(batches);
  }

  // Scheduler::on_batch_begin()
  //
  // A callback to be called at the beginning of a training batch.
  inline void on_batch_begin(const value_type &batch) const noexcept {
    dataset_.on_batch_begin(batch);
  }

  // Scheduler::on_batch_end()
  //
  // A callback to be called at the end of a training batch.
  inline void on_batch_end(const value_type &batch) const noexcept {
    dataset_.on_batch_end(batch);
  }

  // Scheduler::on_epoch_begin()
  //
  // A callback to be called at the beginning of an epoch.
  inline void on_epoch_begin(const value_type &epoch, const value_type &rank) {
    if (rank == 0) {
      step_ = 0;
      epoch_ = epoch;
      dataset_.on_epoch_begin(epoch);
    }
  }

  // Scheduler::on_epoch_end()
  //
  // A callback to be called at the end of an epoch.
  inline void on_epoch_end(const value_type &epoch, const value_type &rank) {
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
    const auto bin_size = static_cast<uint_fast64_t>(
        std::llround(mean_ * static_cast<double>(local_batch_size)));

    auto bins =
        std::vector<uint_fast64_t>(static_cast<std::size_t>(world_size_), 0);
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
      const auto rank = batches.size();

      auto batch = std::vector<value_type>();
      batch.reserve(static_cast<std::size_t>(local_batch_size));

      for (; batch.size() < batch.capacity();) {
        const auto underflow_safe_key =
            bin_size < bins.at(rank) ? 0 : bin_size - bins.at(rank);
        const auto safe_key =
            static_cast<uint_fast64_t>(std::numeric_limits<key_type>::max()) <
                    underflow_safe_key
                ? std::numeric_limits<key_type>::max()
                : static_cast<key_type>(underflow_safe_key);
        const auto [index, size] = dataset_.find(safe_key);
        batch.emplace_back(index);
        bins.at(rank) += static_cast<uint_fast64_t>(size);
      }
      batches.emplace_back(std::move(batch));
    }

    return batches;
  }

  double mean_;
  value_type world_size_, batch_size_, last_batch_, last_batch_size_, step_,
      epoch_, seed_;
  flatflow::data::Dataset<value_type, key_type> dataset_;
};

}  // namespace scheduler
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_SCHEDULER_H_
