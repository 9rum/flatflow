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
#include <functional>
#include <iterator>
#include <tuple>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"

#include "flatflow/ops/graph_generated.h"
#include "flatflow/ops/ops.h"
#include "flatflow/scheduler/internal/partition.h"

namespace flatflow {

// flatflow::Scheduler
//
// A common base class for all scheduler implementations. There may be
// several scheduling policies on how to rearrange the training sequence.
//
// Note that this scheduler implementation is optimized for balancing
// computational workloads, i.e., floating point operations. To this end,
// we provide an operator set and an associated operator registry; models
// containing operators not yet defined in the operator set may fail to run.
// See the note on how to register a new operator in `flatflow/ops/ops.h`.
// Other optimization objectives such as memory footprint may require separate
// scheduler implementations.
class Scheduler {
 public:
  using value_type =
      typename std::vector<typename SymIntAdaptor::return_type>::value_type;
  using size_type =
      typename std::vector<typename SymIntAdaptor::return_type>::size_type;

  // Constructors and assignment operators
  //
  // In addition to the constructor to set up scheduling, `flatflow::Scheduler`
  // supports a default constructor, as well as copy/move constructors and
  // assignment operators.
  Scheduler() {}

  template <typename InputIterator>
  Scheduler(size_type data_parallel_world_size, size_type global_batch_size,
            size_type micro_batch_size, InputIterator first, InputIterator last,
            const Graph *graph)
      : data_parallel_world_size_(data_parallel_world_size),
        global_batch_size_(global_batch_size),
        micro_batch_size_(micro_batch_size) {
    constexpr auto kZero = static_cast<size_type>(0);
    CHECK_NE(data_parallel_world_size, kZero);
    CHECK_NE(global_batch_size, kZero);
    CHECK_NE(micro_batch_size, kZero);
    CHECK_EQ(global_batch_size % (data_parallel_world_size * micro_batch_size),
             kZero);

    const auto total_size = static_cast<size_type>(std::distance(first, last));
    CHECK_NE(total_size, kZero);
    CHECK_EQ(total_size % data_parallel_world_size, kZero);
    CHECK_NE(graph, nullptr);

    LOG(INFO) << absl::StrFormat(
        "Initializing scheduler with the following arguments:\n"
        "  data_parallel_world_size: %u\n"
        "  global_batch_size:        %u\n"
        "  micro_batch_size:         %u",
        data_parallel_world_size, global_batch_size, micro_batch_size);

    // (x - 1) % y + 1 is always equal to x % y == 0 ? y : x % y without any
    // branch instructions.
    last_global_batch_size_ = (total_size - 1) % global_batch_size + 1;

    // The last micro-batch size be calculated since the total number of data
    // samples is guaranteed to be a multiple of data parallel world size, but
    // may not be divisible by the micro-batch size.
    last_micro_batch_size_ =
        (total_size / data_parallel_world_size - 1) % micro_batch_size + 1;

    // (x - 1) / y + 1 is always equal to x % y == 0 ? x / y : x / y + 1 without
    // any branch instructions.
    num_microbatches_ =
        ((total_size / data_parallel_world_size - 1) / micro_batch_size + 1) *
        data_parallel_world_size;

    projections_.resize(total_size);

    const auto trace = symbolic_trace(graph);

    // clang-format off
    #pragma omp parallel for
    for (size_type index = 0; index < total_size; ++index) {
      projections_[index] = trace(*std::next(first, index));
    }
    // clang-format on
  }

  Scheduler(const Scheduler &other) = default;

  Scheduler &operator=(const Scheduler &other) = default;

  Scheduler(Scheduler &&other) = default;

  Scheduler &operator=(Scheduler &&other) = default;

  // Scheduler::Schedule()
  //
  // Reorders the given computation schedule in the range [`first`, `last`) for
  // the next training epoch. The resulting indices are stored in an output
  // range starting from `result`.
  //
  // CAVEATS
  //
  // This scheduler implementation iteratively reorders the training sequence
  // at the granularity of mini-batch, which we call iterative reordering.
  // This may produce somewhat suboptimal training performance due to the
  // constraints in optimization scope, yet still maintains the as-if rule;
  // the observable behavior of the model remains transparent even after
  // reordering.
  template <typename InputIterator, typename OutputIterator>
  OutputIterator Schedule(InputIterator first, InputIterator last,
                          OutputIterator result) const {
    const auto now = omp_get_wtime();

    const auto total_size = static_cast<size_type>(std::distance(first, last));

    const auto comp = std::bind_front(&Scheduler::Compare, this);
    const auto proj = std::bind_front(&Scheduler::Project, this);
    const auto bproj = &internal::Subset<size_type, value_type>::sum;

    // clang-format off
    #pragma omp parallel for
    for (size_type offset = 0; offset < total_size;
         offset += global_batch_size_) {
      const auto num_samples = offset + global_batch_size_ < total_size
                                   ? global_batch_size_
                                   : last_global_batch_size_;

      // `samples` may not be sorted in order of their projected values;
      // sort them first for partitioning.
      auto samples = std::vector<size_type>(
          std::next(first, offset), std::next(first, offset + num_samples));
      std::sort(samples.begin(), samples.end(), comp);

      if (num_samples % (data_parallel_world_size_ * micro_batch_size_) == 0) {
        // If the given batch size is a multiple of both data parallel world
        // size and micro-batch size, the corresponding batch can be directly
        // reordered.
        //
        // NOTE: To minimize both pipeline bubbles across pipeline stages and
        // synchronization latency between pipelines, the given micro-batches
        // are reordered as follows:
        //
        // * In pipeline parallelism, all the micro-batches should take the same
        //   execution time to effectively utilize the massively parallel
        //   compute capability of accelerators, so we first partition the given
        //   batch at the granularity of micro-batch, which we call fine-grained
        //   partitioning. This is hard to be solved in polynomial time, since
        //   partitioning with cardinalities greater than two is NP-hard.
        //   We approximate this using the differencing method.
        //   In addition, we further reduce the pipeline bubbles by sorting
        //   micro-batches in the same pipeline in order of their execution
        //   time, making an earlier pipeline stage take less execution time
        //   than the subsequent one.
        // * On the other hand, in data parallelism, synchronization latency
        //   between pipelines hinders scalability (in both synchronous pipeline
        //   schedules such as GPipe and asynchronous pipeline schedules such as
        //   PipeDream), so we re-partition the resulting micro-batches into
        //   each of the pipelines, which we call coarse-grained partitioning.
        //
        // Note that these kinds of problems do not occur in tensor parallelism,
        // since it always equally distributes the given tensors such as
        // attention heads.
        auto num_microbatches = num_samples / micro_batch_size_;
        auto microbatches =
            std::vector<internal::Subset<size_type, value_type>>(
                num_microbatches);
        internal::Partition(samples.begin(), samples.end(),
                            microbatches.begin(), num_microbatches, proj);

        num_microbatches /= data_parallel_world_size_;
        auto batch = std::vector<internal::Subset<
            internal::Subset<size_type, value_type>, value_type>>(
            data_parallel_world_size_);
        internal::Partition(microbatches.begin(), microbatches.end(),
                            batch.begin(), data_parallel_world_size_, bproj);

        for (size_type rank = 0; rank < data_parallel_world_size_; ++rank) {
          // The partitioned per-replica batches are guaranteed to be sorted in
          // order of their projected values, while the micro-batches in each of
          // the replicas are not; they be sorted to prevent pipeline bubbles.
          auto &per_replica_batch = batch[rank];
          std::sort(per_replica_batch.begin(), per_replica_batch.end());

          const auto base =
              offset + num_samples / data_parallel_world_size_ * rank;

          for (size_type microbatch_id = 0; microbatch_id < num_microbatches;
               ++microbatch_id) {
            std::move(
                per_replica_batch[microbatch_id].begin(),
                per_replica_batch[microbatch_id].end(),
                std::next(result, base + micro_batch_size_ * microbatch_id));
          }
        }
      } else {
        // When the given batch size is not a multiple of both data parallel
        // world size and micro-batch size (that is, this is the last batch
        // and the last micro-batch size is different from the others), the
        // corresponding batch cannot be directly reordered. The remainders are
        // first reordered and then the quotients are in the same way as above.
        const auto num_remainders =
            data_parallel_world_size_ * last_micro_batch_size_;
        auto last_microbatches =
            std::vector<internal::Subset<size_type, value_type>>(
                data_parallel_world_size_);
        internal::Partition(std::prev(samples.end(), num_remainders),
                            samples.end(), last_microbatches.begin(),
                            data_parallel_world_size_, proj);

        for (size_type rank = 0; rank < data_parallel_world_size_; ++rank) {
          auto &per_replica_microbatch = last_microbatches[rank];

          const auto base = offset + (num_samples - num_remainders) /
                                         data_parallel_world_size_;

          std::move(
              per_replica_microbatch.begin(), per_replica_microbatch.end(),
              std::next(result,
                        base + num_samples / data_parallel_world_size_ * rank));
        }

        auto num_microbatches =
            (num_samples - num_remainders) / micro_batch_size_;
        auto microbatches =
            std::vector<internal::Subset<size_type, value_type>>(
                num_microbatches);
        internal::Partition(samples.begin(),
                            std::prev(samples.end(), num_remainders),
                            microbatches.begin(), num_microbatches, proj);

        num_microbatches /= data_parallel_world_size_;
        auto batch = std::vector<internal::Subset<
            internal::Subset<size_type, value_type>, value_type>>(
            data_parallel_world_size_);
        internal::Partition(microbatches.begin(), microbatches.end(),
                            batch.begin(), data_parallel_world_size_, bproj);

        for (size_type rank = 0; rank < data_parallel_world_size_; ++rank) {
          auto &per_replica_batch = batch[rank];
          std::sort(per_replica_batch.begin(), per_replica_batch.end());

          const auto base =
              offset + num_samples / data_parallel_world_size_ * rank;

          for (size_type microbatch_id = 0; microbatch_id < num_microbatches;
               ++microbatch_id) {
            std::move(
                per_replica_batch[microbatch_id].begin(),
                per_replica_batch[microbatch_id].end(),
                std::next(result, base + micro_batch_size_ * microbatch_id));
          }
        }
      }
    }

    LOG(INFO) << absl::StrFormat("Reordering %u micro-batches took %fs", num_microbatches_, omp_get_wtime() - now);
    // clang-format on

    return std::next(result, total_size);
  }

  // Scheduler::on_epoch_start()
  //
  // A callback to be called at the beginning of each training epoch.
  void on_epoch_start(size_type epoch) const noexcept { std::ignore = epoch; }

  // Scheduler::on_epoch_end()
  //
  // A callback to be called at the end of each training epoch.
  void on_epoch_end(size_type epoch) const noexcept { std::ignore = epoch; }

  // Scheduler::on_train_start()
  //
  // A callback to be called at the beginning of training.
  void on_train_start() const noexcept {}

  // Scheduler::on_train_end()
  //
  // A callback to be called at the end of training.
  void on_train_end() const noexcept {}

 private:
  // Helper functions for Schedule()
  //
  // Scheduler::Compare()
  //
  // Compares the two given indices based on their projected values.
  bool Compare(size_type lhs, size_type rhs) const {
    return projections_[lhs] < projections_[rhs];
  }

  // Scheduler::Project()
  //
  // Returns the projected value corresponding to the given index.
  value_type Project(size_type index) const { return projections_[index]; }

 protected:
  size_type data_parallel_world_size_;
  size_type global_batch_size_;
  size_type last_global_batch_size_;
  size_type last_micro_batch_size_;
  size_type micro_batch_size_;
  size_type num_microbatches_;
  std::vector<typename SymIntAdaptor::return_type> projections_;
};

}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_SCHEDULER_H_
