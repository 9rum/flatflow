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

#include <cassert>
#include <functional>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/vector.h"

#include "flatflow/data/dataset.h"
#include "flatflow/data/internal/types.h"
#include "flatflow/scheduler/internal/algorithm/concat.h"
#include "flatflow/scheduler/internal/algorithm/extract.h"
#include "flatflow/scheduler/internal/algorithm/partition.h"
#include "flatflow/scheduler/internal/algorithm/passive_aggressive.h"
#include "flatflow/scheduler/internal/algorithm/reshape.h"
#include "flatflow/scheduler/internal/algorithm/shuffle.h"

namespace flatflow {
namespace scheduler {

// flatflow::scheduler::Scheduler<>
//
// A common base class for all scheduler implementations.
// There are several scheduling policies on how to distribute the given data,
// and each policy has its own partial template specialization.
//
// Note that this scheduling policy is only effective for models with linear
// complexity in the size of each data sample; traditional convolutional neural
// networks (CNNs) and state space models (SSMs) in the Mamba family that
// implement linear-time sequence modeling are of this kind.
template <typename Index, typename Size, int Order, bool Heterogeneous>
  requires(flatflow::data::internal::Unsigned<Index> &&
           flatflow::data::internal::Unsigned<Size>)
class Scheduler {
 public:
  using key_type = Size;
  using mapped_type = Index;

  // Constructors and assignment operators
  //
  // In addition to the below constructor to set up scheduling,
  // a `flatflow::scheduler::Scheduler<>` supports copy and move constructors
  // and assignment operators; the default constructor, on the other hand, is
  // not available since the scheduler is initialized using `std::variant` and
  // `std::monostate` to select one of several scheduling policies at runtime
  // without dynamic dispatch overhead.
  //
  // Note that unlike `flatflow::data::Dataset<>`, the constructors of scheduler
  // are not specified as `explicit` since an implicit conversion from scheduler
  // to `std::variant` is required.
  Scheduler(const flatbuffers::Vector<key_type, mapped_type> *sizes,
            mapped_type data_parallel_size, mapped_type global_batch_size,
            mapped_type micro_batch_size, mapped_type seed,
            bool use_flat_shuffle)
      : data_parallel_size_(data_parallel_size),
        global_batch_size_(global_batch_size),
        micro_batch_size_(micro_batch_size),
        seed_(seed),
        use_flat_shuffle_(use_flat_shuffle) {
    assert(data_parallel_size != 0);
    assert(global_batch_size != 0);
    assert(global_batch_size % data_parallel_size == 0);
    assert(micro_batch_size != 0);
    assert(global_batch_size / data_parallel_size % micro_batch_size == 0);
    assert(sizes != nullptr);
    assert(sizes->size() != 0);
    assert(sizes->size() % data_parallel_size == 0);

    LOG(INFO) << absl::StrFormat(
        "Initializing scheduler with the following arguments:\n  "
        "data_parallel_size: %u\n  global_batch_size: %u\n  micro_batch_size: "
        "%u\n  order: %d\n  seed: %u\n  heterogeneous: %v\n  use_flat_shuffle: "
        "%v",
        data_parallel_size, global_batch_size, micro_batch_size, 1, seed, false,
        use_flat_shuffle);

    // (x - 1) / y + 1 is always equal to x % y == 0 ? x / y : x / y + 1 without
    // any branch instructions.
    num_micro_batches_ =
        ((sizes->size() / data_parallel_size - 1) / micro_batch_size + 1) *
        data_parallel_size;

    // The last micro-batch size must be calculated since the total number of
    // data samples is guaranteed to be a multiple of data parallel size, but
    // may not be divisible by the micro-batch size.
    //
    // (x - 1) % y + 1 is always equal to x % y == 0 ? y : x % y without any
    // branch instructions.
    last_micro_batch_size_ =
        (sizes->size() / data_parallel_size - 1) % micro_batch_size + 1;

    // The below copy assignment is actually not copied but direct-initialized
    // by copy elision.
    dataset_ = flatflow::data::Dataset(sizes, seed);
  }

  Scheduler() = delete;

  Scheduler(const Scheduler &other) = default;

  Scheduler &operator=(const Scheduler &other) = default;

  Scheduler(Scheduler &&other) = default;

  Scheduler &operator=(Scheduler &&other) = default;

  // Scheduler::Schedule()
  //
  // Generates the computation schedule for the next training epoch and then
  // shuffles it.
  //
  // Note that this scheduler does not take the scheduling interval into
  // account; scheduling for models with linear complexity on identical machines
  // occurs at the granularity of epoch.
  std::vector<std::vector<mapped_type>> Schedule() {
    auto now = omp_get_wtime();

    if (micro_batch_size_ == last_micro_batch_size_) {
      const auto items = dataset_.take(
          static_cast<std::size_t>(micro_batch_size_ * num_micro_batches_));
      const auto micro_batches = internal::algorithm::KarmarkarKarp(
          items, num_micro_batches_,
          flatflow::data::internal::OverflowSafeCast<key_type>);

      LOG(INFO) << absl::StrFormat("Partitioning into %u micro-batches took %fs", num_micro_batches_, omp_get_wtime() - now);
      now = omp_get_wtime();

      const auto [indices, sizes] =
          internal::algorithm::extract(internal::algorithm::reshape(
              internal::algorithm::shuffle(micro_batches, epoch_ + seed_,
                                           use_flat_shuffle_),
              data_parallel_size_, global_batch_size_));

      LOG(INFO) << absl::StrFormat("Epoch: %u inter-batch shuffling took %fs", epoch_, omp_get_wtime() - now);

      return indices;
    }

    const auto items = dataset_.take(static_cast<std::size_t>(
        micro_batch_size_ * (num_micro_batches_ - data_parallel_size_)));
    const auto micro_batches = internal::algorithm::KarmarkarKarp(
        items, num_micro_batches_ - data_parallel_size_,
        flatflow::data::internal::OverflowSafeCast<key_type>);

    const auto last_items = dataset_.take(
        static_cast<std::size_t>(last_micro_batch_size_ * data_parallel_size_));
    const auto last_micro_batches = internal::algorithm::KarmarkarKarp(
        last_items, data_parallel_size_,
        flatflow::data::internal::OverflowSafeCast<key_type>);

    LOG(INFO) << absl::StrFormat("Partitioning into %u micro-batches took %fs", num_micro_batches_, omp_get_wtime() - now);
    now = omp_get_wtime();

    auto [indices, sizes] =
        internal::algorithm::extract(internal::algorithm::reshape(
            internal::algorithm::shuffle(micro_batches, epoch_ + seed_,
                                         use_flat_shuffle_),
            data_parallel_size_, global_batch_size_));

    const auto [last_indices, last_sizes] =
        internal::algorithm::extract(internal::algorithm::reshape(
            internal::algorithm::shuffle(last_micro_batches, epoch_ + seed_,
                                         use_flat_shuffle_),
            data_parallel_size_, global_batch_size_));

    internal::algorithm::concat(indices, last_indices);

    LOG(INFO) << absl::StrFormat("Epoch: %u inter-batch shuffling took %fs", epoch_, omp_get_wtime() - now);

    return indices;
  }

  // Scheduler::on_batch_begin()
  //
  // A callback to be called at the beginning of a training batch.
  inline void on_batch_begin(mapped_type batch) const noexcept {
    dataset_.on_batch_begin(batch);
  }

  // Scheduler::on_batch_end()
  //
  // A callback to be called at the end of a training batch.
  inline void on_batch_end(
      mapped_type batch, [[maybe_unused]] mapped_type rank,
      [[maybe_unused]] const flatbuffers::Vector<double, mapped_type> *costs)
      const noexcept {
    dataset_.on_batch_end(batch);
  }

  // Scheduler::on_epoch_begin()
  //
  // A callback to be called at the beginning of an epoch.
  inline void on_epoch_begin(mapped_type epoch) {
    epoch_ = epoch;
    dataset_.on_epoch_begin(epoch);
  }

  // Scheduler::on_epoch_end()
  //
  // A callback to be called at the end of an epoch.
  inline void on_epoch_end(mapped_type epoch) { dataset_.on_epoch_end(epoch); }

  // Scheduler::on_train_begin()
  //
  // A callback to be called at the beginning of training.
  inline void on_train_begin() const noexcept { dataset_.on_train_begin(); }

  // Scheduler::on_train_end()
  //
  // A callback to be called at the end of training.
  inline void on_train_end() const noexcept { dataset_.on_train_end(); }

 protected:
  mapped_type data_parallel_size_;
  mapped_type epoch_;
  mapped_type global_batch_size_;
  mapped_type last_micro_batch_size_;
  mapped_type micro_batch_size_;
  mapped_type num_micro_batches_;
  mapped_type seed_;
  bool use_flat_shuffle_;
  flatflow::data::Dataset<mapped_type, key_type> dataset_;
};

// flatflow::scheduler::Scheduler<>
//
// This is a template specialization of `flatflow::scheduler::Scheduler` for
// models with quadratic complexity, especially for Transformer-based models
// such as large language models.
//
// Note that this scheduling policy is effective only when the model has a
// quadratic complexity in the size of each data sample. To this end,
// FlatFlow provides concatenation-based batching to avoid redundant
// computations and memory footprint due to zero padding.
template <typename Index, typename Size>
  requires(flatflow::data::internal::Unsigned<Index> &&
           flatflow::data::internal::Unsigned<Size>)
class Scheduler<Index, Size, /*Order=*/2, /*Heterogeneous=*/false> {
 public:
  using key_type = Size;
  using mapped_type = Index;

  // Constructors and assignment operators
  //
  // All scheduler implementations support the same overload set as the base
  // class for construction and assignment. Likewise, an implicit conversion
  // from scheduler to `std::variant` is required to avoid dynamic dispatch
  // overhead, so the constructors are not specified as `explicit`.
  //
  // One API difference is that this scheduler takes `hidden_size` as an
  // additional argument to estimate the model's complexity upon construction.
  Scheduler(const flatbuffers::Vector<key_type, mapped_type> *sizes,
            mapped_type data_parallel_size, mapped_type global_batch_size,
            mapped_type micro_batch_size, mapped_type hidden_size,
            mapped_type seed, bool use_flat_shuffle)
      : data_parallel_size_(data_parallel_size),
        global_batch_size_(global_batch_size),
        micro_batch_size_(micro_batch_size),
        seed_(seed),
        use_flat_shuffle_(use_flat_shuffle) {
    assert(data_parallel_size != 0);
    assert(global_batch_size != 0);
    assert(global_batch_size % data_parallel_size == 0);
    assert(micro_batch_size != 0);
    assert(global_batch_size / data_parallel_size % micro_batch_size == 0);
    assert(hidden_size != 0);
    assert(sizes != nullptr);
    assert(sizes->size() != 0);
    assert(sizes->size() % data_parallel_size == 0);

    LOG(INFO) << absl::StrFormat(
        "Initializing scheduler with the following arguments:\n  "
        "data_parallel_size: %u\n  global_batch_size: %u\n  hidden_size: %u\n  "
        "micro_batch_size: %u\n  order: %d\n  seed: %u\n  heterogeneous: %v\n  "
        "use_flat_shuffle: %v",
        data_parallel_size, global_batch_size, hidden_size, micro_batch_size, 2,
        seed, false, use_flat_shuffle);

    num_micro_batches_ =
        ((sizes->size() / data_parallel_size - 1) / micro_batch_size + 1) *
        data_parallel_size;

    last_micro_batch_size_ =
        (sizes->size() / data_parallel_size - 1) % micro_batch_size + 1;

    regressor_ = internal::algorithm::PassiveAggressiveRegressor</*Order=*/2>(
        hidden_size << 3);

    dataset_ = flatflow::data::Dataset(sizes, seed);
  }

  Scheduler() = delete;

  Scheduler(const Scheduler &other) = default;

  Scheduler &operator=(const Scheduler &other) = default;

  Scheduler(Scheduler &&other) = default;

  Scheduler &operator=(Scheduler &&other) = default;

  // Scheduler::Schedule()
  //
  // Generates the computation schedule for the next training epoch and then
  // shuffles it.
  //
  // Note that this naive implementation does not support profile-guided
  // optimization and may change in the future.
  std::vector<std::vector<mapped_type>> Schedule() {
    auto now = omp_get_wtime();

    const auto transform = std::bind(
        &internal::algorithm::PassiveAggressiveRegressor<2>::predict<key_type>,
        regressor_, std::placeholders::_1);

    if (micro_batch_size_ == last_micro_batch_size_) {
      const auto items = dataset_.take(
          static_cast<std::size_t>(micro_batch_size_ * num_micro_batches_));
      const auto micro_batches = internal::algorithm::KarmarkarKarp(
          items, num_micro_batches_, transform);

      LOG(INFO) << absl::StrFormat("Partitioning into %u micro-batches took %fs", num_micro_batches_, omp_get_wtime() - now);
      now = omp_get_wtime();

      const auto [indices, sizes] =
          internal::algorithm::extract(internal::algorithm::reshape(
              internal::algorithm::shuffle(micro_batches, epoch_ + seed_,
                                           use_flat_shuffle_),
              data_parallel_size_, global_batch_size_));

      LOG(INFO) << absl::StrFormat("Epoch: %u inter-batch shuffling took %fs", epoch_, omp_get_wtime() - now);

      return indices;
    }

    const auto items = dataset_.take(static_cast<std::size_t>(
        micro_batch_size_ * (num_micro_batches_ - data_parallel_size_)));
    const auto micro_batches = internal::algorithm::KarmarkarKarp(
        items, num_micro_batches_ - data_parallel_size_, transform);

    const auto last_items = dataset_.take(
        static_cast<std::size_t>(last_micro_batch_size_ * data_parallel_size_));
    const auto last_micro_batches = internal::algorithm::KarmarkarKarp(
        last_items, data_parallel_size_, transform);

    LOG(INFO) << absl::StrFormat("Partitioning into %u micro-batches took %fs", num_micro_batches_, omp_get_wtime() - now);
    now = omp_get_wtime();

    auto [indices, sizes] =
        internal::algorithm::extract(internal::algorithm::reshape(
            internal::algorithm::shuffle(micro_batches, epoch_ + seed_,
                                         use_flat_shuffle_),
            data_parallel_size_, global_batch_size_));

    const auto [last_indices, last_sizes] =
        internal::algorithm::extract(internal::algorithm::reshape(
            internal::algorithm::shuffle(last_micro_batches, epoch_ + seed_,
                                         use_flat_shuffle_),
            data_parallel_size_, global_batch_size_));

    internal::algorithm::concat(indices, last_indices);

    LOG(INFO) << absl::StrFormat("Epoch: %u inter-batch shuffling took %fs", epoch_, omp_get_wtime() - now);

    return indices;
  }

  // Scheduler::on_batch_begin()
  //
  // A callback to be called at the beginning of a training batch.
  inline void on_batch_begin(mapped_type batch) const noexcept {
    dataset_.on_batch_begin(batch);
  }

  // Scheduler::on_batch_end()
  //
  // A callback to be called at the end of a training batch.
  inline void on_batch_end(
      mapped_type batch, [[maybe_unused]] mapped_type rank,
      [[maybe_unused]] const flatbuffers::Vector<double, mapped_type> *costs)
      const noexcept {
    dataset_.on_batch_end(batch);
  }

  // Scheduler::on_epoch_begin()
  //
  // A callback to be called at the beginning of an epoch.
  inline void on_epoch_begin(mapped_type epoch) {
    epoch_ = epoch;
    dataset_.on_epoch_begin(epoch);
  }

  // Scheduler::on_epoch_end()
  //
  // A callback to be called at the end of an epoch.
  inline void on_epoch_end(mapped_type epoch) { dataset_.on_epoch_end(epoch); }

  // Scheduler::on_train_begin()
  //
  // A callback to be called at the beginning of training.
  inline void on_train_begin() const noexcept { dataset_.on_train_begin(); }

  // Scheduler::on_train_end()
  //
  // A callback to be called at the end of training.
  inline void on_train_end() const noexcept { dataset_.on_train_end(); }

 protected:
  mapped_type data_parallel_size_;
  mapped_type epoch_;
  mapped_type global_batch_size_;
  mapped_type last_micro_batch_size_;
  mapped_type micro_batch_size_;
  mapped_type num_micro_batches_;
  mapped_type seed_;
  bool use_flat_shuffle_;
  std::vector<std::vector<key_type>> sizes_;
  std::vector<double> costs_;
  internal::algorithm::PassiveAggressiveRegressor</*Order=*/2> regressor_;
  flatflow::data::Dataset<mapped_type, key_type> dataset_;
};

}  // namespace scheduler
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_SCHEDULER_H_
