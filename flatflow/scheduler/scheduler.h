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
#include <cassert>
#include <cstddef>
#include <execution>
#include <functional>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/vector.h"

#include "flatflow/data/dataset.h"
#include "flatflow/data/internal/types.h"
#include "flatflow/scheduler/internal/concat.h"
#include "flatflow/scheduler/internal/extract.h"
#include "flatflow/scheduler/internal/partition.h"
#include "flatflow/scheduler/internal/reshape.h"
#include "flatflow/scheduler/internal/shuffle.h"
#include "flatflow/sklearn/linear_model/passive_aggressive.h"

namespace flatflow {

// flatflow::Scheduler<>
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
  requires(internal::Unsigned<Index> && internal::Unsigned<Size>)
class Scheduler {
 public:
  using key_type = Size;
  using mapped_type = Index;

  // Constructors and assignment operators
  //
  // In addition to the below constructor to set up scheduling,
  // a `flatflow::Scheduler<>` supports copy and move constructors
  // and assignment operators; the default constructor, on the other hand, is
  // not available since the scheduler is initialized using `std::variant` and
  // `std::monostate` to select one of several scheduling policies at runtime
  // without dynamic dispatch overhead.
  //
  // Note that unlike `flatflow::Dataset<>`, the constructors of scheduler
  // are not specified as `explicit` since an implicit conversion from scheduler
  // to `std::variant` is required.
  Scheduler(const flatbuffers::Vector<key_type> *sizes, mapped_type world_size,
            mapped_type global_batch_size, mapped_type micro_batch_size,
            mapped_type seed, bool use_flat_shuffle)
      : global_batch_size_(global_batch_size),
        micro_batch_size_(micro_batch_size),
        seed_(seed),
        world_size_(world_size),
        use_flat_shuffle_(use_flat_shuffle) {
    assert(world_size != 0);
    assert(global_batch_size != 0);
    assert(global_batch_size % world_size == 0);
    assert(micro_batch_size != 0);
    assert(global_batch_size / world_size % micro_batch_size == 0);
    assert(sizes != nullptr);

    const auto num_samples = static_cast<mapped_type>(sizes->size());
    assert(num_samples != 0);
    assert(num_samples % world_size == 0);

    LOG(INFO) << absl::StrFormat(
        "Initializing scheduler with the following arguments:\n"
        "  world_size:        %u\n"
        "  global_batch_size: %u\n"
        "  micro_batch_size:  %u\n"
        "  order:             %d\n"
        "  seed:              %u\n"
        "  heterogeneous:     %v\n"
        "  use_flat_shuffle:  %v",
        world_size, global_batch_size, micro_batch_size, 1, seed, false,
        use_flat_shuffle);

    // The last micro-batch size must be calculated since the total number of
    // data samples is guaranteed to be a multiple of data parallel size, but
    // may not be divisible by the micro-batch size.
    //
    // (x - 1) % y + 1 is always equal to x % y == 0 ? y : x % y without any
    // branch instructions.
    last_micro_batch_size_ =
        (num_samples / world_size - 1) % micro_batch_size + 1;

    // (x - 1) / y + 1 is always equal to x % y == 0 ? x / y : x / y + 1 without
    // any branch instructions.
    num_micro_batches_ =
        ((num_samples / world_size - 1) / micro_batch_size + 1) * world_size;

    // The below copy assignment is actually not copied but direct-initialized
    // by copy elision.
    dataset_ = Dataset(sizes, seed);
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
  // Note that this kind of scheduler does not take the scheduling intervals
  // into account; scheduling for models with linear complexity on identical
  // machines occurs at the granularity of epoch.
  std::vector<std::vector<mapped_type>> Schedule() {
    auto now = omp_get_wtime();

    if (micro_batch_size_ == last_micro_batch_size_) {
      const auto items =
          dataset_.template take</*Drop=*/false>(dataset_.size());
      const auto micro_batches = internal::KarmarkarKarp(
          items, num_micro_batches_, internal::forward<key_type>);

      LOG(INFO) << absl::StrFormat("Partitioning into %u micro-batches took %fs", num_micro_batches_, omp_get_wtime() - now);
      now = omp_get_wtime();

      const auto [indices, sizes] = internal::extract(internal::reshape(
          internal::shuffle(micro_batches, seed_ + epoch_, use_flat_shuffle_),
          world_size_, global_batch_size_));

      LOG(INFO) << absl::StrFormat("Epoch: %u inter-batch shuffling took %fs", epoch_, omp_get_wtime() - now);

      return indices;
    }

    const auto items = dataset_.template take</*Drop=*/false>(
        micro_batch_size_ * (num_micro_batches_ - world_size_));
    const auto micro_batches = internal::KarmarkarKarp(
        items, num_micro_batches_ - world_size_, internal::forward<key_type>);

    const auto last_items =
        dataset_.template take</*Drop=*/false>(dataset_.size());
    const auto last_micro_batches = internal::KarmarkarKarp(
        last_items, world_size_, internal::forward<key_type>);

    LOG(INFO) << absl::StrFormat("Partitioning into %u micro-batches took %fs", num_micro_batches_, omp_get_wtime() - now);
    now = omp_get_wtime();

    auto [indices, sizes] = internal::extract(internal::reshape(
        internal::shuffle(micro_batches, seed_ + epoch_, use_flat_shuffle_),
        world_size_, global_batch_size_));

    const auto [last_indices, last_sizes] = internal::extract(
        internal::reshape(internal::shuffle(last_micro_batches, seed_ + epoch_,
                                            use_flat_shuffle_),
                          world_size_, global_batch_size_));

    internal::concat(indices, last_indices);

    LOG(INFO) << absl::StrFormat("Epoch: %u inter-batch shuffling took %fs", epoch_, omp_get_wtime() - now);

    return indices;
  }

  // Scheduler::converged()
  //
  // Returns the convergence status of the underlying cost model if it exists,
  // otherwise returns true.
  bool converged() const noexcept { return true; }

  // Scheduler::on_batch_begin()
  //
  // A callback to be called at the beginning of a training batch.
  void on_batch_begin(mapped_type batch) const noexcept {
    dataset_.on_batch_begin(batch);
  }

  // Scheduler::on_batch_end()
  //
  // A callback to be called at the end of a training batch.
  void on_batch_end([[maybe_unused]] mapped_type batch,
                    [[maybe_unused]] mapped_type rank,
                    [[maybe_unused]] const flatbuffers::Vector<double> *costs)
      const noexcept {}

  // Scheduler::on_batch_end()
  //
  // A callback to be called after all rank-wise batch callbacks invocation.
  void on_batch_end(mapped_type batch) const noexcept {
    dataset_.on_batch_end(batch);
  }

  // Scheduler::on_epoch_begin()
  //
  // A callback to be called at the beginning of an epoch.
  void on_epoch_begin(mapped_type epoch) {
    dataset_.on_epoch_begin(epoch);
    epoch_ = epoch;
  }

  // Scheduler::on_epoch_end()
  //
  // A callback to be called at the end of an epoch.
  void on_epoch_end(mapped_type epoch) { dataset_.on_epoch_end(epoch); }

  // Scheduler::on_train_begin()
  //
  // A callback to be called at the beginning of training.
  void on_train_begin() const noexcept { dataset_.on_train_begin(); }

  // Scheduler::on_train_end()
  //
  // A callback to be called at the end of training.
  void on_train_end() const noexcept { dataset_.on_train_end(); }

 protected:
  mapped_type epoch_;
  mapped_type global_batch_size_;
  mapped_type last_micro_batch_size_;
  mapped_type micro_batch_size_;
  mapped_type num_micro_batches_;
  mapped_type seed_;
  mapped_type world_size_;
  bool use_flat_shuffle_;
  Dataset<mapped_type, key_type> dataset_;
};

// flatflow::Scheduler<>
//
// This is a template specialization of `flatflow::Scheduler<>` for models with
// quadratic complexity, especially for Transformer-based models such as large
// language models.
//
// Note that this scheduling policy is effective only when the model has a
// quadratic complexity in the size of each data sample. To this end,
// FlatFlow provides concatenation-based batching to avoid redundant
// computations and memory footprint due to zero padding.
template <typename Index, typename Size>
  requires(internal::Unsigned<Index> && internal::Unsigned<Size>)
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
  Scheduler(const flatbuffers::Vector<key_type> *sizes, mapped_type world_size,
            mapped_type global_batch_size, mapped_type micro_batch_size,
            mapped_type hidden_size, mapped_type seed, bool use_flat_shuffle)
      : global_batch_size_(global_batch_size),
        micro_batch_size_(micro_batch_size),
        seed_(seed),
        world_size_(world_size),
        use_flat_shuffle_(use_flat_shuffle) {
    assert(world_size != 0);
    assert(global_batch_size != 0);
    assert(global_batch_size % world_size == 0);
    assert(micro_batch_size != 0);
    assert(global_batch_size / world_size % micro_batch_size == 0);
    assert(hidden_size != 0);
    assert(sizes != nullptr);

    const auto num_samples = static_cast<mapped_type>(sizes->size());
    assert(num_samples != 0);
    assert(num_samples % world_size == 0);

    LOG(INFO) << absl::StrFormat(
        "Initializing scheduler with the following arguments:\n"
        "  world_size:        %u\n"
        "  global_batch_size: %u\n"
        "  hidden_size:       %u\n"
        "  micro_batch_size:  %u\n"
        "  order:             %d\n"
        "  seed:              %u\n"
        "  heterogeneous:     %v\n"
        "  use_flat_shuffle:  %v",
        world_size, global_batch_size, hidden_size, micro_batch_size, 2, seed,
        false, use_flat_shuffle);

    interval_ = 1;

    last_micro_batch_size_ =
        (num_samples / world_size - 1) % micro_batch_size + 1;

    sizes_ = std::vector<std::vector<std::vector<key_type>>>(world_size);
    costs_ = std::vector<std::vector<double>>(world_size);

    regressor_ = sklearn::linear_model::PassiveAggressiveRegressor</*Order=*/2>(
        hidden_size << 3);

    dataset_ = Dataset(sizes, seed);
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
  // This scheduler implementation provides profile-guided optimization;
  // it adaptively generates computation schedule based on the given feedback
  // until the underlying cost model converges. Once the model converges,
  // profile-guided optimization stops and switches to epoch-level scheduling.
  std::vector<std::vector<mapped_type>> Schedule() {
    auto now = omp_get_wtime();

    const auto op = std::bind(
        &sklearn::linear_model::PassiveAggressiveRegressor</*Order=*/2>::predict<key_type>,
        regressor_, std::placeholders::_1);

    // Once the underlying cost model converges, profile-guided optimization
    // stops and switches to epoch-level scheduling.
    if (regressor_.converged()) {
      if (micro_batch_size_ == last_micro_batch_size_) {
        const auto num_micro_batches = dataset_.size() / micro_batch_size_;
        const auto items =
            dataset_.template take</*Drop=*/false>(dataset_.size());
        const auto micro_batches =
            internal::KarmarkarKarp(items, num_micro_batches, op);

        LOG(INFO) << absl::StrFormat("Partitioning into %u micro-batches took %fs", num_micro_batches, omp_get_wtime() - now);
        now = omp_get_wtime();

        const auto [indices, sizes] = internal::extract(internal::reshape(
            internal::shuffle(micro_batches, seed_ + epoch_, use_flat_shuffle_),
            world_size_, global_batch_size_));

        LOG(INFO) << absl::StrFormat("Epoch: %u inter-batch shuffling took %fs", epoch_, omp_get_wtime() - now);

        return indices;
      }

      const auto num_micro_batches =
          dataset_.size() / world_size_ / micro_batch_size_ * world_size_ +
          world_size_;
      const auto items = dataset_.template take</*Drop=*/false>(
          micro_batch_size_ * (num_micro_batches - world_size_));
      const auto micro_batches =
          internal::KarmarkarKarp(items, num_micro_batches - world_size_, op);

      const auto last_items =
          dataset_.template take</*Drop=*/false>(dataset_.size());
      const auto last_micro_batches =
          internal::KarmarkarKarp(last_items, world_size_, op);

      LOG(INFO) << absl::StrFormat("Partitioning into %u micro-batches took %fs", num_micro_batches, omp_get_wtime() - now);
      now = omp_get_wtime();

      auto [indices, sizes] = internal::extract(internal::reshape(
          internal::shuffle(micro_batches, seed_ + epoch_, use_flat_shuffle_),
          world_size_, global_batch_size_));

      const auto [last_indices, last_sizes] =
          internal::extract(internal::reshape(
              internal::shuffle(last_micro_batches, seed_ + epoch_,
                                use_flat_shuffle_),
              world_size_, global_batch_size_));

      internal::concat(indices, last_indices);

      LOG(INFO) << absl::StrFormat("Epoch: %u inter-batch shuffling took %fs", epoch_, omp_get_wtime() - now);

      return indices;
    }

    if (micro_batch_size_ == last_micro_batch_size_) {
      const auto num_micro_batches = dataset_.size() / micro_batch_size_;
      const auto items = dataset_.template take</*Drop=*/true>(dataset_.size());
      const auto micro_batches =
          internal::KarmarkarKarp(items, num_micro_batches, op);

      LOG(INFO) << absl::StrFormat("Partitioning into %u micro-batches took %fs", num_micro_batches, omp_get_wtime() - now);
      now = omp_get_wtime();

      auto [indices, sizes] = internal::extract(internal::reshape(
          internal::shuffle(micro_batches, seed_ + epoch_, use_flat_shuffle_),
          world_size_, global_batch_size_));

      LOG(INFO) << absl::StrFormat("Epoch: %u inter-batch shuffling took %fs", epoch_, omp_get_wtime() - now);

      const auto threshold = std::min(
          global_batch_size_ / world_size_ * interval_, sizes.front().size());

      for (mapped_type rank = 0; rank < world_size_; ++rank) {
        for (std::size_t index = 0; index < threshold; ++index) {
          dataset_.template insert</*Drop=*/true>(sizes[rank][index],
                                                  indices[rank][index]);
        }

        for (std::size_t index = threshold; index < sizes.front().size();
             ++index) {
          dataset_.template insert</*Drop=*/false>(sizes[rank][index],
                                                   indices[rank][index]);
        }

        indices[rank].erase(std::next(indices[rank].begin(), threshold),
                            indices[rank].end());
        indices[rank].shrink_to_fit();

        sizes_[rank].reserve(sizes_[rank].size() +
                             threshold / micro_batch_size_);
        for (mapped_type offset = 0; offset < threshold;
             offset += micro_batch_size_) {
          sizes_[rank].emplace_back(
              std::next(sizes[rank].cbegin(), offset),
              std::next(sizes[rank].cbegin(), offset + micro_batch_size_));
        }
      }

      interval_ <<= 1;

      return indices;
    }

    const auto num_micro_batches =
        dataset_.size() / world_size_ / micro_batch_size_ * world_size_ +
        world_size_;
    const auto items = dataset_.template take</*Drop=*/true>(
        micro_batch_size_ * (num_micro_batches - world_size_));
    const auto micro_batches =
        internal::KarmarkarKarp(items, num_micro_batches - world_size_, op);

    const auto last_items =
        dataset_.template take</*Drop=*/true>(dataset_.size());
    const auto last_micro_batches =
        internal::KarmarkarKarp(last_items, world_size_, op);

    LOG(INFO) << absl::StrFormat("Partitioning into %u micro-batches took %fs", num_micro_batches, omp_get_wtime() - now);
    now = omp_get_wtime();

    auto [indices, sizes] = internal::extract(internal::reshape(
        internal::shuffle(micro_batches, seed_ + epoch_, use_flat_shuffle_),
        world_size_, global_batch_size_));

    const auto [last_indices, last_sizes] = internal::extract(
        internal::reshape(internal::shuffle(last_micro_batches, seed_ + epoch_,
                                            use_flat_shuffle_),
                          world_size_, global_batch_size_));

    internal::concat(indices, last_indices);
    internal::concat(sizes, last_sizes);

    LOG(INFO) << absl::StrFormat("Epoch: %u inter-batch shuffling took %fs", epoch_, omp_get_wtime() - now);

    const auto threshold = std::min(
        global_batch_size_ / world_size_ * interval_, sizes.front().size());

    for (mapped_type rank = 0; rank < world_size_; ++rank) {
      for (std::size_t index = 0; index < threshold; ++index) {
        dataset_.template insert</*Drop=*/true>(sizes[rank][index],
                                                indices[rank][index]);
      }

      for (std::size_t index = threshold; index < sizes.front().size();
           ++index) {
        dataset_.template insert</*Drop=*/false>(sizes[rank][index],
                                                 indices[rank][index]);
      }

      indices[rank].erase(std::next(indices[rank].begin(), threshold),
                          indices[rank].end());
      indices[rank].shrink_to_fit();

      sizes_[rank].reserve(sizes_[rank].size() +
                           (threshold - 1) / micro_batch_size_ + 1);
      for (mapped_type offset = 0; offset < threshold;
           offset += micro_batch_size_) {
        sizes_[rank].emplace_back(
            std::next(sizes[rank].cbegin(), offset),
            std::next(sizes[rank].cbegin(),
                      std::min(offset + micro_batch_size_, threshold)));
      }
    }

    interval_ <<= 1;

    return indices;
  }

  // Scheduler::converged()
  //
  // Returns the convergence status of the underlying cost model if it exists,
  // otherwise returns true.
  bool converged() const noexcept { return regressor_.converged(); }

  // Scheduler::on_batch_begin()
  //
  // A callback to be called at the beginning of a training batch.
  void on_batch_begin(mapped_type batch) const noexcept {
    dataset_.on_batch_begin(batch);
  }

  // Scheduler::on_batch_end()
  //
  // A callback to be called at the end of a training batch.
  void on_batch_end([[maybe_unused]] mapped_type batch,
                    [[maybe_unused]] mapped_type rank,
                    [[maybe_unused]] const flatbuffers::Vector<double> *costs) {
    // Store the feedback if given; it is later used to fit the underlying
    // cost model.
    if (costs == nullptr || regressor_.converged()) {
      return;
    }

    costs_[rank].reserve(costs->size());
    for (flatbuffers::uoffset_t index = 0; index < costs->size(); ++index) {
      costs_[rank].emplace_back(costs->Get(index));
    }
  }

  // Scheduler::on_batch_end()
  //
  // A callback to be called after all rank-wise batch callbacks invocation.
  void on_batch_end(mapped_type batch) {
    dataset_.on_batch_end(batch);

    if (regressor_.converged()) {
      return;
    }

    auto num_costs = static_cast<std::size_t>(0);
    std::for_each(std::execution::seq, costs_.cbegin(), costs_.cend(),
                  [&](const auto &costs) { num_costs += costs.size(); });

    auto sizes = std::vector<std::vector<key_type>>();
    sizes.reserve(num_costs);
    auto costs = std::vector<double>();
    costs.reserve(num_costs);

    for (mapped_type rank = 0; rank < world_size_; ++rank) {
      num_costs = costs_[rank].size();

      sizes.insert(
          sizes.cend(), std::move_iterator(sizes_[rank].begin()),
          std::move_iterator(std::next(sizes_[rank].begin(), num_costs)));
      costs.insert(costs.cend(), std::move_iterator(costs_[rank].begin()),
                   std::move_iterator(costs_[rank].end()));

      sizes_[rank].erase(sizes_[rank].cbegin(),
                         std::next(sizes_[rank].cbegin(), num_costs));
      sizes_[rank].shrink_to_fit();
      costs_[rank] = std::vector<double>();
    }

    regressor_.fit(sizes, costs);
  }

  // Scheduler::on_epoch_begin()
  //
  // A callback to be called at the beginning of an epoch.
  void on_epoch_begin(mapped_type epoch) {
    dataset_.on_epoch_begin(epoch);
    epoch_ = epoch;
  }

  // Scheduler::on_epoch_end()
  //
  // A callback to be called at the end of an epoch.
  void on_epoch_end(mapped_type epoch) { dataset_.on_epoch_end(epoch); }

  // Scheduler::on_train_begin()
  //
  // A callback to be called at the beginning of training.
  void on_train_begin() const noexcept { dataset_.on_train_begin(); }

  // Scheduler::on_train_end()
  //
  // A callback to be called at the end of training.
  void on_train_end() const noexcept { dataset_.on_train_end(); }

 protected:
  mapped_type epoch_;
  mapped_type global_batch_size_;
  mapped_type interval_;
  mapped_type last_micro_batch_size_;
  mapped_type micro_batch_size_;
  mapped_type seed_;
  mapped_type world_size_;
  bool use_flat_shuffle_;
  std::vector<std::vector<std::vector<key_type>>> sizes_;
  std::vector<std::vector<double>> costs_;
  sklearn::linear_model::PassiveAggressiveRegressor</*Order=*/2> regressor_;
  Dataset<mapped_type, key_type> dataset_;
};

}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_SCHEDULER_H_
