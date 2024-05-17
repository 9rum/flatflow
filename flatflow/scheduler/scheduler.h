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

#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/vector.h"

#include "flatflow/data/dataset.h"
#include "flatflow/data/internal/types.h"

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
template <typename Index, typename Size, std::size_t Order, bool Heterogeneous>
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
  inline Scheduler(const flatbuffers::Vector<key_type, mapped_type> *sizes,
                   const mapped_type &world_size, const mapped_type &batch_size,
                   const mapped_type &micro_batch_size, const mapped_type &seed)
      : world_size_(world_size),
        batch_size_(batch_size),
        micro_batch_size_(micro_batch_size),
        seed_(seed) {
    CHECK_NE(world_size, 0);
    CHECK_NE(batch_size, 0);
    CHECK_EQ(batch_size % world_size, 0);
    CHECK_NE(micro_batch_size, 0);
    CHECK_EQ(batch_size / world_size % micro_batch_size, 0);
    CHECK_NE(sizes, nullptr);
    CHECK_NE(sizes->size(), 0);
    CHECK_EQ(sizes->size() % world_size, 0);

    // (x - 1) / y + 1 is always equal to x % y == 0 ? x / y : x / y + 1 without
    // any branch instructions.
    num_batches_ = (sizes->size() - 1) / batch_size + 1;
    num_micro_batches_ =
        world_size * ((sizes->size() / world_size - 1) / micro_batch_size + 1);

    // The last batch size must be calculated since the total number of data
    // samples may not be a multiple of batch size, while both are multiples of
    // world size.
    //
    // (x - 1) % y + 1 is always equal to x % y == 0 ? y : x % y without any
    // branch instructions.
    last_batch_size_ = (sizes->size() - 1) % batch_size + 1;
    last_micro_batch_size_ =
        (last_batch_size_ / world_size - 1) % micro_batch_size + 1;

    // The below copy assignment is actually not copied but direct-initialized
    // by copy elision.
    dataset_ = flatflow::data::Dataset(sizes, seed);
  }

  Scheduler() = delete;

  inline Scheduler(const Scheduler &other) = default;

  inline Scheduler &operator=(const Scheduler &other) = default;

  inline Scheduler(Scheduler &&other) = default;

  inline Scheduler &operator=(Scheduler &&other) = default;

  // Scheduler::Schedule()
  //
  // Makes schedules for the next training epoch and then shuffles them.
  //
  // Note that this scheduler discards the scheduling interval; scheduling
  // for models with linear complexity on identical machines occurs at the
  // granularity of epoch.
  inline std::vector<std::vector<mapped_type>> Schedule() {
    const auto now = omp_get_wtime();

    LOG(INFO) << absl::StrFormat("Scheduling %u batches took %fs", num_batches_, omp_get_wtime() - now);
  }

  // Scheduler::on_batch_begin()
  //
  // A callback to be called at the beginning of a training batch.
  inline void on_batch_begin(const mapped_type &batch) const noexcept {
    dataset_.on_batch_begin(batch);
  }

  // Scheduler::on_batch_end()
  //
  // A callback to be called at the end of a training batch.
  inline void on_batch_end(
      const mapped_type &batch, [[maybe_unused]] const mapped_type &rank,
      [[maybe_unused]] const flatbuffers::Vector<double, mapped_type> *profiles)
      const noexcept {
    dataset_.on_batch_end(batch);
  }

  // Scheduler::on_epoch_begin()
  //
  // A callback to be called at the beginning of an epoch.
  inline void on_epoch_begin(const mapped_type &epoch) {
    epoch_ = epoch;
    dataset_.on_epoch_begin(epoch);
  }

  // Scheduler::on_epoch_end()
  //
  // A callback to be called at the end of an epoch.
  inline void on_epoch_end(const mapped_type &epoch) {
    dataset_.on_epoch_end(epoch);
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
  mapped_type batch_size_;
  mapped_type epoch_;
  mapped_type last_batch_size_;
  mapped_type last_micro_batch_size_;
  mapped_type micro_batch_size_;
  mapped_type num_batches_;
  mapped_type num_micro_batches_;
  mapped_type seed_;
  mapped_type world_size_;
  flatflow::data::Dataset<mapped_type, key_type> dataset_;
};

}  // namespace scheduler
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_SCHEDULER_H_
