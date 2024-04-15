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

#include "flatflow/scheduler/scheduler.h"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <variant>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/internal/globals.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "flatbuffers/flatbuffers.h"
#include "gtest/gtest.h"

#include "flatflow/scheduler/scheduler_test.h"

namespace {

class SchedulerTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!absl::log_internal::IsInitialized()) {
      absl::InitializeLog();
      absl::SetStderrThreshold(absl::LogSeverity::kInfo);
    }

    auto data = std::vector<uint16_t>(kDatasetSize);
    std::iota(data.begin(), data.end(), 1);

    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    auto generator = std::ranlux48();
    generator.seed(static_cast<uint_fast64_t>(std::rand()));
    std::shuffle(data.begin(), data.end(), generator);

    auto builder = flatbuffers::FlatBufferBuilder64();
    auto data__ = builder.CreateVector64(data);
    auto offset = CreateSizes(builder, data__);
    builder.Finish(offset);

    auto sizes = GetSizes(builder.GetBufferPointer());
    sizes_ = sizes->data();
  }

  void print(const std::vector<std::vector<uint64_t>> &indices) {
    for (uint64_t step = 0; step < (kDatasetSize - 1) / kBatchSize + 1;
         ++step) {
      auto sums = std::vector<uint16_t>(kWorldSize, 0);
      for (uint64_t rank = 0; rank < kWorldSize; ++rank) {
        std::for_each(
            std::next(indices.at(rank).cbegin(),
                      step * kBatchSize / kWorldSize),
            std::next(indices.at(rank).cbegin(),
                      (step + 1) * kBatchSize / kWorldSize),
            [&](const auto &index) { sums.at(rank) += sizes_->Get(index); });
      }
      LOG(INFO) << absl::StrFormat("Step: %u got: [%s]", step,
                                   absl::StrJoin(sums, " "));
    }
  }

  static constexpr auto kDatasetSize = static_cast<std::size_t>(1 << 10);
  static constexpr auto kWorldSize = static_cast<uint64_t>(1 << 2);
  static constexpr auto kBatchSize = static_cast<uint64_t>(1 << 5);
  static constexpr auto kMaxEpoch = static_cast<uint64_t>(1 << 3);
  const flatbuffers::Vector64<uint16_t> *sizes_;
  std::variant<std::monostate,
               flatflow::scheduler::Scheduler<
                   uint64_t, uint16_t, flatflow::scheduler::Schedule::kStatic>,
               flatflow::scheduler::Scheduler<
                   uint64_t, uint16_t, flatflow::scheduler::Schedule::kDynamic>,
               flatflow::scheduler::Scheduler<
                   uint64_t, uint16_t, flatflow::scheduler::Schedule::kGuided>>
      scheduler_;
};

TEST_F(SchedulerTest, StaticScheduler) {
  scheduler_ =
      flatflow::scheduler::Scheduler<uint64_t, uint16_t,
                                     flatflow::scheduler::Schedule::kStatic>(
          sizes_, kWorldSize, kBatchSize, 0);

  std::get<flatflow::scheduler::Scheduler<
      uint64_t, uint16_t, flatflow::scheduler::Schedule::kStatic>>(scheduler_)
      .on_train_begin();

  for (uint64_t epoch = 0; epoch < kMaxEpoch; ++epoch) {
    LOG(INFO) << absl::StrFormat("Epoch: %u", epoch);
    for (uint64_t rank = 0; rank < kWorldSize; ++rank) {
      std::get<flatflow::scheduler::Scheduler<
          uint64_t, uint16_t, flatflow::scheduler::Schedule::kStatic>>(
          scheduler_)
          .on_epoch_begin(epoch, rank);
    }

    print(std::get<flatflow::scheduler::Scheduler<
              uint64_t, uint16_t, flatflow::scheduler::Schedule::kStatic>>(
              scheduler_)
              .schedule((kDatasetSize - 1) / kBatchSize + 1));

    for (uint64_t rank = 0; rank < kWorldSize; ++rank) {
      std::get<flatflow::scheduler::Scheduler<
          uint64_t, uint16_t, flatflow::scheduler::Schedule::kStatic>>(
          scheduler_)
          .on_epoch_end(epoch, rank);
    }
  }

  std::get<flatflow::scheduler::Scheduler<
      uint64_t, uint16_t, flatflow::scheduler::Schedule::kStatic>>(scheduler_)
      .on_train_end();
}

class SchedulerWithRemainderTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!absl::log_internal::IsInitialized()) {
      absl::InitializeLog();
      absl::SetStderrThreshold(absl::LogSeverity::kInfo);
    }

    auto data = std::vector<uint16_t>(kDatasetSize);
    std::iota(data.begin(), data.end(), 1);

    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    auto generator = std::ranlux48();
    generator.seed(static_cast<uint_fast64_t>(std::rand()));
    std::shuffle(data.begin(), data.end(), generator);

    auto builder = flatbuffers::FlatBufferBuilder64();
    auto data__ = builder.CreateVector64(data);
    auto offset = CreateSizes(builder, data__);
    builder.Finish(offset);

    auto sizes = GetSizes(builder.GetBufferPointer());
    sizes_ = sizes->data();
  }

  void print(const std::vector<std::vector<uint64_t>> &indices) {
    for (uint64_t step = 0; step < (kDatasetSize - 1) / kBatchSize + 1;
         ++step) {
      auto sums = std::vector<uint16_t>(kWorldSize, 0);
      for (uint64_t rank = 0; rank < kWorldSize; ++rank) {
        std::for_each(
            std::next(indices.at(rank).cbegin(),
                      step * kBatchSize / kWorldSize),
            std::next(indices.at(rank).cbegin(),
                      (step + 1) * kBatchSize / kWorldSize),
            [&](const auto &index) { sums.at(rank) += sizes_->Get(index); });
      }
      LOG(INFO) << absl::StrFormat("Step: %u got: [%s]", step,
                                   absl::StrJoin(sums, " "));
    }
  }

  static constexpr auto kDatasetSize =
      static_cast<std::size_t>((1 << 10) + (1 << 3));
  static constexpr auto kWorldSize = static_cast<uint64_t>(1 << 2);
  static constexpr auto kBatchSize = static_cast<uint64_t>(1 << 5);
  static constexpr auto kMaxEpoch = static_cast<uint64_t>(1 << 3);
  const flatbuffers::Vector64<uint16_t> *sizes_;
  std::variant<std::monostate,
               flatflow::scheduler::Scheduler<
                   uint64_t, uint16_t, flatflow::scheduler::Schedule::kStatic>,
               flatflow::scheduler::Scheduler<
                   uint64_t, uint16_t, flatflow::scheduler::Schedule::kDynamic>,
               flatflow::scheduler::Scheduler<
                   uint64_t, uint16_t, flatflow::scheduler::Schedule::kGuided>>
      scheduler_;
};

TEST_F(SchedulerWithRemainderTest, StaticScheduler) {
  scheduler_ =
      flatflow::scheduler::Scheduler<uint64_t, uint16_t,
                                     flatflow::scheduler::Schedule::kStatic>(
          sizes_, kWorldSize, kBatchSize, 0);

  std::get<flatflow::scheduler::Scheduler<
      uint64_t, uint16_t, flatflow::scheduler::Schedule::kStatic>>(scheduler_)
      .on_train_begin();

  for (uint64_t epoch = 0; epoch < kMaxEpoch; ++epoch) {
    LOG(INFO) << absl::StrFormat("Epoch: %u", epoch);
    for (uint64_t rank = 0; rank < kWorldSize; ++rank) {
      std::get<flatflow::scheduler::Scheduler<
          uint64_t, uint16_t, flatflow::scheduler::Schedule::kStatic>>(
          scheduler_)
          .on_epoch_begin(epoch, rank);
    }

    print(std::get<flatflow::scheduler::Scheduler<
              uint64_t, uint16_t, flatflow::scheduler::Schedule::kStatic>>(
              scheduler_)
              .schedule((kDatasetSize - 1) / kBatchSize + 1));

    for (uint64_t rank = 0; rank < kWorldSize; ++rank) {
      std::get<flatflow::scheduler::Scheduler<
          uint64_t, uint16_t, flatflow::scheduler::Schedule::kStatic>>(
          scheduler_)
          .on_epoch_end(epoch, rank);
    }
  }

  std::get<flatflow::scheduler::Scheduler<
      uint64_t, uint16_t, flatflow::scheduler::Schedule::kStatic>>(scheduler_)
      .on_train_end();
}

}  // namespace
