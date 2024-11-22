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
#include <cmath>
#include <iterator>
#include <random>
#include <string>
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
#include "tests/scheduler/scheduler_test_generated.h"

namespace {

class SchedulerTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!absl::log_internal::IsInitialized()) {
      absl::InitializeLog();
      absl::SetStderrThreshold(absl::LogSeverity::kInfo);
    }

    auto distribution = std::lognormal_distribution(5.252, 0.293);
    auto generator = std::default_random_engine();

    data_.reserve(kDatasetSize);

    while (data_.size() < data_.capacity()) {
      const auto size = distribution(generator);
      if (0.5 <= size && size < 8192.5) {
        data_.emplace_back(std::lround(size));
      }
    }
  }

  void print(const std::vector<std::vector<uint64_t>> &indices, bool linear) {
    auto sums = std::vector<std::string>(kWorldSize);

    if (linear) {
      for (std::size_t step = 0; step < kNumSteps; ++step) {
        const auto begin = step * kMicroBatchSize;
        const auto end = (step + 1) * kMicroBatchSize;

        for (std::size_t rank = 0; rank < kWorldSize; ++rank) {
          auto sum = static_cast<uint16_t>(0);
          for (std::size_t index = begin; index < end; ++index) {
            sum += data_[indices[rank][index]];
          }
          sums[rank] = absl::StrFormat("%4u", sum);
        }

        LOG(INFO) << absl::StrFormat("Step: %4u got: [%s]", step, absl::StrJoin(sums, " "));
      }
    } else {
      for (std::size_t step = 0; step < kNumSteps; ++step) {
        const auto begin = step * kMicroBatchSize;
        const auto end = (step + 1) * kMicroBatchSize;

        for (std::size_t rank = 0; rank < kWorldSize; ++rank) {
          auto sum = static_cast<uint64_t>(0);
          for (std::size_t index = begin; index < end; ++index) {
            const auto size =
                static_cast<uint64_t>(data_[indices[rank][index]]);
            sum += size * (size + 8 * kHiddenSize);
          }
          sums[rank] = absl::StrFormat("%7u", sum);
        }

        LOG(INFO) << absl::StrFormat("Step: %4u got: [%s]", step, absl::StrJoin(sums, " "));
      }
    }
  }

  static constexpr auto kDatasetSize = static_cast<std::size_t>(1 << 16);
  static constexpr auto kGlobalBatchSize = static_cast<uint64_t>(1 << 8);
  static constexpr auto kHiddenSize = static_cast<uint64_t>(1 << 8);
  static constexpr auto kMicroBatchSize = static_cast<uint64_t>(1 << 2);
  static constexpr auto kNumEpochs = static_cast<uint64_t>(1 << 2);
  static constexpr auto kNumSteps = static_cast<std::size_t>(1 << 11);
  static constexpr auto kWorldSize = static_cast<uint64_t>(1 << 3);

  std::vector<uint16_t> data_;
  std::variant<std::monostate,
               flatflow::Scheduler<uint64_t, uint16_t, 1, false>,
               flatflow::Scheduler<uint64_t, uint16_t, 2, false>>
      scheduler_;
};

TEST_F(SchedulerTest, LinearModelOnIdenticalMachines) {
  auto builder = flatbuffers::FlatBufferBuilder();
  auto data = builder.CreateVector(data_);
  auto offset = CreateSizes(builder, data);
  builder.Finish(offset);

  auto sizes = GetSizes(builder.GetBufferPointer());
  scheduler_ = flatflow::Scheduler<uint64_t, uint16_t, 1, false>(
      sizes->data(), kWorldSize, kGlobalBatchSize, kMicroBatchSize, 0, true);
  auto scheduler =
      std::get<flatflow::Scheduler<uint64_t, uint16_t, 1, false>>(scheduler_);

  scheduler.on_train_begin();
  for (uint64_t epoch = 0; epoch < kNumEpochs; ++epoch) {
    scheduler.on_epoch_begin(epoch);
    scheduler.on_batch_begin(0);
    print(scheduler.Schedule(), true);
    for (uint64_t rank = 0; rank < kWorldSize; ++rank) {
      scheduler.on_batch_end(0, rank, nullptr);
    }
    scheduler.on_batch_end(0);
    scheduler.on_epoch_end(epoch);
  }
  scheduler.on_train_end();
}

TEST_F(SchedulerTest, LinearModelOnIdenticalMachinesWithoutFlatShuffle) {
  auto builder = flatbuffers::FlatBufferBuilder();
  auto data = builder.CreateVector(data_);
  auto offset = CreateSizes(builder, data);
  builder.Finish(offset);

  auto sizes = GetSizes(builder.GetBufferPointer());
  scheduler_ = flatflow::Scheduler<uint64_t, uint16_t, 1, false>(
      sizes->data(), kWorldSize, kGlobalBatchSize, kMicroBatchSize, 0, false);
  auto scheduler =
      std::get<flatflow::Scheduler<uint64_t, uint16_t, 1, false>>(scheduler_);

  scheduler.on_train_begin();
  for (uint64_t epoch = 0; epoch < kNumEpochs; ++epoch) {
    scheduler.on_epoch_begin(epoch);
    scheduler.on_batch_begin(0);
    print(scheduler.Schedule(), true);
    for (uint64_t rank = 0; rank < kWorldSize; ++rank) {
      scheduler.on_batch_end(0, rank, nullptr);
    }
    scheduler.on_batch_end(0);
    scheduler.on_epoch_end(epoch);
  }
  scheduler.on_train_end();
}

TEST_F(SchedulerTest, QuadraticModelOnIdenticalMachines) {
  auto builder = flatbuffers::FlatBufferBuilder();
  auto data = builder.CreateVector(data_);
  auto offset = CreateSizes(builder, data);
  builder.Finish(offset);

  auto sizes = GetSizes(builder.GetBufferPointer());
  scheduler_ = flatflow::Scheduler<uint64_t, uint16_t, 2, false>(
      sizes->data(), kWorldSize, kGlobalBatchSize, kMicroBatchSize, kHiddenSize,
      0, true);
  auto scheduler =
      std::get<flatflow::Scheduler<uint64_t, uint16_t, 2, false>>(scheduler_);

  auto stride = static_cast<uint64_t>(1);

  scheduler.on_train_begin();
  for (uint64_t epoch = 0; epoch < kNumEpochs; ++epoch) {
    scheduler.on_epoch_begin(epoch);
    auto indices = std::vector<std::vector<uint64_t>>(kWorldSize);
    for (std::size_t rank = 0; rank < kWorldSize; ++rank) {
      indices[rank].reserve(kNumSteps);
    }
    for (uint64_t batch = 0; batch < kDatasetSize / kGlobalBatchSize;
         batch += stride, stride <<= 1) {
      scheduler.on_batch_begin(batch);
      auto schedule = scheduler.Schedule();
      for (uint64_t rank = 0; rank < kWorldSize; ++rank) {
        indices[rank].insert(indices[rank].cend(),
                             std::move_iterator(schedule[rank].begin()),
                             std::move_iterator(schedule[rank].end()));
        scheduler.on_batch_end(batch, rank, nullptr);
      }
      scheduler.on_batch_end(batch);
    }
    print(indices, false);
    scheduler.on_epoch_end(epoch);
  }
  scheduler.on_train_end();
}

TEST_F(SchedulerTest, QuadraticModelOnIdenticalMachinesWithoutFlatShuffle) {
  auto builder = flatbuffers::FlatBufferBuilder();
  auto data = builder.CreateVector(data_);
  auto offset = CreateSizes(builder, data);
  builder.Finish(offset);

  auto sizes = GetSizes(builder.GetBufferPointer());
  scheduler_ = flatflow::Scheduler<uint64_t, uint16_t, 2, false>(
      sizes->data(), kWorldSize, kGlobalBatchSize, kMicroBatchSize, kHiddenSize,
      0, false);
  auto scheduler =
      std::get<flatflow::Scheduler<uint64_t, uint16_t, 2, false>>(scheduler_);

  auto stride = static_cast<uint64_t>(1);

  scheduler.on_train_begin();
  for (uint64_t epoch = 0; epoch < kNumEpochs; ++epoch) {
    scheduler.on_epoch_begin(epoch);
    auto indices = std::vector<std::vector<uint64_t>>(kWorldSize);
    for (std::size_t rank = 0; rank < kWorldSize; ++rank) {
      indices[rank].reserve(kNumSteps);
    }
    for (uint64_t batch = 0; batch < kDatasetSize / kGlobalBatchSize;
         batch += stride, stride <<= 1) {
      scheduler.on_batch_begin(batch);
      auto schedule = scheduler.Schedule();
      for (uint64_t rank = 0; rank < kWorldSize; ++rank) {
        indices[rank].insert(indices[rank].cend(),
                             std::move_iterator(schedule[rank].begin()),
                             std::move_iterator(schedule[rank].end()));
        scheduler.on_batch_end(batch, rank, nullptr);
      }
      scheduler.on_batch_end(batch);
    }
    print(indices, false);
    scheduler.on_epoch_end(epoch);
  }
  scheduler.on_train_end();
}

class SchedulerWithRemainderTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!absl::log_internal::IsInitialized()) {
      absl::InitializeLog();
      absl::SetStderrThreshold(absl::LogSeverity::kInfo);
    }

    auto distribution = std::lognormal_distribution(5.252, 0.293);
    auto generator = std::default_random_engine();

    data_.reserve(kDatasetSize);

    while (data_.size() < data_.capacity()) {
      const auto size = distribution(generator);
      if (0.5 <= size && size < 8192.5) {
        data_.emplace_back(std::lround(size));
      }
    }
  }

  void print(const std::vector<std::vector<uint64_t>> &indices, bool linear) {
    auto sums = std::vector<std::string>(kWorldSize);

    if (linear) {
      for (std::size_t step = 0; step < kNumSteps; ++step) {
        const auto begin = step * kMicroBatchSize;
        const auto end =
            std::min((step + 1) * kMicroBatchSize, kDatasetSize / kWorldSize);

        for (std::size_t rank = 0; rank < kWorldSize; ++rank) {
          auto sum = static_cast<uint16_t>(0);
          for (std::size_t index = begin; index < end; ++index) {
            sum += data_[indices[rank][index]];
          }
          sums[rank] = absl::StrFormat("%4u", sum);
        }

        LOG(INFO) << absl::StrFormat("Step: %4u got: [%s]", step, absl::StrJoin(sums, " "));
      }
    } else {
      for (std::size_t step = 0; step < kNumSteps; ++step) {
        const auto begin = step * kMicroBatchSize;
        const auto end =
            std::min((step + 1) * kMicroBatchSize, kDatasetSize / kWorldSize);

        for (std::size_t rank = 0; rank < kWorldSize; ++rank) {
          auto sum = static_cast<uint64_t>(0);
          for (std::size_t index = begin; index < end; ++index) {
            const auto size =
                static_cast<uint64_t>(data_[indices[rank][index]]);
            sum += size * (size + 8 * kHiddenSize);
          }
          sums[rank] = absl::StrFormat("%7u", sum);
        }

        LOG(INFO) << absl::StrFormat("Step: %4u got: [%s]", step, absl::StrJoin(sums, " "));
      }
    }
  }

  static constexpr auto kDatasetSize = static_cast<std::size_t>(1 << 16);
  static constexpr auto kGlobalBatchSize = static_cast<uint64_t>(3 << 6);
  static constexpr auto kHiddenSize = static_cast<uint64_t>(1 << 8);
  static constexpr auto kMicroBatchSize = static_cast<uint64_t>(3 << 1);
  static constexpr auto kNumEpochs = static_cast<uint64_t>(1 << 2);
  static constexpr auto kNumSteps = static_cast<std::size_t>(1366);
  static constexpr auto kWorldSize = static_cast<uint64_t>(1 << 3);

  std::vector<uint16_t> data_;
  std::variant<std::monostate,
               flatflow::Scheduler<uint64_t, uint16_t, 1, false>,
               flatflow::Scheduler<uint64_t, uint16_t, 2, false>>
      scheduler_;
};

TEST_F(SchedulerWithRemainderTest, LinearModelOnIdenticalMachines) {
  auto builder = flatbuffers::FlatBufferBuilder();
  auto data = builder.CreateVector(data_);
  auto offset = CreateSizes(builder, data);
  builder.Finish(offset);

  auto sizes = GetSizes(builder.GetBufferPointer());
  scheduler_ = flatflow::Scheduler<uint64_t, uint16_t, 1, false>(
      sizes->data(), kWorldSize, kGlobalBatchSize, kMicroBatchSize, 0, true);
  auto scheduler =
      std::get<flatflow::Scheduler<uint64_t, uint16_t, 1, false>>(scheduler_);

  scheduler.on_train_begin();
  for (uint64_t epoch = 0; epoch < kNumEpochs; ++epoch) {
    scheduler.on_epoch_begin(epoch);
    scheduler.on_batch_begin(0);
    print(scheduler.Schedule(), true);
    for (uint64_t rank = 0; rank < kWorldSize; ++rank) {
      scheduler.on_batch_end(0, rank, nullptr);
    }
    scheduler.on_batch_end(0);
    scheduler.on_epoch_end(epoch);
  }
  scheduler.on_train_end();
}

TEST_F(SchedulerWithRemainderTest, LinearModelOnIdenticalMachinesWithoutFlatShuffle) {
  auto builder = flatbuffers::FlatBufferBuilder();
  auto data = builder.CreateVector(data_);
  auto offset = CreateSizes(builder, data);
  builder.Finish(offset);

  auto sizes = GetSizes(builder.GetBufferPointer());
  scheduler_ = flatflow::Scheduler<uint64_t, uint16_t, 1, false>(
      sizes->data(), kWorldSize, kGlobalBatchSize, kMicroBatchSize, 0, false);
  auto scheduler =
      std::get<flatflow::Scheduler<uint64_t, uint16_t, 1, false>>(scheduler_);

  scheduler.on_train_begin();
  for (uint64_t epoch = 0; epoch < kNumEpochs; ++epoch) {
    scheduler.on_epoch_begin(epoch);
    scheduler.on_batch_begin(0);
    print(scheduler.Schedule(), true);
    for (uint64_t rank = 0; rank < kWorldSize; ++rank) {
      scheduler.on_batch_end(0, rank, nullptr);
    }
    scheduler.on_batch_end(0);
    scheduler.on_epoch_end(epoch);
  }
  scheduler.on_train_end();
}

TEST_F(SchedulerWithRemainderTest, QuadraticModelOnIdenticalMachines) {
  auto builder = flatbuffers::FlatBufferBuilder();
  auto data = builder.CreateVector(data_);
  auto offset = CreateSizes(builder, data);
  builder.Finish(offset);

  auto sizes = GetSizes(builder.GetBufferPointer());
  scheduler_ = flatflow::Scheduler<uint64_t, uint16_t, 2, false>(
      sizes->data(), kWorldSize, kGlobalBatchSize, kMicroBatchSize, kHiddenSize,
      0, true);
  auto scheduler =
      std::get<flatflow::Scheduler<uint64_t, uint16_t, 2, false>>(scheduler_);

  auto stride = static_cast<uint64_t>(1);

  scheduler.on_train_begin();
  for (uint64_t epoch = 0; epoch < kNumEpochs; ++epoch) {
    scheduler.on_epoch_begin(epoch);
    auto indices = std::vector<std::vector<uint64_t>>(kWorldSize);
    for (std::size_t rank = 0; rank < kWorldSize; ++rank) {
      indices[rank].reserve(kNumSteps);
    }
    for (uint64_t batch = 0; batch < kDatasetSize / kGlobalBatchSize;
         batch += stride, stride <<= 1) {
      scheduler.on_batch_begin(batch);
      auto schedule = scheduler.Schedule();
      for (uint64_t rank = 0; rank < kWorldSize; ++rank) {
        indices[rank].insert(indices[rank].cend(),
                             std::move_iterator(schedule[rank].begin()),
                             std::move_iterator(schedule[rank].end()));
        scheduler.on_batch_end(batch, rank, nullptr);
      }
      scheduler.on_batch_end(batch);
    }
    print(indices, false);
    scheduler.on_epoch_end(epoch);
  }
  scheduler.on_train_end();
}

TEST_F(SchedulerWithRemainderTest, QuadraticModelOnIdenticalMachinesWithoutFlatShuffle) {
  auto builder = flatbuffers::FlatBufferBuilder();
  auto data = builder.CreateVector(data_);
  auto offset = CreateSizes(builder, data);
  builder.Finish(offset);

  auto sizes = GetSizes(builder.GetBufferPointer());
  scheduler_ = flatflow::Scheduler<uint64_t, uint16_t, 2, false>(
      sizes->data(), kWorldSize, kGlobalBatchSize, kMicroBatchSize, kHiddenSize,
      0, false);
  auto scheduler =
      std::get<flatflow::Scheduler<uint64_t, uint16_t, 2, false>>(scheduler_);

  auto stride = static_cast<uint64_t>(1);

  scheduler.on_train_begin();
  for (uint64_t epoch = 0; epoch < kNumEpochs; ++epoch) {
    scheduler.on_epoch_begin(epoch);
    auto indices = std::vector<std::vector<uint64_t>>(kWorldSize);
    for (std::size_t rank = 0; rank < kWorldSize; ++rank) {
      indices[rank].reserve(kNumSteps);
    }
    for (uint64_t batch = 0; batch < kDatasetSize / kGlobalBatchSize;
         batch += stride, stride <<= 1) {
      scheduler.on_batch_begin(batch);
      auto schedule = scheduler.Schedule();
      for (uint64_t rank = 0; rank < kWorldSize; ++rank) {
        indices[rank].insert(indices[rank].cend(),
                             std::move_iterator(schedule[rank].begin()),
                             std::move_iterator(schedule[rank].end()));
        scheduler.on_batch_end(batch, rank, nullptr);
      }
      scheduler.on_batch_end(batch);
    }
    print(indices, false);
    scheduler.on_epoch_end(epoch);
  }
  scheduler.on_train_end();
}

}  // namespace
