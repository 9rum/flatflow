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

#include <cstdlib>
#include <ctime>
#include <map>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "flatbuffers/flatbuffers.h"
#include "gtest/gtest.h"

#include "flatflow/data/dataset_test.h"

namespace {
class SchedulerTest final
    : private flatflow::scheduler::Scheduler<uint64_t, uint16_t> {
 public:
  inline SchedulerTest(const flatbuffers::Vector64<uint16_t> *sizes,
                       uint64_t world_size, uint64_t batch_size, uint64_t seed)
      : flatflow::scheduler::Scheduler<uint64_t, uint16_t>(sizes, world_size,
                                                           batch_size, seed) {}
  inline auto on_schedule(uint64_t interval) { return schedule(interval); }
  inline void batch_begin(uint64_t batch) { on_batch_begin(batch); }
  inline void batch_end(uint64_t batch) { on_batch_end(batch); }
  inline void epoch_end(uint64_t epoch) { on_epoch_end(epoch, 0); }
  inline void epoch_begin(uint64_t epoch) { on_epoch_begin(epoch, 0); }
  inline void train_begin() { on_train_begin(); }
  inline void train_end() { on_train_end(); }
};

TEST(SchedulerTest, StaticScheduler) {
  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  auto datasetsize = 1 << 10;
  const auto world_size = 1 << 2;
  auto batch_size = 1 << 5;
  auto seed = 0UL;

  auto items = std::map<uint16_t, std::size_t>();
  for (uint16_t size = 1; size <= datasetsize; ++size) {
    items.emplace(size, static_cast<std::size_t>(1));
  }

  auto sizes = std::vector<uint16_t>();
  for (const auto item : items) {
    const auto size = item.first;
    auto count = item.second;
    for (; 0 < count; --count) {
      sizes.push_back(size);
    }
  }
  sizes.shrink_to_fit();
  // As of FlatBuffers v24.3.7, it is not possible to initialize a 64-bit
  // vector directly; use generated code from the FlatBuffers schema.
  auto builder = flatbuffers::FlatBufferBuilder64();
  auto sizes__ = builder.CreateVector64(sizes);
  auto offset = CreateSizes(builder, sizes__);
  builder.Finish(offset);
  auto sizes_ = GetSizes(builder.GetBufferPointer());

  auto scheduler = SchedulerTest(sizes_->sizes(), world_size, batch_size, seed);
  scheduler.train_begin();
  for (auto epoch = 0; epoch < 10; ++epoch) {
    for (auto step = 0; step < datasetsize / batch_size; ++step) {
      scheduler.epoch_begin(epoch);
      const auto &indices = scheduler.on_schedule(step);

      std::vector<uint64_t> sums;
      for (auto &indice : indices) {
        auto value = 0;
        for (const auto index : indice) {
          value += sizes_->sizes()->Get(index);
        }
        sums.push_back(value);
      }

      std::string vector_str = absl::StrJoin(sums, ", ");
      LOG(INFO) << " step : " << step << "\t got : [" << vector_str << "]\n";
      scheduler.epoch_end(epoch);
    }
  }
  scheduler.train_end();
}

TEST(SchedulerTest, StaticSchedulerWithRemainder) {
  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  auto datasetsize = (1 << 10) + (1 << 3);
  const auto world_size = 1 << 2;
  auto batch_size = 1 << 5;
  auto seed = 0UL;

  auto items = std::map<uint16_t, std::size_t>();
  for (uint16_t size = 1; size <= datasetsize; ++size) {
    items.emplace(size, static_cast<std::size_t>(1));
  }

  auto sizes = std::vector<uint16_t>();
  for (const auto item : items) {
    const auto size = item.first;
    auto count = item.second;
    for (; 0 < count; --count) {
      sizes.push_back(size);
    }
  }
  sizes.shrink_to_fit();
  // As of FlatBuffers v24.3.7, it is not possible to initialize a 64-bit
  // vector directly; use generated code from the FlatBuffers schema.
  auto builder = flatbuffers::FlatBufferBuilder64();
  auto sizes__ = builder.CreateVector64(sizes);
  auto offset = CreateSizes(builder, sizes__);
  builder.Finish(offset);
  auto sizes_ = GetSizes(builder.GetBufferPointer());

  auto scheduler = SchedulerTest(sizes_->sizes(), world_size, batch_size, seed);
  scheduler.train_begin();
  for (auto epoch = 0; epoch < 10; ++epoch) {
    for (auto step = 0; step < (datasetsize / batch_size) + 1; ++step) {
      scheduler.epoch_begin(epoch);
      const auto &indices = scheduler.on_schedule(step);

      std::vector<uint64_t> sums;
      for (auto &indice : indices) {
        auto value = 0;
        for (const auto index : indice) {
          value += sizes_->sizes()->Get(index);
        }
        sums.push_back(value);
      }

      std::string vector_str = absl::StrJoin(sums, ", ");
      LOG(INFO) << " step : " << step << "\t got : [" << vector_str << "]\n";
      scheduler.epoch_end(epoch);
    }
  }
  scheduler.train_end();
}
}  // namespace
