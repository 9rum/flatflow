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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <random>
#include <set>
#include <string>
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

#include "flatflow/ops/graph_generated.h"
#include "flatflow/ops/node_generated.h"
#include "flatflow/ops/operator_generated.h"

namespace {

// A read-only scheduler wrapper used only for testing purpose.
class SchedChecker : public flatflow::Scheduler {
 public:
  using Base = typename SchedChecker::Scheduler;

  using Base::Base;

  // SchedChecker::Check()
  //
  // This check verifies the implementation details of partial reordering
  // with the following questions:
  //
  // * Does the reordered computation plan have the same composition as the
  //   original computation plan?
  // * Does each batch of the reordered computation plan have the same
  //   composition as that of the original computation plan? That is, does
  //   partial reordering maintain the observable behavior of the model?
  void Check(const std::vector<size_t> &schedule) const {
    const auto total_size = schedule.size();
    auto indices = std::vector<size_t>(total_size);

    const auto result =
        Schedule(schedule.begin(), schedule.end(), indices.begin());
    EXPECT_EQ(std::distance(result, indices.end()), 0);

    EXPECT_EQ(std::set<size_t>(indices.begin(), indices.end()),
              std::set<size_t>(schedule.begin(), schedule.end()));

    constexpr auto kZero = static_cast<value_type>(0);
    auto buf = std::vector<std::string>(data_parallel_world_size_);

    for (size_t offset = 0; offset < total_size; offset += global_batch_size_) {
      const auto num_samples = offset + global_batch_size_ < total_size
                                   ? global_batch_size_
                                   : last_global_batch_size_;

      EXPECT_EQ(
          std::set<size_t>(std::next(indices.begin(), offset),
                           std::next(indices.begin(), offset + num_samples)),
          std::set<size_t>(std::next(schedule.begin(), offset),
                           std::next(schedule.begin(), offset + num_samples)));

      const auto num_microbatches_per_replica =
          (num_samples / data_parallel_world_size_ - 1) / micro_batch_size_;
      const auto last_micro_batch_size = num_samples == global_batch_size_
                                             ? micro_batch_size_
                                             : last_micro_batch_size_;

      for (size_t step = 0; step < num_microbatches_per_replica; ++step) {
        const auto base = offset + micro_batch_size_ * step;

        for (size_t rank = 0; rank < data_parallel_world_size_; ++rank) {
          const auto first =
              base + num_samples / data_parallel_world_size_ * rank;
          buf[rank] = absl::StrFormat(
              "%11d",
              std::transform_reduce(
                  std::next(indices.begin(), first),
                  std::next(indices.begin(), first + micro_batch_size_), kZero,
                  std::plus<>(), [&](size_t index) { return preds_[index]; }));
        }

        LOG(INFO) << absl::StrFormat("[%s]", absl::StrJoin(buf, " "));
      }

      for (size_t rank = 0; rank < data_parallel_world_size_; ++rank) {
        const auto last =
            offset + num_samples / data_parallel_world_size_ * (rank + 1);
        buf[rank] = absl::StrFormat(
            "%11d",
            std::transform_reduce(
                std::next(indices.begin(), last - last_micro_batch_size),
                std::next(indices.begin(), last), kZero, std::plus<>(),
                [&](size_t index) { return preds_[index]; }));
      }

      // clang-format off
      LOG(INFO) << absl::StrFormat("[%s]", absl::StrJoin(buf, " "));
      LOG(INFO) << "-------------------------------------------------------------------------------------------------";
      // clang-format on
    }
  }
};

flatflow::SymInt CreateSymInt(int64_t x, int64_t y) {
  return flatflow::SymInt(flatbuffers::make_span({x, y}));
}

template <typename... Args>
std::vector<flatflow::SymInt> CreateVectorOfSymInts(Args... args) {
  return std::vector<flatflow::SymInt>{args...};
}

class SchedulerTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!absl::log_internal::IsInitialized()) {
      absl::InitializeLog();
      absl::SetStderrThreshold(absl::LogSeverity::kInfo);
    }

    auto distribution = std::lognormal_distribution(5.252, 0.293);
    auto generator = std::default_random_engine();

    sizes_.reserve(kTotalSize);

    while (sizes_.size() < sizes_.capacity()) {
      const auto size = distribution(generator);
      if (0.5 <= size && size < 8192.5) {
        sizes_.emplace_back(std::lround(size));
      }
    }
  }

  static constexpr auto kDataParallelWorldSize = static_cast<size_t>(1 << 3);
  static constexpr auto kGlobalBatchSize = static_cast<size_t>(1 << 9);
  static constexpr auto kMicroBatchSize = static_cast<size_t>(1 << 2);
  static constexpr auto kNumEpochs = static_cast<size_t>(1 << 1);
  static constexpr auto kTotalSize = static_cast<size_t>(1 << 15);
  std::vector<uint32_t> sizes_;
};

TEST_F(SchedulerTest, Llama3) {
  auto builder = flatbuffers::FlatBufferBuilder();

  auto target = flatflow::Operator::EMBEDDING;
  auto sym_int0 = CreateSymInt(128256, 0);
  auto sym_int1 = CreateSymInt(4096, 0);
  auto shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  auto arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  auto arg1 = flatflow::CreateTensorMetadata(builder, shape);
  auto args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  auto sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  auto meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node0 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ARANGE_START;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node1 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node2 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::FULL;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node3 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRIU;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node4 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ARANGE;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node5 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node6 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::GT_TENSOR;
  sym_int0 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node7 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node8 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node9 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node10 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node11 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node12 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node13 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node14 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node15 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node16 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node17 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node18 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node19 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node20 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node21 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node22 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CAT;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node23 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::COS;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node24 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SIN;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node25 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node26 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::POW_TENSOR_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node27 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MEAN_DIM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node28 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node29 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::RSQRT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node30 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node31 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node32 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node33 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node34 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node35 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node36 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  auto sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node37 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(1024, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node38 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node39 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1024, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node40 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1024, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node41 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node42 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node43 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node44 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node45 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::NEG;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node46 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node47 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node48 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node49 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::NEG;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node50 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node51 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  auto sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node52 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node53 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node54 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node55 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_UNSAFE_VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node56 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node57 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node58 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node59 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node60 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node61 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node62 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node63 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node64 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node65 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node66 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node67 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node68 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node69 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node70 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node71 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_SOFTMAX;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node72 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node73 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node74 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node75 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node76 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node77 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node78 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node79 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(14336, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node80 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node81 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node82 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SILU;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node83 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node84 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(14336, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node85 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node86 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(14336, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node87 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node88 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(128256, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(128256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node89 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(128256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(128256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node90 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(128256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node91 = flatflow::CreateNode(builder, target, args, meta);

  auto nodes = builder.CreateVector(
      {node0,  node1,  node2,  node3,  node4,  node5,  node6,  node7,  node8,
       node9,  node10, node11, node12, node13, node14, node15, node16, node17,
       node18, node19, node20, node21, node22, node23, node24, node25, node26,
       node27, node28, node29, node30, node31, node32, node33, node34, node35,
       node36, node37, node38, node39, node40, node41, node42, node43, node44,
       node45, node46, node47, node48, node49, node50, node51, node52, node53,
       node54, node55, node56, node57, node58, node59, node60, node61, node62,
       node63, node64, node65, node66, node67, node68, node69, node70, node71,
       node72, node73, node74, node75, node76, node77, node78, node79, node80,
       node81, node82, node83, node84, node85, node86, node87, node88, node89,
       node90, node91});
  auto root = flatflow::CreateGraph(builder, nodes);
  builder.Finish(root);

  auto graph =
      flatbuffers::GetRoot<flatflow::Graph>(builder.GetBufferPointer());

  const auto checker =
      SchedChecker(kDataParallelWorldSize, kGlobalBatchSize, kMicroBatchSize,
                   sizes_.begin(), sizes_.end(), graph);

  checker.on_train_begin();
  for (size_t epoch = 0; epoch < kNumEpochs; ++epoch) {
    checker.on_epoch_begin(epoch);

    auto schedule = std::vector<size_t>(kTotalSize);
    std::iota(schedule.begin(), schedule.end(), 0);

    auto generator = std::mt19937();
    generator.seed(epoch);
    std::shuffle(schedule.begin(), schedule.end(), generator);

    checker.Check(schedule);

    checker.on_epoch_end(epoch);
  }
  checker.on_train_end();
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

    sizes_.reserve(kTotalSize);

    while (sizes_.size() < sizes_.capacity()) {
      const auto size = distribution(generator);
      if (0.5 <= size && size < 8192.5) {
        sizes_.emplace_back(std::lround(size));
      }
    }
  }

  static constexpr auto kDataParallelWorldSize = static_cast<size_t>(1 << 3);
  static constexpr auto kGlobalBatchSize = static_cast<size_t>(3 << 8);
  static constexpr auto kMicroBatchSize = static_cast<size_t>(3 << 1);
  static constexpr auto kNumEpochs = static_cast<size_t>(1 << 1);
  static constexpr auto kTotalSize = static_cast<size_t>(1 << 15);
  std::vector<uint32_t> sizes_;
};

TEST_F(SchedulerWithRemainderTest, Llama3) {
  auto builder = flatbuffers::FlatBufferBuilder();

  auto target = flatflow::Operator::EMBEDDING;
  auto sym_int0 = CreateSymInt(128256, 0);
  auto sym_int1 = CreateSymInt(4096, 0);
  auto shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  auto arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  auto arg1 = flatflow::CreateTensorMetadata(builder, shape);
  auto args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  auto sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  auto meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node0 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ARANGE_START;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node1 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node2 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::FULL;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node3 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRIU;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node4 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ARANGE;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node5 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node6 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::GT_TENSOR;
  sym_int0 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node7 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node8 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node9 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node10 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node11 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node12 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node13 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node14 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node15 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node16 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node17 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node18 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node19 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node20 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node21 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node22 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CAT;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node23 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::COS;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node24 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SIN;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node25 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node26 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::POW_TENSOR_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node27 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MEAN_DIM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node28 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node29 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::RSQRT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node30 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node31 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node32 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node33 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node34 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node35 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node36 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  auto sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node37 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(1024, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node38 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node39 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1024, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node40 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1024, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node41 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node42 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node43 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node44 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node45 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::NEG;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node46 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node47 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node48 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node49 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::NEG;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node50 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node51 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  auto sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node52 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node53 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node54 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node55 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_UNSAFE_VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node56 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node57 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node58 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node59 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node60 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node61 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node62 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node63 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node64 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node65 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node66 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node67 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node68 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node69 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node70 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node71 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_SOFTMAX;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node72 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node73 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node74 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node75 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node76 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node77 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node78 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node79 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(14336, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node80 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node81 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node82 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SILU;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node83 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node84 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(14336, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node85 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node86 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(14336, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node87 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node88 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(128256, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(128256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node89 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(128256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(128256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node90 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(128256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node91 = flatflow::CreateNode(builder, target, args, meta);

  auto nodes = builder.CreateVector(
      {node0,  node1,  node2,  node3,  node4,  node5,  node6,  node7,  node8,
       node9,  node10, node11, node12, node13, node14, node15, node16, node17,
       node18, node19, node20, node21, node22, node23, node24, node25, node26,
       node27, node28, node29, node30, node31, node32, node33, node34, node35,
       node36, node37, node38, node39, node40, node41, node42, node43, node44,
       node45, node46, node47, node48, node49, node50, node51, node52, node53,
       node54, node55, node56, node57, node58, node59, node60, node61, node62,
       node63, node64, node65, node66, node67, node68, node69, node70, node71,
       node72, node73, node74, node75, node76, node77, node78, node79, node80,
       node81, node82, node83, node84, node85, node86, node87, node88, node89,
       node90, node91});
  auto root = flatflow::CreateGraph(builder, nodes);
  builder.Finish(root);

  auto graph =
      flatbuffers::GetRoot<flatflow::Graph>(builder.GetBufferPointer());

  const auto checker =
      SchedChecker(kDataParallelWorldSize, kGlobalBatchSize, kMicroBatchSize,
                   sizes_.begin(), sizes_.end(), graph);

  checker.on_train_begin();
  for (size_t epoch = 0; epoch < kNumEpochs; ++epoch) {
    checker.on_epoch_begin(epoch);

    auto schedule = std::vector<size_t>(kTotalSize);
    std::iota(schedule.begin(), schedule.end(), 0);

    auto generator = std::mt19937();
    generator.seed(epoch);
    std::shuffle(schedule.begin(), schedule.end(), generator);

    checker.Check(schedule);

    checker.on_epoch_end(epoch);
  }
  checker.on_train_end();
}

class SchedulerWithRemainderOnlyTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!absl::log_internal::IsInitialized()) {
      absl::InitializeLog();
      absl::SetStderrThreshold(absl::LogSeverity::kInfo);
    }

    auto distribution = std::lognormal_distribution(5.252, 0.293);
    auto generator = std::default_random_engine();

    sizes_.reserve(kTotalSize);

    while (sizes_.size() < sizes_.capacity()) {
      const auto size = distribution(generator);
      if (0.5 <= size && size < 8192.5) {
        sizes_.emplace_back(std::lround(size));
      }
    }
  }

  static constexpr auto kDataParallelWorldSize = static_cast<size_t>(1 << 3);
  static constexpr auto kGlobalBatchSize = static_cast<size_t>(1 << 9);
  static constexpr auto kMicroBatchSize = static_cast<size_t>(1 << 2);
  static constexpr auto kNumEpochs = static_cast<size_t>(1 << 1);
  static constexpr auto kTotalSize = static_cast<size_t>(4035 << 3);
  std::vector<uint32_t> sizes_;
};

TEST_F(SchedulerWithRemainderOnlyTest, Llama3) {
  auto builder = flatbuffers::FlatBufferBuilder();

  auto target = flatflow::Operator::EMBEDDING;
  auto sym_int0 = CreateSymInt(128256, 0);
  auto sym_int1 = CreateSymInt(4096, 0);
  auto shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  auto arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  auto arg1 = flatflow::CreateTensorMetadata(builder, shape);
  auto args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  auto sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  auto meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node0 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ARANGE_START;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node1 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node2 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::FULL;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node3 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRIU;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node4 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ARANGE;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node5 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node6 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::GT_TENSOR;
  sym_int0 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node7 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node8 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node9 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node10 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node11 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node12 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node13 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node14 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node15 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node16 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node17 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node18 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node19 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node20 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node21 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node22 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CAT;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node23 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::COS;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node24 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SIN;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node25 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node26 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::POW_TENSOR_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node27 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MEAN_DIM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node28 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node29 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::RSQRT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node30 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node31 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node32 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node33 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node34 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node35 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node36 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  auto sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node37 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(1024, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node38 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node39 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1024, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node40 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1024, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node41 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node42 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node43 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node44 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node45 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::NEG;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node46 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node47 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node48 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node49 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::NEG;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node50 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node51 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  auto sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node52 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node53 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node54 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node55 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_UNSAFE_VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node56 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node57 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node58 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node59 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node60 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node61 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node62 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node63 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node64 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node65 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node66 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node67 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node68 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node69 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node70 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node71 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_SOFTMAX;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node72 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node73 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node74 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node75 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node76 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node77 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node78 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node79 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(14336, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node80 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node81 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node82 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SILU;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node83 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node84 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(14336, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node85 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(14336, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node86 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(14336, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(14336, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node87 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node88 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(128256, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(128256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node89 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(128256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(128256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node90 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(128256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node91 = flatflow::CreateNode(builder, target, args, meta);

  auto nodes = builder.CreateVector(
      {node0,  node1,  node2,  node3,  node4,  node5,  node6,  node7,  node8,
       node9,  node10, node11, node12, node13, node14, node15, node16, node17,
       node18, node19, node20, node21, node22, node23, node24, node25, node26,
       node27, node28, node29, node30, node31, node32, node33, node34, node35,
       node36, node37, node38, node39, node40, node41, node42, node43, node44,
       node45, node46, node47, node48, node49, node50, node51, node52, node53,
       node54, node55, node56, node57, node58, node59, node60, node61, node62,
       node63, node64, node65, node66, node67, node68, node69, node70, node71,
       node72, node73, node74, node75, node76, node77, node78, node79, node80,
       node81, node82, node83, node84, node85, node86, node87, node88, node89,
       node90, node91});
  auto root = flatflow::CreateGraph(builder, nodes);
  builder.Finish(root);

  auto graph =
      flatbuffers::GetRoot<flatflow::Graph>(builder.GetBufferPointer());

  const auto checker =
      SchedChecker(kDataParallelWorldSize, kGlobalBatchSize, kMicroBatchSize,
                   sizes_.begin(), sizes_.end(), graph);

  checker.on_train_begin();
  for (size_t epoch = 0; epoch < kNumEpochs; ++epoch) {
    checker.on_epoch_begin(epoch);

    auto schedule = std::vector<size_t>(kTotalSize);
    std::iota(schedule.begin(), schedule.end(), 0);

    auto generator = std::mt19937();
    generator.seed(epoch);
    std::shuffle(schedule.begin(), schedule.end(), generator);

    checker.Check(schedule);

    checker.on_epoch_end(epoch);
  }
  checker.on_train_end();
}

}  // namespace
