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

TEST_F(SchedulerTest, Llama31) {
  auto builder = flatbuffers::FlatBufferBuilder();

  auto op = flatflow::Operator::EMBEDDING;
  auto sym_int0 = CreateSymInt(128256, 0);
  auto sym_int1 = CreateSymInt(8192, 0);
  auto shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  auto meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  auto meta1 = flatflow::CreateTensorMetadata(builder, shape);
  auto args = builder.CreateVector({meta0, meta1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  auto sym_int2 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  auto meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node0 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SYM_SIZE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  shape = builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>());
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node1 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::ARANGE_START;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(-7, 8);
  shape =
      builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>({sym_int0}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node2 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(-7, 8);
  shape =
      builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>({sym_int0}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node3 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::FULL;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node4 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::TRIU;
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node5 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::ARANGE;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(-6, 8);
  shape =
      builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>({sym_int0}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node6 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(-7, 8);
  shape =
      builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>({sym_int0}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node7 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::GT_TENSOR;
  sym_int0 = CreateSymInt(-6, 8);
  shape =
      builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>({sym_int0}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node8 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node9 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node10 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::_TO_COPY;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node11 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node12 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node13 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node14 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::CAT;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node15 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::COS;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node16 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SIN;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node17 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::POW_TENSOR_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node18 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MEAN_DIM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node19 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node20 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::RSQRT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node21 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::T;
  sym_int0 = CreateSymInt(8192, 0);
  sym_int1 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(8192, 0);
  sym_int1 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node22 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(8192, 0);
  sym_int1 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node23 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::NEG;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  auto sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node24 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(-7, 8);
  auto sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>(
      {sym_int0, sym_int1, sym_int2, sym_int3, sym_int4}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(-7, 8);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>(
      {sym_int0, sym_int1, sym_int2, sym_int3, sym_int4}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node25 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::_UNSAFE_VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(-7, 8);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>(
      {sym_int0, sym_int1, sym_int2, sym_int3, sym_int4}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node26 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MUL_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node27 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::_SOFTMAX;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node28 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SILU;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(28672, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(28672, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node29 = flatflow::CreateNode(builder, op, args, meta);

  auto nodes = builder.CreateVector(
      {node0,  node1,  node2,  node3,  node4,  node5,  node6,  node7,
       node8,  node9,  node10, node11, node12, node13, node14, node15,
       node16, node17, node18, node19, node20, node21, node22, node23,
       node24, node25, node26, node27, node28, node29});
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

TEST_F(SchedulerWithRemainderTest, Llama31) {
  auto builder = flatbuffers::FlatBufferBuilder();

  auto op = flatflow::Operator::EMBEDDING;
  auto sym_int0 = CreateSymInt(128256, 0);
  auto sym_int1 = CreateSymInt(8192, 0);
  auto shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  auto meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  auto meta1 = flatflow::CreateTensorMetadata(builder, shape);
  auto args = builder.CreateVector({meta0, meta1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  auto sym_int2 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  auto meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node0 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SYM_SIZE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  shape = builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>());
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node1 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::ARANGE_START;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(-7, 8);
  shape =
      builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>({sym_int0}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node2 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(-7, 8);
  shape =
      builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>({sym_int0}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node3 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::FULL;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node4 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::TRIU;
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node5 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::ARANGE;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(-6, 8);
  shape =
      builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>({sym_int0}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node6 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(-7, 8);
  shape =
      builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>({sym_int0}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node7 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::GT_TENSOR;
  sym_int0 = CreateSymInt(-6, 8);
  shape =
      builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>({sym_int0}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node8 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node9 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node10 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::_TO_COPY;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node11 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node12 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node13 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node14 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::CAT;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node15 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::COS;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node16 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SIN;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node17 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::POW_TENSOR_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node18 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MEAN_DIM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node19 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node20 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::RSQRT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node21 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::T;
  sym_int0 = CreateSymInt(8192, 0);
  sym_int1 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(8192, 0);
  sym_int1 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node22 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(8192, 0);
  sym_int1 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node23 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::NEG;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  auto sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node24 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(-7, 8);
  auto sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>(
      {sym_int0, sym_int1, sym_int2, sym_int3, sym_int4}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(-7, 8);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>(
      {sym_int0, sym_int1, sym_int2, sym_int3, sym_int4}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node25 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::_UNSAFE_VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(-7, 8);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>(
      {sym_int0, sym_int1, sym_int2, sym_int3, sym_int4}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node26 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MUL_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node27 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::_SOFTMAX;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node28 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SILU;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(28672, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(28672, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node29 = flatflow::CreateNode(builder, op, args, meta);

  auto nodes = builder.CreateVector(
      {node0,  node1,  node2,  node3,  node4,  node5,  node6,  node7,
       node8,  node9,  node10, node11, node12, node13, node14, node15,
       node16, node17, node18, node19, node20, node21, node22, node23,
       node24, node25, node26, node27, node28, node29});
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

TEST_F(SchedulerWithRemainderOnlyTest, Llama31) {
  auto builder = flatbuffers::FlatBufferBuilder();

  auto op = flatflow::Operator::EMBEDDING;
  auto sym_int0 = CreateSymInt(128256, 0);
  auto sym_int1 = CreateSymInt(8192, 0);
  auto shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  auto meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  auto meta1 = flatflow::CreateTensorMetadata(builder, shape);
  auto args = builder.CreateVector({meta0, meta1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  auto sym_int2 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  auto meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node0 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SYM_SIZE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  shape = builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>());
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node1 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::ARANGE_START;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(-7, 8);
  shape =
      builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>({sym_int0}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node2 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(-7, 8);
  shape =
      builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>({sym_int0}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node3 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::FULL;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node4 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::TRIU;
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node5 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::ARANGE;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(-6, 8);
  shape =
      builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>({sym_int0}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node6 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(-7, 8);
  shape =
      builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>({sym_int0}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node7 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::GT_TENSOR;
  sym_int0 = CreateSymInt(-6, 8);
  shape =
      builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>({sym_int0}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node8 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(-6, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node9 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node10 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::_TO_COPY;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node11 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node12 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node13 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node14 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::CAT;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node15 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::COS;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node16 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SIN;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node17 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::POW_TENSOR_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node18 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MEAN_DIM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node19 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node20 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::RSQRT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node21 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::T;
  sym_int0 = CreateSymInt(8192, 0);
  sym_int1 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(8192, 0);
  sym_int1 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node22 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(8192, 0);
  sym_int1 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = CreateSymInt(-7, 8);
  sym_int1 = CreateSymInt(8192, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node23 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::NEG;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  auto sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node24 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(-7, 8);
  auto sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>(
      {sym_int0, sym_int1, sym_int2, sym_int3, sym_int4}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(-7, 8);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>(
      {sym_int0, sym_int1, sym_int2, sym_int3, sym_int4}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node25 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::_UNSAFE_VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(-7, 8);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(std::vector<flatflow::SymInt>(
      {sym_int0, sym_int1, sym_int2, sym_int3, sym_int4}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node26 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MUL_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node27 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::_SOFTMAX;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(-7, 8);
  sym_int3 = CreateSymInt(-7, 8);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2, sym_int3}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node28 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SILU;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(28672, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(-7, 8);
  sym_int2 = CreateSymInt(28672, 0);
  shape = builder.CreateVectorOfStructs(
      std::vector<flatflow::SymInt>({sym_int0, sym_int1, sym_int2}));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node29 = flatflow::CreateNode(builder, op, args, meta);

  auto nodes = builder.CreateVector(
      {node0,  node1,  node2,  node3,  node4,  node5,  node6,  node7,
       node8,  node9,  node10, node11, node12, node13, node14, node15,
       node16, node17, node18, node19, node20, node21, node22, node23,
       node24, node25, node26, node27, node28, node29});
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
