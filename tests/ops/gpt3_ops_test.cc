// Copyright 2025 The FlatFlow Authors
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

#include <vector>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/internal/globals.h"
#include "absl/log/log.h"
#include "flatbuffers/flatbuffers.h"
#include "gtest/gtest.h"

#include "flatflow/ops/graph_generated.h"
#include "flatflow/ops/node_generated.h"
#include "flatflow/ops/operator_generated.h"
#include "flatflow/ops/ops.h"

namespace {

flatflow::SymInt CreateSymInt(int64_t x, int64_t y) {
  return flatflow::SymInt(flatbuffers::make_span({x, y}));
}

template <typename... Args>
std::vector<flatflow::SymInt> CreateVectorOfSymInts(Args... args) {
  return std::vector<flatflow::SymInt>{args...};
}

class SymbolicTraceTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!absl::log_internal::IsInitialized()) {
      absl::InitializeLog();
      absl::SetStderrThreshold(absl::LogSeverity::kInfo);
    }
  }
};

// This test verifies whether the current implementation of operator registry
// supports all the operators included in GPT-3.
TEST_F(SymbolicTraceTest, GPT3) {
  auto builder = flatbuffers::FlatBufferBuilder();

  auto target = flatflow::Operator::VIEW;
  auto sym_int0 = CreateSymInt(1, 0);
  auto sym_int1 = CreateSymInt(0, 1);
  auto shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  auto arg0 = flatflow::CreateTensorMetadata(builder, shape);
  auto args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  auto meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node0 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ARANGE_START;
  args = builder.CreateVector(
      std::vector<flatbuffers::Offset<flatflow::TensorMetadata>>());
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

  target = flatflow::Operator::EMBEDDING;
  sym_int0 = CreateSymInt(50257, 0);
  sym_int1 = CreateSymInt(768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  auto arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  auto sym_int2 = CreateSymInt(768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node3 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EMBEDDING;
  sym_int0 = CreateSymInt(1024, 0);
  sym_int1 = CreateSymInt(768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node4 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node5 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::FULL;
  args = builder.CreateVector(
      std::vector<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node6 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ARANGE;
  args = builder.CreateVector(
      std::vector<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node7 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node8 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node9 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::LT_TENSOR;
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node10 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MASKED_FILL_SCALAR;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node11 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node12 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node13 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  auto sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node14 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node15 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node16 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node17 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::NATIVE_LAYER_NORM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(768, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(768, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  auto arg2 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1, arg2});
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts());
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node18 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node19 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADDMM;
  sym_int0 = CreateSymInt(2304, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(768, 0);
  sym_int1 = CreateSymInt(2304, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg2 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1, arg2});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(2304, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node20 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(2304, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(2304, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node21 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SPLIT_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(2304, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts());
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node22 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::PERMUTE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(12, 0);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node23 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node24 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(64, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node25 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(64, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(64, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node26 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node27 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(12, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node28 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(64, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(64, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node29 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(64, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(12, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node30 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(12, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(12, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(12, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node31 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(12, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node32 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
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
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node33 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_SOFTMAX;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node34 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node35 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(12, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node36 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(12, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(12, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(12, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node37 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(12, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(12, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node38 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(12, 0);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(12, 0);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node39 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(12, 0);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node40 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADDMM;
  sym_int0 = CreateSymInt(768, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(768, 0);
  sym_int1 = CreateSymInt(768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg2 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1, arg2});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node41 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node42 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADDMM;
  sym_int0 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(768, 0);
  sym_int1 = CreateSymInt(3072, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg2 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1, arg2});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(3072, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node43 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(3072, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node44 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node45 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::POW_TENSOR_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node46 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node47 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TANH;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node48 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node49 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node50 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(3072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(3072, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node51 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADDMM;
  sym_int0 = CreateSymInt(768, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(3072, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(3072, 0);
  sym_int1 = CreateSymInt(768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg2 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1, arg2});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node52 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(50257, 0);
  sym_int1 = CreateSymInt(768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(768, 0);
  sym_int1 = CreateSymInt(50257, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node53 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(768, 0);
  sym_int1 = CreateSymInt(50257, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(50257, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node54 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(50257, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(50257, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node55 = flatflow::CreateNode(builder, target, args, meta);

  auto nodes = builder.CreateVector(
      {node0,  node1,  node2,  node3,  node4,  node5,  node6,  node7,
       node8,  node9,  node10, node11, node12, node13, node14, node15,
       node16, node17, node18, node19, node20, node21, node22, node23,
       node24, node25, node26, node27, node28, node29, node30, node31,
       node32, node33, node34, node35, node36, node37, node38, node39,
       node40, node41, node42, node43, node44, node45, node46, node47,
       node48, node49, node50, node51, node52, node53, node54, node55});
  auto root = flatflow::CreateGraph(builder, nodes);
  builder.Finish(root);

  auto graph =
      flatbuffers::GetRoot<flatflow::Graph>(builder.GetBufferPointer());
  const auto trace =
      flatflow::symbolic_trace(graph);  // 3156 s0^2 + 91416581 s0

  EXPECT_EQ(trace(0), 0);
  EXPECT_EQ(trace(1), 91419737);
  EXPECT_EQ(trace(1024), 96919884800);
  EXPECT_EQ(trace(2048), 200458381312);
}

}  // namespace
