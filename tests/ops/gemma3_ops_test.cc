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

// This test checks whether the current implementation of operator registry
// supports all the operators included in Gemma 3.
TEST_F(SymbolicTraceTest, Gemma3) {
  auto builder = flatbuffers::FlatBufferBuilder();

  auto target = flatflow::Operator::EMBEDDING;
  auto sym_int0 = CreateSymInt(262144, 0);
  auto sym_int1 = CreateSymInt(1152, 0);
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
  auto sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  auto meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node0 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts());
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts());
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node1 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts());
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node2 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ARANGE_START;
  args = builder.CreateVector(
      std::vector<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node3 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node4 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::FULL;
  args = builder.CreateVector(
      std::vector<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node5 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRIU;
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
  auto node6 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ARANGE;
  args = builder.CreateVector(
      std::vector<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node7 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node8 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::GT_TENSOR;
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
  auto node9 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
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
  auto node10 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(128, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node11 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(128, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(128, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node12 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(128, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node13 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node14 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node15 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node16 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node17 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node18 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node19 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node20 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node21 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node22 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(128, 0);
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
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node23 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node24 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(0, 1);
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

  target = flatflow::Operator::CAT;
  args = builder.CreateVector(
      std::vector<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node26 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::COS;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node27 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node28 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SIN;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node29 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node30 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node31 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node32 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node33 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node34 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ONES_LIKE;
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
  auto node35 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRIL;
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
  auto node36 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SCALAR_TENSOR;
  args = builder.CreateVector(
      std::vector<flatbuffers::Offset<flatflow::TensorMetadata>>());
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts());
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node37 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::WHERE_SELF;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts());
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  auto arg2 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1, arg2});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node38 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node39 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::POW_TENSOR_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node40 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MEAN_DIM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
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
  auto node41 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node42 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node43 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
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
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node44 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  sym_int0 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node45 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node46 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node47 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(1024, 0);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1152, 0);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node48 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node49 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1152, 0);
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
  auto node50 = flatflow::CreateNode(builder, target, args, meta);

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
  auto node51 = flatflow::CreateNode(builder, target, args, meta);

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
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node52 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node53 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(256, 0);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1152, 0);
  sym_int1 = CreateSymInt(256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node54 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1152, 0);
  sym_int1 = CreateSymInt(256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node55 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(256, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node56 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node57 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node58 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node59 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::POW_TENSOR_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node60 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MEAN_DIM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node61 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node62 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::RSQRT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node63 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node64 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  sym_int0 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node65 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node66 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node67 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node68 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::POW_TENSOR_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node69 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MEAN_DIM;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node70 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node71 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::RSQRT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node72 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node73 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node74 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node75 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node76 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node77 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::NEG;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node78 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node79 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node80 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node81 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::NEG;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node82 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node83 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
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
  auto node84 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  auto sym_int4 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node85 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node86 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node87 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node88 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node89 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node90 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(256, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node91 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_SCALAR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(256, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(256, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node92 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node93 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(4, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node94 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(256, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(256, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node95 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(256, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(4, 0);
  sym_int1 = CreateSymInt(256, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node96 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(4, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(4, 0);
  sym_int1 = CreateSymInt(256, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(4, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node97 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(4, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node98 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
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
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node99 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_SOFTMAX;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node100 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node101 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(4, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node102 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  sym_int0 = CreateSymInt(4, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(4, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(4, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node103 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(4, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(4, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node104 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node105 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(256, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1024, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node106 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(1152, 0);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1024, 0);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node107 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1024, 0);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node108 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node109 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node110 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(6912, 0);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1152, 0);
  sym_int1 = CreateSymInt(6912, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node111 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1152, 0);
  sym_int1 = CreateSymInt(6912, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(6912, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node112 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(6912, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(6912, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node113 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::GELU;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(6912, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(6912, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node114 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(6912, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(6912, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(6912, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node115 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(1152, 0);
  sym_int1 = CreateSymInt(6912, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(6912, 0);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node116 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(6912, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(6912, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node117 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(6912, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(6912, 0);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node118 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1152, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node119 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  sym_int0 = CreateSymInt(262144, 0);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1152, 0);
  sym_int1 = CreateSymInt(262144, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node120 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1152, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = CreateSymInt(1152, 0);
  sym_int1 = CreateSymInt(262144, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0, arg1});
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(262144, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node121 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(262144, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({arg0});
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(262144, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node122 = flatflow::CreateNode(builder, target, args, meta);

  auto nodes = builder.CreateVector(
      {node0,   node1,   node2,   node3,   node4,   node5,   node6,   node7,
       node8,   node9,   node10,  node11,  node12,  node13,  node14,  node15,
       node16,  node17,  node18,  node19,  node20,  node21,  node22,  node23,
       node24,  node25,  node26,  node27,  node28,  node29,  node30,  node31,
       node32,  node33,  node34,  node35,  node36,  node37,  node38,  node39,
       node40,  node41,  node42,  node43,  node44,  node45,  node46,  node47,
       node48,  node49,  node50,  node51,  node52,  node53,  node54,  node55,
       node56,  node57,  node58,  node59,  node60,  node61,  node62,  node63,
       node64,  node65,  node66,  node67,  node68,  node69,  node70,  node71,
       node72,  node73,  node74,  node75,  node76,  node77,  node78,  node79,
       node80,  node81,  node82,  node83,  node84,  node85,  node86,  node87,
       node88,  node89,  node90,  node91,  node92,  node93,  node94,  node95,
       node96,  node97,  node98,  node99,  node100, node101, node102, node103,
       node104, node105, node106, node107, node108, node109, node110, node111,
       node112, node113, node114, node115, node116, node117, node118, node119,
       node120, node121, node122});
  auto root = flatflow::CreateGraph(builder, nodes);
  builder.Finish(root);

  auto graph =
      flatbuffers::GetRoot<flatflow::Graph>(builder.GetBufferPointer());
  const auto trace =
      flatflow::symbolic_trace(graph);  // 825 s0^2 + 128252522 s0

  EXPECT_EQ(trace(0), 0);
  EXPECT_EQ(trace(1), 128253347);
  EXPECT_EQ(trace(1024), 132195657728);
  EXPECT_EQ(trace(2048), 266121465856);
}

}  // namespace
