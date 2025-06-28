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

#include <cstdint>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/internal/globals.h"
#include "flatbuffers/flatbuffers.h"
#include "gtest/gtest.h"

#include "flatflow/ops/dtype_generated.h"
#include "flatflow/ops/graph_generated.h"
#include "flatflow/ops/node_generated.h"
#include "flatflow/ops/operator_generated.h"
#include "flatflow/ops/ops.h"

namespace {

flatflow::SymInt CreateSymInt(std::int64_t x, std::int64_t y) {
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
// supports all the operators included in Mistral 3.
TEST_F(SymbolicTraceTest, Mistral3) {
  auto builder = flatbuffers::FlatBufferBuilder();

  auto target = flatflow::Operator::EMBEDDING;
  auto dtype = flatflow::ScalarType::BFLOAT16;
  auto sym_int0 = CreateSymInt(131072, 0);
  auto sym_int1 = CreateSymInt(5120, 0);
  auto shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  auto arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  auto arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  auto sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  auto meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node0 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ARANGE_START;
  args = builder.CreateVector(
      std::vector<flatbuffers::Offset<flatflow::TensorMetadata>>());
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node1 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node2 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::FULL;
  args = builder.CreateVector(
      std::vector<flatbuffers::Offset<flatflow::TensorMetadata>>());
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node3 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ARANGE;
  args = builder.CreateVector(
      std::vector<flatbuffers::Offset<flatflow::TensorMetadata>>());
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node4 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node5 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::GT_TENSOR;
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BOOL;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node6 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BOOL;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node7 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node8 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node9 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node10 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node11 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node12 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node13 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node14 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node15 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  dtype = flatflow::ScalarType::INT64;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node16 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node17 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node18 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node19 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node20 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node21 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node22 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(64, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node23 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CAT;
  args = builder.CreateVector(
      std::vector<flatbuffers::Offset<flatflow::TensorMetadata>>());
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node24 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::COS;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node25 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node26 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SIN;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node27 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node28 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node29 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::POW_TENSOR_SCALAR;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node30 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MEAN_DIM;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node31 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node32 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::RSQRT;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node33 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node34 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_TO_COPY;
  dtype = flatflow::ScalarType::FLOAT32;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node35 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node36 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(5120, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node37 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node38 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(5120, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node39 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node40 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  auto sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node41 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node42 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1024, 0);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(5120, 0);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node43 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(5120, 0);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node44 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1024, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1024, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node45 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1024, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node46 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(8, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node47 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node48 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node49 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node50 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::NEG;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node51 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node52 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node53 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node54 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::NEG;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(64, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node55 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node56 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  auto sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node57 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node58 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(1, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node59 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node60 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_UNSAFE_VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(8, 0);
  sym_int2 = CreateSymInt(4, 0);
  sym_int3 = CreateSymInt(0, 1);
  sym_int4 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3, sym_int4));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node61 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node62 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(1, 1);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node63 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::UNSQUEEZE;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node64 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node65 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(1, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node66 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_SCALAR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node67 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::TRANSPOSE_INT;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node68 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_SCALAR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node69 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node70 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node71 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node72 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(128, 0);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node73 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(128, 0);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node74 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node75 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(1, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node76 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::_SOFTMAX;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node77 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::EXPAND;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node78 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node79 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::BMM;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(0, 1);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node80 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(32, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(32, 0);
  sym_int2 = CreateSymInt(0, 1);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node81 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::CLONE;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node82 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32, 0);
  sym_int3 = CreateSymInt(128, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2, sym_int3));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(4096, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node83 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(5120, 0);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node84 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(4096, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(4096, 0);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node85 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node86 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::ADD_TENSOR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node87 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(32768, 0);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(5120, 0);
  sym_int1 = CreateSymInt(32768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node88 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(5120, 0);
  sym_int1 = CreateSymInt(32768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(32768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node89 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(32768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node90 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SILU;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node91 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MUL_TENSOR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node92 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(5120, 0);
  sym_int1 = CreateSymInt(32768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(32768, 0);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node93 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(32768, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(32768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node94 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(32768, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(32768, 0);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node95 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::SLICE_TENSOR;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(5120, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node96 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::T;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(131072, 0);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(5120, 0);
  sym_int1 = CreateSymInt(131072, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node97 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::MM;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(5120, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(5120, 0);
  sym_int1 = CreateSymInt(131072, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg1 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0, arg1});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(131072, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node98 = flatflow::CreateNode(builder, target, args, meta);

  target = flatflow::Operator::VIEW;
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(0, 1);
  sym_int1 = CreateSymInt(131072, 0);
  shape =
      builder.CreateVectorOfStructs(CreateVectorOfSymInts(sym_int0, sym_int1));
  arg0 = flatflow::CreateTensorMetadata(builder, dtype, shape);
  args = builder.CreateVector({arg0});
  dtype = flatflow::ScalarType::BFLOAT16;
  sym_int0 = CreateSymInt(1, 0);
  sym_int1 = CreateSymInt(0, 1);
  sym_int2 = CreateSymInt(131072, 0);
  shape = builder.CreateVectorOfStructs(
      CreateVectorOfSymInts(sym_int0, sym_int1, sym_int2));
  meta = flatflow::CreateTensorMetadata(builder, dtype, shape);
  auto node99 = flatflow::CreateNode(builder, target, args, meta);

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
       node90, node91, node92, node93, node94, node95, node96, node97, node98,
       node99});
  auto root = flatflow::CreateGraph(builder, nodes);
  builder.Finish(root);

  auto graph =
      flatbuffers::GetRoot<flatflow::Graph>(builder.GetBufferPointer());
  const auto trace =
      flatflow::symbolic_trace(graph);  // 16609 s0^2 + 2107858692 s0

  EXPECT_EQ(trace(0), 0);
  EXPECT_EQ(trace(1), 2107875301);
  EXPECT_EQ(trace(1024), 2175863099392);
  EXPECT_EQ(trace(2048), 4386557796352);
}

}  // namespace
