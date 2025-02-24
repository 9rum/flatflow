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

#include "flatflow/ops/ops.h"

#include <initializer_list>

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

namespace {

class SymbolicTraceTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!absl::log_internal::IsInitialized()) {
      absl::InitializeLog();
      absl::SetStderrThreshold(absl::LogSeverity::kInfo);
    }
  }
};

// This test checks whether symbolic tracing works as intended for Llama 3.1.
// Specifically, this test answers the following questions:
//
// * Does the current implementation of operator registry support all the
//   operators included in Llama 3.1?
// * Does symbolic tracing work deterministically on the same model?
//
// Note that the original Llama 3.1 models have thousands of nodes when
// converted to graphs; 3887 nodes for the 8B, 9567 nodes for the 70B.
// This produces hundreds of thousands of lines of code when generated, severely
// slowing down the build; 60,178 lines for the 8B, 148,226 lines for the 70B.
// To this end, this test simulates Llama 3.1 where the operators appear only
// once, limiting the computational graph to have only 30 nodes.
TEST_F(SymbolicTraceTest, Llama31) {
  auto builder = flatbuffers::FlatBufferBuilder();

  auto op = flatflow::Operator::EMBEDDING;
  auto sym_int0 = flatflow::CreateSymInt(builder, 128256, 0);
  auto sym_int1 = flatflow::CreateSymInt(builder, 8192, 0);
  auto shape = builder.CreateVector({sym_int0, sym_int1});
  auto meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  shape = builder.CreateVector({sym_int0, sym_int1});
  auto meta1 = flatflow::CreateTensorMetadata(builder, shape);
  auto args = builder.CreateVector({meta0, meta1});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  auto sym_int2 = flatflow::CreateSymInt(builder, 8192, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  auto meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node0 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SYM_SIZE_INT;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  shape = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::SymInt>>());
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node1 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::ARANGE_START;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = flatflow::CreateSymInt(builder, -7, 8);
  shape = builder.CreateVector({sym_int0});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node2 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::UNSQUEEZE;
  sym_int0 = flatflow::CreateSymInt(builder, -7, 8);
  shape = builder.CreateVector({sym_int0});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node3 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::FULL;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int1 = flatflow::CreateSymInt(builder, -6, 8);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node4 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::TRIU;
  sym_int0 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int1 = flatflow::CreateSymInt(builder, -6, 8);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int1 = flatflow::CreateSymInt(builder, -6, 8);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node5 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::ARANGE;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = flatflow::CreateSymInt(builder, -6, 8);
  shape = builder.CreateVector({sym_int0});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node6 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::VIEW;
  sym_int0 = flatflow::CreateSymInt(builder, -7, 8);
  shape = builder.CreateVector({sym_int0});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int1 = flatflow::CreateSymInt(builder, 1, 0);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node7 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::GT_TENSOR;
  sym_int0 = flatflow::CreateSymInt(builder, -6, 8);
  shape = builder.CreateVector({sym_int0});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int1 = flatflow::CreateSymInt(builder, 1, 0);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int1 = flatflow::CreateSymInt(builder, -6, 8);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node8 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MUL_TENSOR;
  sym_int0 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int1 = flatflow::CreateSymInt(builder, -6, 8);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int1 = flatflow::CreateSymInt(builder, -6, 8);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int1 = flatflow::CreateSymInt(builder, -6, 8);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node9 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SLICE_TENSOR;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node10 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::_TO_COPY;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  sym_int2 = flatflow::CreateSymInt(builder, 1, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  sym_int2 = flatflow::CreateSymInt(builder, 1, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node11 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::EXPAND;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  sym_int2 = flatflow::CreateSymInt(builder, 1, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  sym_int2 = flatflow::CreateSymInt(builder, 1, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node12 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::BMM;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  sym_int2 = flatflow::CreateSymInt(builder, 1, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int2 = flatflow::CreateSymInt(builder, -7, 8);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  sym_int2 = flatflow::CreateSymInt(builder, -7, 8);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node13 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::TRANSPOSE_INT;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  sym_int2 = flatflow::CreateSymInt(builder, -7, 8);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 64, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node14 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::CAT;
  args = builder.CreateVector(
      std::initializer_list<flatbuffers::Offset<flatflow::TensorMetadata>>());
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 128, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node15 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::COS;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 128, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 128, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node16 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SIN;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 128, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 128, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node17 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::POW_TENSOR_SCALAR;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 8192, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 8192, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node18 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MEAN_DIM;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 8192, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 1, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node19 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::ADD_TENSOR;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 1, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 1, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node20 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::RSQRT;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 1, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 1, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node21 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::T;
  sym_int0 = flatflow::CreateSymInt(builder, 8192, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 8192, 0);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 8192, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 8192, 0);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node22 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MM;
  sym_int0 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int1 = flatflow::CreateSymInt(builder, 8192, 0);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  sym_int0 = flatflow::CreateSymInt(builder, 8192, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 8192, 0);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta1 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0, meta1});
  sym_int0 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int1 = flatflow::CreateSymInt(builder, 8192, 0);
  shape = builder.CreateVector({sym_int0, sym_int1});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node23 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::NEG;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  sym_int2 = flatflow::CreateSymInt(builder, -7, 8);
  auto sym_int3 = flatflow::CreateSymInt(builder, 64, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2, sym_int3});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  sym_int2 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int3 = flatflow::CreateSymInt(builder, 64, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2, sym_int3});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node24 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::CLONE;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 8, 0);
  sym_int2 = flatflow::CreateSymInt(builder, 8, 0);
  sym_int3 = flatflow::CreateSymInt(builder, -7, 8);
  auto sym_int4 = flatflow::CreateSymInt(builder, 128, 0);
  shape =
      builder.CreateVector({sym_int0, sym_int1, sym_int2, sym_int3, sym_int4});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 8, 0);
  sym_int2 = flatflow::CreateSymInt(builder, 8, 0);
  sym_int3 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int4 = flatflow::CreateSymInt(builder, 128, 0);
  shape =
      builder.CreateVector({sym_int0, sym_int1, sym_int2, sym_int3, sym_int4});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node25 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::_UNSAFE_VIEW;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 8, 0);
  sym_int2 = flatflow::CreateSymInt(builder, 8, 0);
  sym_int3 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int4 = flatflow::CreateSymInt(builder, 128, 0);
  shape =
      builder.CreateVector({sym_int0, sym_int1, sym_int2, sym_int3, sym_int4});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  sym_int2 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int3 = flatflow::CreateSymInt(builder, 128, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2, sym_int3});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node26 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::MUL_SCALAR;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  sym_int2 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int3 = flatflow::CreateSymInt(builder, 128, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2, sym_int3});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  sym_int2 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int3 = flatflow::CreateSymInt(builder, 128, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2, sym_int3});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node27 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::_SOFTMAX;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  sym_int2 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int3 = flatflow::CreateSymInt(builder, -7, 8);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2, sym_int3});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, 64, 0);
  sym_int2 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int3 = flatflow::CreateSymInt(builder, -7, 8);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2, sym_int3});
  meta = flatflow::CreateTensorMetadata(builder, shape);
  auto node28 = flatflow::CreateNode(builder, op, args, meta);

  op = flatflow::Operator::SILU;
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 28672, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
  meta0 = flatflow::CreateTensorMetadata(builder, shape);
  args = builder.CreateVector({meta0});
  sym_int0 = flatflow::CreateSymInt(builder, 1, 0);
  sym_int1 = flatflow::CreateSymInt(builder, -7, 8);
  sym_int2 = flatflow::CreateSymInt(builder, 28672, 0);
  shape = builder.CreateVector({sym_int0, sym_int1, sym_int2});
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
  const auto trace =
      flatflow::symbolic_trace(graph);  // 1284 s1^2 + 67178363 s1

  EXPECT_EQ(trace(0), 0);
  EXPECT_EQ(trace(1), 67179647);
  EXPECT_EQ(trace(1024), 70137015296);
  EXPECT_EQ(trace(2048), 142966773760);
}

}  // namespace
