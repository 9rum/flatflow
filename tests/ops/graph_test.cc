// SPDX-License-Identifier: Apache-2.0

#include "flatflow/ops/graph.h"

#include <vector>

#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/verifier.h"
#include "gtest/gtest.h"

#include "flatflow/ops/graph_generated.h"
#include "flatflow/ops/operator_generated.h"
#include "flatflow/ops/scalar_type_generated.h"

namespace flatflow {
namespace {

TEST(GraphTest, FromEmptyGraph) {
  auto builder = flatbuffers::FlatBufferBuilder();

  const auto nodes =
      builder.CreateVector(std::vector<flatbuffers::Offset<internal::Node>>());
  const auto root = internal::CreateGraph(builder, nodes);
  builder.Finish(root);

  auto verifier =
      flatbuffers::Verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(internal::VerifyGraphBuffer(verifier));

  const auto graph = internal::GetGraph(builder.GetBufferPointer());
  EXPECT_EQ(Graph(graph), Graph());
}

TEST(GraphTest, FromFeedForwardNetwork) {
  const auto ffn = Graph{{
      Node{
          Operator::mm,
          {
              TensorMetadata{
                  ScalarType::bfloat16,
                  {SymInt{0, 1}, SymInt{4096, 0}},
              },
              TensorMetadata{
                  ScalarType::bfloat16,
                  {SymInt{4096, 0}, SymInt{4096, 0}},
              },
          },
          TensorMetadata{
              ScalarType::bfloat16,
              {SymInt{0, 1}, SymInt{4096, 0}},
          },
      },
      Node{
          Operator::relu,
          {TensorMetadata{
              ScalarType::bfloat16,
              {SymInt{0, 1}, SymInt{4096, 0}},
          }},
          TensorMetadata{
              ScalarType::bfloat16,
              {SymInt{0, 1}, SymInt{4096, 0}},
          },
      },
      Node{
          Operator::mm,
          {
              TensorMetadata{
                  ScalarType::bfloat16,
                  {SymInt{0, 1}, SymInt{4096, 0}},
              },
              TensorMetadata{
                  ScalarType::bfloat16,
                  {SymInt{4096, 0}, SymInt{1024, 0}},
              },
          },
          TensorMetadata{
              ScalarType::bfloat16,
              {SymInt{0, 1}, SymInt{1024, 0}},
          },
      },
  }};

  auto builder = flatbuffers::FlatBufferBuilder();

  auto nodes = std::vector<flatbuffers::Offset<internal::Node>>();
  for (const auto &node : ffn.nodes) {
    auto args = std::vector<flatbuffers::Offset<internal::TensorMetadata>>();
    for (const auto &arg : node.args) {
      auto shape = std::vector<internal::SymInt>();
      for (const auto &s : arg.shape) {
        shape.emplace_back(s.data);
      }
      args.emplace_back(internal::CreateTensorMetadata(
          builder, arg.dtype, builder.CreateVectorOfStructs(shape)));
    }

    auto shape = std::vector<internal::SymInt>();
    for (const auto &s : node.meta.shape) {
      shape.emplace_back(s.data);
    }
    const auto meta = internal::CreateTensorMetadata(
        builder, node.meta.dtype, builder.CreateVectorOfStructs(shape));

    nodes.emplace_back(internal::CreateNode(builder, node.target,
                                            builder.CreateVector(args), meta));
  }

  const auto root = internal::CreateGraph(builder, builder.CreateVector(nodes));
  builder.Finish(root);

  auto verifier =
      flatbuffers::Verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(internal::VerifyGraphBuffer(verifier));

  const auto graph = internal::GetGraph(builder.GetBufferPointer());
  EXPECT_EQ(Graph(graph), ffn);
}

}  // namespace
}  // namespace flatflow
