// SPDX-License-Identifier: Apache-2.0

#include "flatflow/ops/view.h"

#include <array>
#include <cstdint>
#include <type_traits>

#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/verifier.h"
#include "gtest/gtest.h"

#include "flatflow/ops/graph_generated.h"
#include "flatflow/ops/scalar_type_generated.h"

namespace flatflow {
namespace {

// Tests whether views are as inexpensive to pass by value as raw pointers.
static_assert(std::is_trivially_copyable_v<detail::SymInt>);
static_assert(std::is_trivially_copyable_v<detail::TensorMetadata>);
static_assert(std::is_trivially_copyable_v<detail::Vector<detail::SymInt>>);
static_assert(
    std::is_trivially_copyable_v<detail::Vector<detail::TensorMetadata>>);

// Tests whether `SymInt` provides a transparent view into the serialized
// symbolic integer.
TEST(ViewTest, FromSymInt) {
  auto builder = flatbuffers::FlatBufferBuilder();

  const auto shape = builder.CreateVectorOfStructs<SymInt>({
      SymInt(std::to_array<int64_t>({0, 0})),
      SymInt(std::to_array<int64_t>({4096, 0})),
      SymInt(std::to_array<int64_t>({0, 1})),
      SymInt(std::to_array<int64_t>({-1, 2})),
  });
  const auto root = CreateTensorMetadata(builder, ScalarType::float32, shape);
  builder.Finish(root);

  const auto meta =
      flatbuffers::GetRoot<TensorMetadata>(builder.GetBufferPointer());

  const auto s0 = detail::SymInt(meta->shape()->Get(0));
  ASSERT_EQ(s0[0], 0);
  ASSERT_EQ(s0[1], 0);

  const auto s1 = detail::SymInt(meta->shape()->Get(1));
  ASSERT_EQ(s1[0], 4096);
  ASSERT_EQ(s1[1], 0);

  const auto s2 = detail::SymInt(meta->shape()->Get(2));
  ASSERT_EQ(s2[0], 0);
  ASSERT_EQ(s2[1], 1);

  const auto s3 = detail::SymInt(meta->shape()->Get(3));
  ASSERT_EQ(s3[0], -1);
  ASSERT_EQ(s3[1], 2);
}

// Tests whether `TensorMetadata` provides a transparent view of the serialized
// tensor metadata.
TEST(ViewTest, FromTensorMetadata) {
  auto builder = flatbuffers::FlatBufferBuilder();

  const auto shape = builder.CreateVectorOfStructs<SymInt>({
      SymInt(std::to_array<int64_t>({4096, 0})),
      SymInt(std::to_array<int64_t>({0, 1})),
  });
  const auto root = CreateTensorMetadata(builder, ScalarType::bfloat16, shape);
  builder.Finish(root);

  const auto meta = detail::TensorMetadata(
      flatbuffers::GetRoot<TensorMetadata>(builder.GetBufferPointer()));
  ASSERT_EQ(meta.dtype(), ScalarType::bfloat16);
  ASSERT_EQ(meta.shape().size(), 2);
  ASSERT_EQ(meta.shape()[0][0], 4096);
  ASSERT_EQ(meta.shape()[0][1], 0);
  ASSERT_EQ(meta.shape()[1][0], 0);
  ASSERT_EQ(meta.shape()[1][1], 1);
}

// Tests whether `Vector` provides a transparent view of a FlatBuffers vector
// whose elements are structures.
TEST(ViewTest, FromVectorOfStructs) {
  auto builder = flatbuffers::FlatBufferBuilder();

  const auto shape = builder.CreateVectorOfStructs<SymInt>({
      SymInt(std::to_array<int64_t>({0, 0})),
      SymInt(std::to_array<int64_t>({4096, 0})),
      SymInt(std::to_array<int64_t>({0, 1})),
      SymInt(std::to_array<int64_t>({-1, 2})),
  });
  const auto root = CreateTensorMetadata(builder, ScalarType::float32, shape);
  builder.Finish(root);

  const auto meta =
      flatbuffers::GetRoot<TensorMetadata>(builder.GetBufferPointer());

  const auto v = detail::Vector<detail::SymInt>(meta->shape());
  ASSERT_EQ(v.size(), 4);
  ASSERT_EQ(v[0][0], 0);
  ASSERT_EQ(v[0][1], 0);
  ASSERT_EQ(v[1][0], 4096);
  ASSERT_EQ(v[1][1], 0);
  ASSERT_EQ(v[2][0], 0);
  ASSERT_EQ(v[2][1], 1);
  ASSERT_EQ(v[3][0], -1);
  ASSERT_EQ(v[3][1], 2);
}

// Tests whether `Vector` provides a transparent view of a FlatBuffers vector
// whose elements are tables.
TEST(ViewTest, FromVectorOfTables) {
  auto builder = flatbuffers::FlatBufferBuilder();

  const auto input =
      CreateTensorMetadata(builder, ScalarType::bfloat16,
                           builder.CreateVectorOfStructs<SymInt>({
                               SymInt(std::to_array<int64_t>({0, 1})),
                               SymInt(std::to_array<int64_t>({4096, 0})),
                           }));
  const auto other =
      CreateTensorMetadata(builder, ScalarType::bfloat16,
                           builder.CreateVectorOfStructs<SymInt>({
                               SymInt(std::to_array<int64_t>({4096, 0})),
                               SymInt(std::to_array<int64_t>({1024, 0})),
                           }));
  const auto out =
      CreateTensorMetadata(builder, ScalarType::bfloat16,
                           builder.CreateVectorOfStructs<SymInt>({
                               SymInt(std::to_array<int64_t>({0, 1})),
                               SymInt(std::to_array<int64_t>({1024, 0})),
                           }));

  const auto node = CreateNode(builder, Operator::mm,
                               builder.CreateVector({input, other}), out);
  const auto root = CreateGraph(builder, builder.CreateVector({node}));
  builder.Finish(root);

  auto verifier =
      flatbuffers::Verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(VerifyGraphBuffer(verifier));

  const auto graph = GetGraph(builder.GetBufferPointer());
  ASSERT_EQ(graph->nodes()->size(), 1);

  const auto args =
      detail::Vector<detail::TensorMetadata>(graph->nodes()->Get(0)->args());
  ASSERT_EQ(args.size(), 2);
  ASSERT_EQ(args[0].dtype(), ScalarType::bfloat16);
  ASSERT_EQ(args[0].shape().size(), 2);
  ASSERT_EQ(args[0].shape()[0][0], 0);
  ASSERT_EQ(args[0].shape()[0][1], 1);
  ASSERT_EQ(args[0].shape()[1][0], 4096);
  ASSERT_EQ(args[0].shape()[1][1], 0);
  ASSERT_EQ(args[1].dtype(), ScalarType::bfloat16);
  ASSERT_EQ(args[1].shape().size(), 2);
  ASSERT_EQ(args[1].shape()[0][0], 4096);
  ASSERT_EQ(args[1].shape()[0][1], 0);
  ASSERT_EQ(args[1].shape()[1][0], 1024);
  ASSERT_EQ(args[1].shape()[1][1], 0);

  const auto meta = detail::TensorMetadata(graph->nodes()->Get(0)->meta());
  ASSERT_EQ(meta.dtype(), ScalarType::bfloat16);
  ASSERT_EQ(meta.shape()[0][0], 0);
  ASSERT_EQ(meta.shape()[0][1], 1);
  ASSERT_EQ(meta.shape()[1][0], 1024);
  ASSERT_EQ(meta.shape()[1][1], 0);
}

}  // namespace
}  // namespace flatflow
