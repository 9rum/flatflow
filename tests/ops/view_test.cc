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
static_assert(std::is_trivially_copyable_v<detail::FakeTensor>);
static_assert(std::is_trivially_copyable_v<detail::Vector<detail::SymInt>>);
static_assert(std::is_trivially_copyable_v<detail::Vector<detail::FakeTensor>>);

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
  const auto root = CreateFakeTensor(builder, ScalarType::float32, shape);
  builder.Finish(root);

  const auto tensor =
      flatbuffers::GetRoot<FakeTensor>(builder.GetBufferPointer());

  const auto s0 = detail::SymInt(tensor->shape()->Get(0));
  ASSERT_EQ(s0[0], 0);
  ASSERT_EQ(s0[1], 0);

  const auto s1 = detail::SymInt(tensor->shape()->Get(1));
  ASSERT_EQ(s1[0], 4096);
  ASSERT_EQ(s1[1], 0);

  const auto s2 = detail::SymInt(tensor->shape()->Get(2));
  ASSERT_EQ(s2[0], 0);
  ASSERT_EQ(s2[1], 1);

  const auto s3 = detail::SymInt(tensor->shape()->Get(3));
  ASSERT_EQ(s3[0], -1);
  ASSERT_EQ(s3[1], 2);
}

// Tests whether `FakeTensor` provides a transparent view of the serialized fake
// tensor.
TEST(ViewTest, FromFakeTensor) {
  auto builder = flatbuffers::FlatBufferBuilder();

  const auto shape = builder.CreateVectorOfStructs<SymInt>({
      SymInt(std::to_array<int64_t>({4096, 0})),
      SymInt(std::to_array<int64_t>({0, 1})),
  });
  const auto root = CreateFakeTensor(builder, ScalarType::bfloat16, shape);
  builder.Finish(root);

  const auto tensor = detail::FakeTensor(
      flatbuffers::GetRoot<FakeTensor>(builder.GetBufferPointer()));
  ASSERT_EQ(tensor.dtype(), ScalarType::bfloat16);
  ASSERT_EQ(tensor.shape().size(), 2);
  ASSERT_EQ(tensor.shape()[0][0], 4096);
  ASSERT_EQ(tensor.shape()[0][1], 0);
  ASSERT_EQ(tensor.shape()[1][0], 0);
  ASSERT_EQ(tensor.shape()[1][1], 1);
}

// Tests whether `Vector` provides a transparent view of a FlatBuffers vector
// whose elements are structures.
TEST(ViewTest, FromVectorOfStructs) {
  auto builder = flatbuffers::FlatBufferBuilder();

  const auto root =
      CreateFakeTensor(builder, ScalarType::float32,
                       builder.CreateVectorOfStructs<SymInt>({
                           SymInt(std::to_array<int64_t>({0, 0})),
                           SymInt(std::to_array<int64_t>({4096, 0})),
                           SymInt(std::to_array<int64_t>({0, 1})),
                           SymInt(std::to_array<int64_t>({-1, 2})),
                       }));
  builder.Finish(root);

  const auto tensor =
      flatbuffers::GetRoot<FakeTensor>(builder.GetBufferPointer());

  const auto shape = detail::Vector<detail::SymInt>(tensor->shape());
  ASSERT_EQ(shape.size(), 4);
  ASSERT_EQ(shape[0][0], 0);
  ASSERT_EQ(shape[0][1], 0);
  ASSERT_EQ(shape[1][0], 4096);
  ASSERT_EQ(shape[1][1], 0);
  ASSERT_EQ(shape[2][0], 0);
  ASSERT_EQ(shape[2][1], 1);
  ASSERT_EQ(shape[3][0], -1);
  ASSERT_EQ(shape[3][1], 2);
}

// Tests whether `Vector` provides a transparent view of a FlatBuffers vector
// whose elements are tables.
TEST(ViewTest, FromVectorOfTables) {
  auto builder = flatbuffers::FlatBufferBuilder();

  const auto input =
      CreateFakeTensor(builder, ScalarType::bfloat16,
                       builder.CreateVectorOfStructs<SymInt>({
                           SymInt(std::to_array<int64_t>({0, 1})),
                           SymInt(std::to_array<int64_t>({4096, 0})),
                       }));
  const auto other =
      CreateFakeTensor(builder, ScalarType::bfloat16,
                       builder.CreateVectorOfStructs<SymInt>({
                           SymInt(std::to_array<int64_t>({4096, 0})),
                           SymInt(std::to_array<int64_t>({1024, 0})),
                       }));
  const auto out =
      CreateFakeTensor(builder, ScalarType::bfloat16,
                       builder.CreateVectorOfStructs<SymInt>({
                           SymInt(std::to_array<int64_t>({0, 1})),
                           SymInt(std::to_array<int64_t>({1024, 0})),
                       }));

  const auto node = CreateNode(builder, Operator::matmul,
                               builder.CreateVector({input, other}),
                               builder.CreateVector({out}));
  const auto root = CreateGraph(builder, builder.CreateVector({node}));
  builder.Finish(root);

  auto verifier =
      flatbuffers::Verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(VerifyGraphBuffer(verifier));

  const auto graph = GetGraph(builder.GetBufferPointer());
  ASSERT_EQ(graph->nodes()->size(), 1);

  const auto args =
      detail::Vector<detail::FakeTensor>(graph->nodes()->Get(0)->args());
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

  const auto meta =
      detail::Vector<detail::FakeTensor>(graph->nodes()->Get(0)->meta());
  ASSERT_EQ(meta.size(), 1);
  ASSERT_EQ(meta[0].dtype(), ScalarType::bfloat16);
  ASSERT_EQ(meta[0].shape()[0][0], 0);
  ASSERT_EQ(meta[0].shape()[0][1], 1);
  ASSERT_EQ(meta[0].shape()[1][0], 1024);
  ASSERT_EQ(meta[0].shape()[1][1], 0);
}

}  // namespace
}  // namespace flatflow
