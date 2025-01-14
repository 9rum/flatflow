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

#ifndef FLATFLOW_OPS_OPS_H_
#define FLATFLOW_OPS_OPS_H_

#include <cstdint>
#include <execution>
#include <functional>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "flatbuffers/base.h"
#include "flatbuffers/vector.h"

#include "flatflow/ops/graph_generated.h"
#include "flatflow/ops/node_generated.h"
#include "flatflow/ops/operator_generated.h"

namespace flatflow {

// flatflow::SymFLOPs
//
// `flatflow::SymFLOPs` is a symbolic expression for the absolute number
// of floating point operations (FLOPs). These expressions are created by
// measuring the absolute number of FLOPs or multiply-accumulates (MACs)
// for FMA instructions required by each pair of operator and symbolic shapes,
// for a given size such as sequence length.
class SymFLOPs {
 public:
  using value_type = int64_t;

  // Constructors and assignment operators
  //
  // In addition to the default constructor below, a `flatflow::SymFLOPs`
  // supports copy and move constructors and assignment operators.
  explicit SymFLOPs(value_type coef0 = 0, value_type coef1 = 0,
                    value_type coef2 = 0, value_type coef3 = 0)
      : coef0_(coef0), coef1_(coef1), coef2_(coef2), coef3_(coef3) {}

  explicit SymFLOPs(const SymFLOPs &other) = default;

  SymFLOPs &operator=(const SymFLOPs &other) = default;

  explicit SymFLOPs(SymFLOPs &&other) = default;

  SymFLOPs &operator=(SymFLOPs &&other) = default;

  // SymFLOPs::operator+()
  //
  // Combines the two given symbolic expressions for FLOPs in coefficient-wise.
  SymFLOPs &operator+(const SymFLOPs &other) const noexcept {
    auto expr = SymFLOPs(coef0_ + other.coef0(), coef1_ + other.coef1(),
                         coef2_ + other.coef2(), coef3_ + other.coef3());
    return expr;
  }

  value_type coef0() const noexcept { return coef0_; }

  value_type coef1() const noexcept { return coef1_; }

  value_type coef2() const noexcept { return coef2_; }

  value_type coef3() const noexcept { return coef3_; }

 protected:
  value_type coef0_;  // constant
  value_type coef1_;  // linear
  value_type coef2_;  // quadratic
  value_type coef3_;  // cubic
};

// flatflow::symbolic_trace_impl<>()
//
// This is a base template for implementing a symbolic transformation for the
// corresponding operator; calling this means that the program is ill-formed
// and should fail to compile.
template <Operator>
SymFLOPs symbolic_trace_impl(
    const flatbuffers::Vector<flatbuffers::Offset<TensorMetadata>> *args,
    const TensorMetadata *meta) {
  static_assert(std::false_type::value);
}

// flatflow::OperatorRegistry
//
// A `flatflow::OperatorRegistry` holds the key information to identify
// operators and generate optimized computation plans. It has an operator table
// in a form of `absl::flat_hash_map<Operator, std::_Bind<...>>`, where each
// value contains a transformation from the corresponding operator and symbolic
// shapes to a symbolic FLOPs expression for a given size.
//
// NOTE: To register a new operator, please follow the instructions below:
//
// * First, declare a new operator as an enumerator value of `Operator` in the
//   FlatBuffers schema - i.e., `flatflow/ops/operator.fbs`. The enumerator
//   values are recommended to be sorted in order of their original operator
//   names (not the enumerator values themselves) for searching convenience.
// * Second, generate codes from the updated schema by running `make generate`
//   in the root directory of this source tree. This will create
//   `flatflow/ops/operator_generated.h`, etc.
// * Third, add template specializations of `symbolic_trace_impl` and
//   `OperatorRegistry::RegisterOperator` for the new operator; please refer to
//   the implementations below for supported operators. Note that omitting this
//   step will call their base template, which raises an assertion failure at
//   compile time.
// * Finally, register the new operator to the operator table by calling its
//   specialized `OperatorRegistry::RegisterOperator` in the constructor below.
class OperatorRegistry {
 public:
  using key_type = Operator;
  using mapped_type =
      decltype(std::bind(symbolic_trace_impl<Operator::MM>,
                         std::placeholders::_1, std::placeholders::_2));

  // Constructors and assignment operators
  //
  // The constructor below creates the operator table and registers
  // symbolic transformations for all supported operators to it.
  //
  // CAVEATS
  //
  // We provide only a handful of ATen operator set for now. The operator set
  // is under development and more operators will be added in the future. For
  // expanding the operator set, please refer to the note above.
  explicit OperatorRegistry() {
    constexpr auto kOpTableSpace =
        sizeof(EnumValuesOperator()) / sizeof(Operator);
    table_.reserve(kOpTableSpace);

    RegisterOperator<Operator::ARANGE>();
    RegisterOperator<Operator::EMBEDDING>();
    RegisterOperator<Operator::MM>();
    RegisterOperator<Operator::T>();
    RegisterOperator<Operator::TRANSPOSE_INT>();
    RegisterOperator<Operator::UNSQUEEZE>();
    RegisterOperator<Operator::VIEW>();
  }

  explicit OperatorRegistry(const OperatorRegistry &other) = default;

  OperatorRegistry &operator=(const OperatorRegistry &other) = default;

  explicit OperatorRegistry(OperatorRegistry &&other) = default;

  OperatorRegistry &operator=(OperatorRegistry &&other) = default;

  // OperatorRegistry::RegisterOperator<>()
  //
  // This is a base template for registering operators; calling this means that
  // the program is ill-formed and should fail to compile.
  template <Operator>
  void RegisterOperator() {
    static_assert(std::false_type::value);
  }

  // OperatorRegistry::DeregisterOperator()
  //
  // Excludes `op` from the operator table.
  void DeregisterOperator(key_type op) {
    // TODO: Check the number of elements removed.
    table_.erase(op);
  }

  // OperatorRegistry::Dispatch()
  //
  // Executes the symbolic transformation corresponding to the given operator.
  SymFLOPs Dispatch(
      key_type op,
      const flatbuffers::Vector<flatbuffers::Offset<TensorMetadata>> *args,
      const TensorMetadata *meta) const {
    CHECK(table_.contains(op));
    return table_.at(op)(args, meta);
  }

 protected:
  absl::flat_hash_map<key_type, mapped_type> table_;
};

////////////////////////////////////////////////////////////////////////////////
///////////////////// OPERATOR REGISTRATIONS SECTION BEGIN /////////////////////
////////////////////////////////////////////////////////////////////////////////

// flatflow::symbolic_trace_impl<ARANGE>()
//
// Implements a symbolic transformation for `arange`.
//
// func: arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None,
//              Device? device=None, bool? pin_memory=None) -> Tensor
template <>
SymFLOPs symbolic_trace_impl<Operator::ARANGE>(
    const flatbuffers::Vector<flatbuffers::Offset<TensorMetadata>> *args,
    const TensorMetadata *meta) {
  // arange returns a tensor, so it has zero FLOPs.
  return SymFLOPs(0, 0, 0, 0);
}

// OperatorRegistry::RegisterOperator<ARANGE>()
//
// Registers `arange` to the operator table.
template <>
void OperatorRegistry::RegisterOperator<Operator::ARANGE>() {
  table_.insert(
      std::make_pair(Operator::ARANGE,
                     std::bind(symbolic_trace_impl<Operator::ARANGE>,
                               std::placeholders::_1, std::placeholders::_2)));
}

// flatflow::symbolic_trace_impl<EMBEDDING>()
//
// Implements a symbolic transformation for `embedding`.
//
// func: embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1,
//                 bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
template <>
SymFLOPs symbolic_trace_impl<Operator::EMBEDDING>(
    const flatbuffers::Vector<flatbuffers::Offset<TensorMetadata>> *args,
    const TensorMetadata *meta) {
  // embedding is a dictionary lookup, so technically it has zero FLOPs.
  return SymFLOPs(0, 0, 0, 0);
}

// OperatorRegistry::RegisterOperator<EMBEDDING>()
//
// Registers `embedding` to the operator table.
template <>
void OperatorRegistry::RegisterOperator<Operator::EMBEDDING>() {
  table_.insert(
      std::make_pair(Operator::EMBEDDING,
                     std::bind(symbolic_trace_impl<Operator::EMBEDDING>,
                               std::placeholders::_1, std::placeholders::_2)));
}

// flatflow::symbolic_trace_impl<MM>()
//
// Implements a symbolic transformation for `mm`.
//
// func: mm(Tensor self, Tensor mat2) -> Tensor
template <>
SymFLOPs symbolic_trace_impl<Operator::MM>(
    const flatbuffers::Vector<flatbuffers::Offset<TensorMetadata>> *args,
    const TensorMetadata *meta) {
  CHECK_NE(args, nullptr);
  CHECK_EQ(args->size(), 2);

  CHECK_NE(args->Get(0), nullptr);
  auto shape0 = args->Get(0)->shape();
  CHECK_NE(shape0, nullptr);
  CHECK_EQ(shape0->size(), 2);

  CHECK_NE(args->Get(1), nullptr);
  auto shape1 = args->Get(1)->shape();
  CHECK_NE(shape1, nullptr);
  CHECK_EQ(shape1->size(), 2);

  // mm performs a matrix multiplication of the matrices `self` and `mat2`.
  // If `self` is a (n x m) tensor and `mat2` is a (m x p) tensor, then it
  // produces a (n x p) tensor with n x m x p MACs, i.e., 2 x n x m x p FLOPs.
  auto n = shape0->Get(0);
  CHECK_NE(n, nullptr);
  auto m = shape0->Get(1);
  CHECK_NE(m, nullptr);
  CHECK_NE(shape1->Get(0), nullptr);

  // The last dimension of `self` and the first dimension of `mat2` must be
  // symbolically identical.
  CHECK_EQ(m->coef0(), shape1->Get(0)->coef0());
  CHECK_EQ(m->coef1(), shape1->Get(0)->coef1());

  auto p = shape1->Get(1);
  CHECK_NE(p, nullptr);

  const auto coef0 = n->coef0() * m->coef0() * p->coef0();
  const auto coef1 =
      (n->coef1() * m->coef0() + n->coef0() * m->coef1()) * p->coef0() +
      n->coef0() * m->coef0() * p->coef1();
  const auto coef2 =
      (n->coef0() * m->coef1() + n->coef1() + m->coef0()) * p->coef1() +
      n->coef1() * m->coef1() * p->coef0();
  const auto coef3 = n->coef1() * m->coef1() * p->coef1();

  return SymFLOPs(coef0 << 1, coef1 << 1, coef2 << 1, coef3 << 1);
}

// OperatorRegistry::RegisterOperator<MM>()
//
// Registers `mm` to the operator table.
template <>
void OperatorRegistry::RegisterOperator<Operator::MM>() {
  // TODO: Check if the insertion took place.
  table_.insert(std::make_pair(
      Operator::MM, std::bind(symbolic_trace_impl<Operator::MM>,
                              std::placeholders::_1, std::placeholders::_2)));
}

// flatflow::symbolic_trace_impl<T>()
//
// Implements a symbolic transformation for `t`.
//
// func: t(Tensor(a) self) -> Tensor(a)
template <>
SymFLOPs symbolic_trace_impl<Operator::T>(
    const flatbuffers::Vector<flatbuffers::Offset<TensorMetadata>> *args,
    const TensorMetadata *meta) {
  // t is a dimension swap, so technically it has zero FLOPs.
  return SymFLOPs(0, 0, 0, 0);
}

// OperatorRegistry::RegisterOperator<T>()
//
// Registers `t` to the operator table.
template <>
void OperatorRegistry::RegisterOperator<Operator::T>() {
  table_.insert(std::make_pair(
      Operator::T, std::bind(symbolic_trace_impl<Operator::T>,
                             std::placeholders::_1, std::placeholders::_2)));
}

// flatflow::symbolic_trace_impl<TRANSPOSE_INT>()
//
// Implements a symbolic transformation for `transpose.int`.
//
// func: transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
template <>
SymFLOPs symbolic_trace_impl<Operator::TRANSPOSE_INT>(
    const flatbuffers::Vector<flatbuffers::Offset<TensorMetadata>> *args,
    const TensorMetadata *meta) {
  // transpose.int is a dimension swap, so technically it has zero FLOPs.
  return SymFLOPs(0, 0, 0, 0);
}

// OperatorRegistry::RegisterOperator<TRANSPOSE_INT>()
//
// Registers `transpose.int` to the operator table.
template <>
void OperatorRegistry::RegisterOperator<Operator::TRANSPOSE_INT>() {
  table_.insert(
      std::make_pair(Operator::TRANSPOSE_INT,
                     std::bind(symbolic_trace_impl<Operator::TRANSPOSE_INT>,
                               std::placeholders::_1, std::placeholders::_2)));
}

// flatflow::symbolic_trace_impl<UNSQUEEZE>()
//
// Implements a symbolic transformation for `unsqueeze`.
//
// func: unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
template <>
SymFLOPs symbolic_trace_impl<Operator::UNSQUEEZE>(
    const flatbuffers::Vector<flatbuffers::Offset<TensorMetadata>> *args,
    const TensorMetadata *meta) {
  // unsqueeze inserts a singleton dimension at the specified position,
  // so it has zero FLOPs.
  return SymFLOPs(0, 0, 0, 0);
}

// OperatorRegistry::RegisterOperator<UNSQUEEZE>()
//
// Registers `unsqueeze` to the operator table.
template <>
void OperatorRegistry::RegisterOperator<Operator::UNSQUEEZE>() {
  table_.insert(
      std::make_pair(Operator::UNSQUEEZE,
                     std::bind(symbolic_trace_impl<Operator::UNSQUEEZE>,
                               std::placeholders::_1, std::placeholders::_2)));
}

// flatflow::symbolic_trace_impl<VIEW>()
//
// Implements a symbolic transformation for `view`.
//
// func: view(Tensor(a) self, SymInt[] size) -> Tensor(a)
template <>
SymFLOPs symbolic_trace_impl<Operator::VIEW>(
    const flatbuffers::Vector<flatbuffers::Offset<TensorMetadata>> *args,
    const TensorMetadata *meta) {
  // view is a tensor view operation, so technically it has zero FLOPs.
  // See https://pytorch.org/docs/stable/tensor_view.html.
  return SymFLOPs(0, 0, 0, 0);
}

// OperatorRegistry::RegisterOperator<VIEW>()
//
// Registers `view` to the operator table.
template <>
void OperatorRegistry::RegisterOperator<Operator::VIEW>() {
  table_.insert(std::make_pair(
      Operator::VIEW, std::bind(symbolic_trace_impl<Operator::VIEW>,
                                std::placeholders::_1, std::placeholders::_2)));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////// OPERATOR REGISTRATIONS SECTION END //////////////////////
////////////////////////////////////////////////////////////////////////////////

// flatflow::PolyEval()
//
// Based on Horner's rule, evaluates a given polynomial of degree three with
// only three multiplications and three additions, applying Horner's method.
//
// This is optimal, since there are polynomials of degree three that cannot be
// evaluated with fewer arithmetic operations.
// See https://doi.org/10.1070%2Frm1966v021n01abeh004147.
constexpr SymFLOPs::value_type PolyEval(SymFLOPs::value_type size,
                                        SymFLOPs::value_type coef0,
                                        SymFLOPs::value_type coef1,
                                        SymFLOPs::value_type coef2,
                                        SymFLOPs::value_type coef3) noexcept {
  return coef0 + size * (coef1 + size * (coef2 + size * coef3));
}

// flatflow::symbolic_trace()
//
// Generates a forwarding call wrapper for a function that evaluates the FLOPs
// of the graph for a given size upon forward call.
decltype(auto) symbolic_trace(const Graph *graph) {
  const auto registry = OperatorRegistry();

  CHECK_NE(graph, nullptr);

  auto nodes = graph->nodes();
  CHECK_NE(nodes, nullptr);

  auto exprs = std::vector<SymFLOPs>(nodes->size());

  // clang-format off
  #pragma omp parallel for
  for (flatbuffers::uoffset_t index = 0; index < nodes->size(); ++index) {
    auto node = nodes->Get(index);
    CHECK_NE(node, nullptr);
    exprs[index] = registry.Dispatch(node->op(), node->args(), node->meta());
  }
  // clang-format on

  const auto expr = std::reduce(std::execution::par, exprs.cbegin(),
                                exprs.cend(), SymFLOPs());
  return std::bind(PolyEval, std::placeholders::_1, expr.coef0(), expr.coef1(),
                   expr.coef2(), expr.coef3());
}

}  // namespace flatflow

#endif  // FLATFLOW_OPS_OPS_H_
