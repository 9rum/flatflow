// SPDX-License-Identifier: Apache-2.0

#ifndef FLATFLOW_OPS_GRAPH_H_
#define FLATFLOW_OPS_GRAPH_H_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/die_if_null.h"

#include "flatflow/ops/graph_generated.h"
#include "flatflow/ops/operator_generated.h"
#include "flatflow/ops/scalar_type_generated.h"
#include "flatflow/types.h"

namespace flatflow {

// `SymInt` records a value within the symbolic shape of a tensor.
struct SymInt {
  using value_type =
      make_array_t<decltype(internal::SymInt().data())>::value_type;
  using size_type =
      make_array_t<decltype(internal::SymInt().data())>::size_type;

  template <typename... Args>
  constexpr SymInt(Args... args) noexcept : data{args...} {}

  constexpr SymInt(const SymInt &) noexcept = default;

  constexpr SymInt &operator=(const SymInt &) noexcept = default;

  constexpr SymInt(SymInt &&) noexcept = default;

  constexpr SymInt &operator=(SymInt &&) noexcept = default;

  SymInt(const internal::SymInt *s) noexcept {
    std::ranges::copy(*ABSL_DIE_IF_NULL(s)->data(), data.begin());
  }

  constexpr value_type operator[](size_type pos) const noexcept {
    return data[pos];
  }

  constexpr bool operator==(const SymInt &) const noexcept = default;

  make_array_t<decltype(internal::SymInt().data())> data;
};

// `TensorMetadata` is a structure containing pertinent information about a
// tensor within a PyTorch program.
struct TensorMetadata {
  TensorMetadata() noexcept = default;

  TensorMetadata(ScalarType dtype,
                 const absl::InlinedVector<SymInt, 4> &shape) noexcept
      : dtype(dtype), shape(shape) {}

  TensorMetadata(ScalarType dtype,
                 absl::InlinedVector<SymInt, 4> &&shape) noexcept
      : dtype(dtype), shape(std::move(shape)) {}

  TensorMetadata(const TensorMetadata &) noexcept = default;

  TensorMetadata &operator=(const TensorMetadata &) noexcept = default;

  TensorMetadata(TensorMetadata &&) noexcept = default;

  TensorMetadata &operator=(TensorMetadata &&) noexcept = default;

  TensorMetadata(const internal::TensorMetadata *meta) noexcept
      : dtype(ABSL_DIE_IF_NULL(meta)->dtype()),
        shape(meta->shape()->begin(), meta->shape()->end()) {}

  bool operator==(const TensorMetadata &) const noexcept = default;

  ScalarType dtype;
  // Tensors rarely have more than four dimensions so the shape is stored inline
  // without heap allocation in most cases.
  absl::InlinedVector<SymInt, 4> shape;
};

// `Node` is a data structure that represents individual operations in the
// computational graph. Each node contains an opcode identifying operators and
// the input/output shapes of the operator. Unlike `torch.fx.Node`, this
// excludes operations other than callsites to ATen operators; i.e., operations
// whose `op` property are not `call_function`.
struct Node {
  Node() noexcept = default;

  Node(Operator target, const std::vector<TensorMetadata> &args,
       const TensorMetadata &meta) noexcept
      : target(target), args(args), meta(meta) {}

  Node(Operator target, const std::vector<TensorMetadata> &args,
       TensorMetadata &&meta) noexcept
      : target(target), args(args), meta(std::move(meta)) {}

  Node(Operator target, std::vector<TensorMetadata> &&args,
       const TensorMetadata &meta) noexcept
      : target(target), args(std::move(args)), meta(meta) {}

  Node(Operator target, std::vector<TensorMetadata> &&args,
       TensorMetadata &&meta) noexcept
      : target(target), args(std::move(args)), meta(std::move(meta)) {}

  Node(const Node &) noexcept = default;

  Node &operator=(const Node &) noexcept = default;

  Node(Node &&) noexcept = default;

  Node &operator=(Node &&) noexcept = default;

  Node(const internal::Node *node) noexcept
      : target(ABSL_DIE_IF_NULL(node)->target()),
        args(node->args()->begin(), node->args()->end()),
        meta(node->meta()) {}

  bool operator==(const Node &) const noexcept = default;

  Operator target;
  std::vector<TensorMetadata> args;
  TensorMetadata meta;
};

// `Graph` is the main data structure for tracing a given PyTorch program at the
// intermediate representation level. It consists of a series of `Node`s, each
// representing callsites such as opcode and the input/output shapes of the
// corresponding operator.
struct Graph {
  constexpr Graph() noexcept = default;

  constexpr Graph(const std::vector<Node> &nodes) noexcept : nodes(nodes) {}

  constexpr Graph(std::vector<Node> &&nodes) noexcept
      : nodes(std::move(nodes)) {}

  constexpr Graph(const Graph &) noexcept = default;

  constexpr Graph &operator=(const Graph &) noexcept = default;

  constexpr Graph(Graph &&) noexcept = default;

  constexpr Graph &operator=(Graph &&) noexcept = default;

  Graph(const internal::Graph *graph) noexcept
      : nodes(ABSL_DIE_IF_NULL(graph)->nodes()->size()) {
#pragma omp parallel for
    for (remove_cvptr_t<decltype(graph->nodes())>::size_type index = 0;
         index < graph->nodes()->size(); ++index) {
      std::destroy_at(&nodes[index]);
      std::construct_at(&nodes[index], graph->nodes()->Get(index));
    }
  }

  constexpr bool operator==(const Graph &) const noexcept = default;

  std::vector<Node> nodes;
};

}  // namespace flatflow

#endif  // FLATFLOW_OPS_GRAPH_H_
