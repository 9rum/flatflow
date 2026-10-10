// SPDX-License-Identifier: Apache-2.0

#ifndef FLATFLOW_OPS_GRAPH_H_
#define FLATFLOW_OPS_GRAPH_H_

#include <algorithm>
#include <utility>
#include <vector>

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
  constexpr TensorMetadata() noexcept = default;

  constexpr TensorMetadata(ScalarType dtype,
                           const std::vector<SymInt> &shape) noexcept
      : dtype(dtype), shape(shape) {}

  constexpr TensorMetadata(ScalarType dtype,
                           std::vector<SymInt> &&shape) noexcept
      : dtype(dtype), shape(std::move(shape)) {}

  constexpr TensorMetadata(const TensorMetadata &) noexcept = default;

  constexpr TensorMetadata &operator=(const TensorMetadata &) noexcept =
      default;

  constexpr TensorMetadata(TensorMetadata &&) noexcept = default;

  constexpr TensorMetadata &operator=(TensorMetadata &&) noexcept = default;

  TensorMetadata(const internal::TensorMetadata *meta) noexcept
      : dtype(ABSL_DIE_IF_NULL(meta)->dtype()),
        shape(meta->shape()->begin(), meta->shape()->end()) {}

  constexpr bool operator==(const TensorMetadata &) const noexcept = default;

  ScalarType dtype;
  std::vector<SymInt> shape;
};

// `Node` is a data structure that represents individual operations in the
// computational graph. Each node contains an opcode identifying operators and
// the input/output shapes of the operator. Unlike `torch.fx.Node`, this
// excludes operations other than callsites to ATen operators; i.e., operations
// whose `op` property are not `call_function`.
struct Node {
  constexpr Node() noexcept = default;

  constexpr Node(Operator target, const std::vector<TensorMetadata> &args,
                 const TensorMetadata &meta) noexcept
      : target(target), args(args), meta(meta) {}

  constexpr Node(Operator target, const std::vector<TensorMetadata> &args,
                 TensorMetadata &&meta) noexcept
      : target(target), args(args), meta(std::move(meta)) {}

  constexpr Node(Operator target, std::vector<TensorMetadata> &&args,
                 const TensorMetadata &meta) noexcept
      : target(target), args(std::move(args)), meta(meta) {}

  constexpr Node(Operator target, std::vector<TensorMetadata> &&args,
                 TensorMetadata &&meta) noexcept
      : target(target), args(std::move(args)), meta(std::move(meta)) {}

  constexpr Node(const Node &) noexcept = default;

  constexpr Node &operator=(const Node &) noexcept = default;

  constexpr Node(Node &&) noexcept = default;

  constexpr Node &operator=(Node &&) noexcept = default;

  Node(const internal::Node *node) noexcept
      : target(ABSL_DIE_IF_NULL(node)->target()),
        args(node->args()->begin(), node->args()->end()),
        meta(node->meta()) {}

  constexpr bool operator==(const Node &) const noexcept = default;

  Operator target;
  std::vector<TensorMetadata> args;
  TensorMetadata meta;
};

}  // namespace flatflow

#endif  // FLATFLOW_OPS_GRAPH_H_
