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

#ifndef FLATFLOW_OPS_ADAPTORS_H_
#define FLATFLOW_OPS_ADAPTORS_H_

#include <omp.h>

#include <array>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/base.h"

#include "flatflow/ops/graph_generated.h"
#include "flatflow/ops/node_generated.h"
#include "flatflow/ops/operator_generated.h"

namespace flatflow {

// flatflow::SymIntAdaptor
//
// `flatflow::SymIntAdaptor` is an adaptor that encapsulates access to a value
// within the symbolic shape of a tensor.
class SymIntAdaptor {
 public:
  using value_type = typename std::array<int64_t, 2>::value_type;
  using size_type = typename std::array<int64_t, 2>::size_type;

  SymIntAdaptor() {}

  SymIntAdaptor(const SymInt *s) {
    CHECK_NE(s, nullptr);

    auto data = s->data();
    CHECK_NE(data, nullptr);

    data_[0] = data->Get(0);
    data_[1] = data->Get(1);
  }

  SymIntAdaptor(const SymIntAdaptor &other) = default;

  SymIntAdaptor &operator=(const SymIntAdaptor &other) = default;

  SymIntAdaptor(SymIntAdaptor &&other) = default;

  SymIntAdaptor &operator=(SymIntAdaptor &&other) = default;

  constexpr size_type size() const noexcept { return data_.size(); }

  std::array<int64_t, 2> &data() { return data_; }

  const std::array<int64_t, 2> &data() const { return data_; }

  constexpr value_type &operator[](size_type index) noexcept {
    return data_[index];
  }

  constexpr value_type operator[](size_type index) const noexcept {
    return data_[index];
  }

 protected:
  std::array<int64_t, 2> data_;
};

// flatflow::TensorMetadataAdaptor
//
// `flatflow::TensorMetadataAdaptor` is an adaptor that encapsulates access to
// pertinent information about a tensor within a PyTorch program.
class TensorMetadataAdaptor {
 public:
  TensorMetadataAdaptor() {}

  TensorMetadataAdaptor(const TensorMetadata *meta) {
    CHECK_NE(meta, nullptr);

    auto shape = meta->shape();
    CHECK_NE(shape, nullptr);

    shape_.reserve(shape->size());

    for (flatbuffers::uoffset_t index = 0; index < shape->size(); ++index) {
      shape_.emplace_back(shape->Get(index));
    }
  }

  TensorMetadataAdaptor(const TensorMetadataAdaptor &other) = default;

  TensorMetadataAdaptor &operator=(const TensorMetadataAdaptor &other) =
      default;

  TensorMetadataAdaptor(TensorMetadataAdaptor &&other) = default;

  TensorMetadataAdaptor &operator=(TensorMetadataAdaptor &&other) = default;

  std::vector<SymIntAdaptor> &shape() { return shape_; }

  const std::vector<SymIntAdaptor> &shape() const { return shape_; }

 protected:
  std::vector<SymIntAdaptor> shape_;
};

// flatflow::NodeAdaptor
//
// `flatflow::NodeAdaptor` is an adaptor that encapsulates access to each node
// in the computational graph.
class NodeAdaptor {
 public:
  NodeAdaptor() {}

  NodeAdaptor(const Node *node) {
    CHECK_NE(node, nullptr);

    auto args = node->args();
    CHECK_NE(args, nullptr);

    args_.reserve(args->size());

    target_ = node->target();
    for (flatbuffers::uoffset_t index = 0; index < args->size(); ++index) {
      args_.emplace_back(args->Get(index));
    }
    meta_ = TensorMetadataAdaptor(node->meta());
  }

  NodeAdaptor(const NodeAdaptor &other) = default;

  NodeAdaptor &operator=(const NodeAdaptor &other) = default;

  NodeAdaptor(NodeAdaptor &&other) = default;

  NodeAdaptor &operator=(NodeAdaptor &&other) = default;

  Operator &target() { return target_; }

  Operator target() const { return target_; }

  std::vector<TensorMetadataAdaptor> &args() { return args_; }

  const std::vector<TensorMetadataAdaptor> &args() const { return args_; }

  TensorMetadataAdaptor &meta() { return meta_; }

  const TensorMetadataAdaptor &meta() const { return meta_; }

 protected:
  Operator target_;
  std::vector<TensorMetadataAdaptor> args_;
  TensorMetadataAdaptor meta_;
};

// flatflow::GraphAdaptor
//
// `flatflow::GraphAdaptor` is an adaptor that encapsulates access to the given
// computational graph.
class GraphAdaptor {
 public:
  using value_type = typename std::vector<NodeAdaptor>::value_type;
  using size_type = typename std::vector<NodeAdaptor>::size_type;

  GraphAdaptor() {}

  GraphAdaptor(const Graph *graph) {
    CHECK_NE(graph, nullptr);

    auto nodes = graph->nodes();
    CHECK_NE(nodes, nullptr);

    const auto now = omp_get_wtime();

    nodes_.resize(nodes->size());

    // clang-format off
    #pragma omp parallel for
    for (flatbuffers::uoffset_t index = 0; index < nodes->size(); ++index) {
      nodes_[index] = NodeAdaptor(nodes->Get(index));
    }

    LOG(INFO) << absl::StrFormat("Decoding a graph with %u nodes took %fs", nodes->size(), omp_get_wtime() - now);
    // clang-format on
  }

  GraphAdaptor(const GraphAdaptor &other) = default;

  GraphAdaptor &operator=(const GraphAdaptor &other) = default;

  GraphAdaptor(GraphAdaptor &&other) = default;

  GraphAdaptor &operator=(GraphAdaptor &&other) = default;

  size_type size() const noexcept { return nodes_.size(); }

  std::vector<NodeAdaptor> &nodes() { return nodes_; }

  const std::vector<NodeAdaptor> &nodes() const { return nodes_; }

  value_type &operator[](size_type index) { return nodes_[index]; }

  const value_type &operator[](size_type index) const { return nodes_[index]; }

 protected:
  std::vector<NodeAdaptor> nodes_;
};

}  // namespace flatflow

#endif  // FLATFLOW_OPS_ADAPTORS_H_
