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

#include <array>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "flatbuffers/base.h"

#include "flatflow/ops/graph_generated.h"
#include "flatflow/ops/node_generated.h"
#include "flatflow/ops/operator_generated.h"

namespace flatflow {

// flatflow::SymIntAdaptor
//
// `flatflow::SymIntAdaptor` is an adaptor that encapsulates access to a value
// within the symbolic shape of a tensor.
struct SymIntAdaptor {
  SymIntAdaptor() {}

  SymIntAdaptor(const SymInt *s) {
    CHECK_NE(s, nullptr);
    CHECK_NE(s->data(), nullptr);
    data[0] = s->data()->Get(0);
    data[1] = s->data()->Get(1);
  }

  std::array<int64_t, 2> data;
};

// flatflow::TensorMetadataAdaptor
//
// `flatflow::TensorMetadataAdaptor` is an adaptor that encapsulates access to
// pertinent information about a tensor within a PyTorch program.
struct TensorMetadataAdaptor {
  TensorMetadataAdaptor() {}

  TensorMetadataAdaptor(const TensorMetadata *meta) {
    CHECK_NE(meta, nullptr);
    CHECK_NE(meta->shape(), nullptr);
    shape.reserve(meta->shape()->size());

    for (flatbuffers::uoffset_t index = 0; index < meta->shape()->size();
         ++index) {
      shape.emplace_back(meta->shape()->Get(index));
    }
  }

  std::vector<SymIntAdaptor> shape;
};

// flatflow::NodeAdaptor
//
// `flatflow::NodeAdaptor` is an adaptor that encapsulates access to each node
// in the computational graph.
struct NodeAdaptor {
  NodeAdaptor() {}

  NodeAdaptor(const Node *node) {
    CHECK_NE(node, nullptr);
    CHECK_NE(node->args(), nullptr);
    args.reserve(node->args()->size());

    target = node->target();
    for (flatbuffers::uoffset_t index = 0; index < node->args()->size();
         ++index) {
      args.emplace_back(node->args()->Get(index));
    }
    meta = TensorMetadataAdaptor(node->meta());
  }

  Operator target;
  std::vector<TensorMetadataAdaptor> args;
  TensorMetadataAdaptor meta;
};

// flatflow::GraphAdaptor
//
// `flatflow::GraphAdaptor` is an adaptor that encapsulates access to the given
// computational graph.
struct GraphAdaptor {
  GraphAdaptor() {}

  GraphAdaptor(const Graph *graph) {
    CHECK_NE(graph, nullptr);
    CHECK_NE(graph->nodes(), nullptr);
    nodes.reserve(graph->nodes()->size());

    for (flatbuffers::uoffset_t index = 0; index < graph->nodes()->size();
         ++index) {
      nodes.emplace_back(graph->nodes()->Get(index));
    }
  }

  std::vector<NodeAdaptor> nodes;
};

}  // namespace flatflow

#endif  // FLATFLOW_OPS_ADAPTORS_H_
