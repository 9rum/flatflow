# Copyright 2025 The FlatFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections.abc import Mapping, Sequence
from typing import Union

import flatbuffers
import torch
import torch.fx
from torch._library.custom_ops import CustomOpDef
from torch._ops import OpOverload, OpOverloadPacket

from flatflow.ops.graph_generated import (
    GraphAddNodes,
    GraphEnd,
    GraphStart,
    GraphStartNodesVector,
)
from flatflow.ops.node_generated import (
    CreateSymInt,
    NodeAddArgs,
    NodeAddMeta,
    NodeAddTarget,
    NodeEnd,
    NodeStart,
    NodeStartArgsVector,
    TensorMetadataAddShape,
    TensorMetadataEnd,
    TensorMetadataStart,
    TensorMetadataStartShapeVector,
)
from flatflow.ops.operator_generated import Operator

aten = torch._ops.ops.aten  # type: ignore[has-type]

__all__ = ["serialize"]

_OPS_TABLE: Mapping[Union[OpOverload, OpOverloadPacket, CustomOpDef], int] = {
    aten._softmax: Operator._SOFTMAX,
    aten._to_copy: Operator._TO_COPY,
    aten._unsafe_view: Operator._UNSAFE_VIEW,
    aten.add.Tensor: Operator.ADD_TENSOR,
    aten.addmm: Operator.ADDMM,
    aten.alias: Operator.ALIAS,
    aten.arange: Operator.ARANGE,
    aten.arange.start: Operator.ARANGE_START,
    aten.bmm: Operator.BMM,
    aten.cat: Operator.CAT,
    aten.clone: Operator.CLONE,
    aten.cos: Operator.COS,
    aten.cumsum: Operator.CUMSUM,
    aten.embedding: Operator.EMBEDDING,
    aten.expand: Operator.EXPAND,
    aten.full: Operator.FULL,
    aten.gelu: Operator.GELU,
    aten.gt.Tensor: Operator.GT_TENSOR,
    aten.lt.Tensor: Operator.LT_TENSOR,
    aten.masked_fill.Scalar: Operator.MASKED_FILL_SCALAR,
    aten.mean.dim: Operator.MEAN_DIM,
    aten.mm: Operator.MM,
    aten.mul.Scalar: Operator.MUL_SCALAR,
    aten.mul.Tensor: Operator.MUL_TENSOR,
    aten.native_layer_norm: Operator.NATIVE_LAYER_NORM,
    aten.neg: Operator.NEG,
    aten.ones: Operator.ONES,
    aten.ones_like: Operator.ONES_LIKE,
    aten.permute: Operator.PERMUTE,
    aten.pow.Tensor_Scalar: Operator.POW_TENSOR_SCALAR,
    aten.relu: Operator.RELU,
    aten.rsqrt: Operator.RSQRT,
    aten.rsub.Scalar: Operator.RSUB_SCALAR,
    aten.scalar_tensor: Operator.SCALAR_TENSOR,
    aten.silu: Operator.SILU,
    aten.sin: Operator.SIN,
    aten.slice.Tensor: Operator.SLICE_TENSOR,
    aten.split.Tensor: Operator.SPLIT_TENSOR,
    aten.sub.Tensor: Operator.SUB_TENSOR,
    aten.t: Operator.T,
    aten.tanh: Operator.TANH,
    aten.transpose.int: Operator.TRANSPOSE_INT,
    aten.tril: Operator.TRIL,
    aten.triu: Operator.TRIU,
    aten.unsqueeze: Operator.UNSQUEEZE,
    aten.view: Operator.VIEW,
    aten.where.self: Operator.WHERE_SELF,
}


class UnsupportedOperatorWarning(UserWarning):
    """Warning that signals the presence of unsupported operators."""

    def __init__(
        self, args: Sequence[Union[OpOverload, OpOverloadPacket, CustomOpDef]]
    ) -> None:
        self.args = tuple(set(args))

    def __str__(self) -> str:
        message = (
            "The following operators are not supported\n{}\n"
            "Please make sure you are using the latest version of FlatFlow\n"
            "or file an issue to https://github.com/9rum/flatflow/issues\n"
            "The latest release can be found at https://github.com/9rum/flatflow/tags"
        )
        return message.format(
            "\n".join(sorted("\t{}".format(arg) for arg in self.args))
        )


def is_accessor_node(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_method"
        and isinstance(node.args[0], torch.fx.Node)
        and isinstance(node.args[0].meta["example_value"], torch.Tensor)
        and node.target in ["size", "stride", "storage_offset", "item"]
    ) or (
        node.op == "call_function"
        and node.target
        in [
            aten.sym_size,
            aten.sym_size.default,
            aten.sym_size.int,
            aten.sym_stride,
            aten.sym_stride.default,
            aten.sym_stride.int,
            aten.sym_storage_offset,
            aten.sym_storage_offset.default,
            aten.sym_numel.default,
        ]
    )


def serialize(builder: flatbuffers.Builder, graph: torch.fx.Graph) -> int:
    """Serializes the given computational graph."""
    blacklist = []
    nodes = []

    for node in graph.nodes:
        if not is_accessor_node(node) and isinstance(
            node.target, (OpOverload, OpOverloadPacket, CustomOpDef)
        ):
            if node.target not in _OPS_TABLE:
                blacklist.append(node.target)
                continue
            target = _OPS_TABLE[node.target]
            args = []

            for arg in node.args:
                if isinstance(arg, torch.fx.Node) and "tensor_meta" in arg.meta:
                    shape = []

                    for maybe_sym_int in arg.meta["tensor_meta"].shape:
                        if isinstance(maybe_sym_int, torch.SymInt):
                            expr = maybe_sym_int.node.expr
                            symbol = next(iter(expr.free_symbols))
                            shape.append([expr.coeff(symbol, 0), expr.coeff(symbol, 1)])
                        else:
                            shape.append([maybe_sym_int, 0])

                    TensorMetadataStartShapeVector(builder, len(shape))
                    for sym_int in reversed(shape):
                        CreateSymInt(builder, sym_int)
                    _shape = builder.EndVector()

                    TensorMetadataStart(builder)
                    TensorMetadataAddShape(builder, _shape)
                    _arg = TensorMetadataEnd(builder)
                    args.append(_arg)

            NodeStartArgsVector(builder, len(args))
            for arg in reversed(args):
                builder.PrependUOffsetTRelative(arg)
            _args = builder.EndVector()

            shape = []

            if "tensor_meta" in node.meta:
                for maybe_sym_int in node.meta["tensor_meta"].shape:
                    if isinstance(maybe_sym_int, torch.SymInt):
                        expr = maybe_sym_int.node.expr
                        symbol = next(iter(expr.free_symbols))
                        shape.append([expr.coeff(symbol, 0), expr.coeff(symbol, 1)])
                    else:
                        shape.append([maybe_sym_int, 0])

            TensorMetadataStartShapeVector(builder, len(shape))
            for sym_int in reversed(shape):
                CreateSymInt(builder, sym_int)
            _shape = builder.EndVector()

            TensorMetadataStart(builder)
            TensorMetadataAddShape(builder, _shape)
            _meta = TensorMetadataEnd(builder)

            NodeStart(builder)
            NodeAddTarget(builder, target)
            NodeAddArgs(builder, _args)
            NodeAddMeta(builder, _meta)
            _node = NodeEnd(builder)
            nodes.append(_node)

    if blacklist:
        warnings.warn(UnsupportedOperatorWarning(blacklist), stacklevel=2)

    GraphStartNodesVector(builder, len(nodes))
    for node in reversed(nodes):
        builder.PrependUOffsetTRelative(node)
    _nodes = builder.EndVector()

    GraphStart(builder)
    GraphAddNodes(builder, _nodes)
    return GraphEnd(builder)
