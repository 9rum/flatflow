# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Mapping, Sequence

import flatbuffers
import torch
import torch.fx
from torch._library.custom_ops import CustomOpDef
from torch._ops import OpOverload, OpOverloadPacket
from torch.fx.experimental.symbolic_shapes import is_accessor_node

from flatflow.ops.graph_generated import (
    CreateSymInt,
    GraphAddNodes,
    GraphEnd,
    GraphStart,
    GraphStartNodesVector,
    NodeAddArgs,
    NodeAddMeta,
    NodeAddTarget,
    NodeEnd,
    NodeStart,
    NodeStartArgsVector,
    TensorMetadataAddDtype,
    TensorMetadataAddShape,
    TensorMetadataEnd,
    TensorMetadataStart,
    TensorMetadataStartShapeVector,
)
from flatflow.ops.operator_generated import Operator
from flatflow.ops.scalar_type_generated import ScalarType

aten = torch.ops.aten

__all__ = ["serialize"]

_DTYPE_TABLE: Mapping[torch.dtype, int] = {
    torch.float32: ScalarType.float32,
    torch.float64: ScalarType.float64,
    torch.float16: ScalarType.float16,
    torch.bfloat16: ScalarType.bfloat16,
    torch.bool: ScalarType.bool,
    torch.int8: ScalarType.int8,
    torch.int16: ScalarType.int16,
    torch.int32: ScalarType.int32,
    torch.int64: ScalarType.int64,
    torch.uint8: ScalarType.uint8,
    torch.uint16: ScalarType.uint16,
    torch.uint32: ScalarType.uint32,
    torch.uint64: ScalarType.uint64,
    torch.float8_e4m3fn: ScalarType.float8_e4m3fn,
    torch.float8_e4m3fnuz: ScalarType.float8_e4m3fnuz,
    torch.float8_e5m2: ScalarType.float8_e5m2,
    torch.float8_e5m2fnuz: ScalarType.float8_e5m2fnuz,
}

_OPS_TABLE: Mapping[OpOverload | OpOverloadPacket | CustomOpDef, int] = {
    aten._softmax: Operator._softmax,
    aten._to_copy: Operator._to_copy,
    aten._unsafe_view: Operator._unsafe_view,
    aten.add.Tensor: Operator.add_tensor,
    aten.addmm: Operator.addmm,
    aten.alias: Operator.alias,
    aten.all.dim: Operator.all_dim,
    aten.arange: Operator.arange,
    aten.arange.start: Operator.arange_start,
    aten.bitwise_not: Operator.bitwise_not,
    aten.bmm: Operator.bmm,
    aten.cat: Operator.cat,
    aten.clone: Operator.clone,
    aten.copy: Operator.copy,
    aten.cumsum: Operator.cumsum,
    aten.embedding: Operator.embedding,
    aten.eq.Scalar: Operator.eq_scalar,
    aten.expand: Operator.expand,
    aten.full: Operator.full,
    aten.gelu: Operator.gelu,
    aten.gt.Tensor: Operator.gt_tensor,
    aten.lt.Tensor: Operator.lt_tensor,
    aten.masked_fill.Scalar: Operator.masked_fill_scalar,
    aten.mean.dim: Operator.mean_dim,
    aten.mm: Operator.mm,
    aten.mul.Scalar: Operator.mul_scalar,
    aten.mul.Tensor: Operator.mul_tensor,
    aten.native_layer_norm: Operator.native_layer_norm,
    aten.neg: Operator.neg,
    aten.ones: Operator.ones,
    aten.ones_like: Operator.ones_like,
    aten.permute: Operator.permute,
    aten.pow.Tensor_Scalar: Operator.pow_tensor_scalar,
    aten.relu: Operator.relu,
    aten.rsqrt: Operator.rsqrt,
    aten.rsub.Scalar: Operator.rsub_scalar,
    aten.scalar_tensor: Operator.scalar_tensor,
    aten.silu: Operator.silu,
    aten.slice.Tensor: Operator.slice_tensor,
    aten.slice_scatter: Operator.slice_scatter,
    aten.split.Tensor: Operator.split_tensor,
    aten.sub.Tensor: Operator.sub_tensor,
    aten.t: Operator.t,
    aten.tanh: Operator.tanh,
    aten.transpose.int: Operator.transpose_int,
    aten.tril: Operator.tril,
    aten.triu: Operator.triu,
    aten.unsqueeze: Operator.unsqueeze,
    aten.view: Operator.view,
    aten.where.self: Operator.where_self,
}


class UnsupportedOperatorWarning(UserWarning):
    """Warning that signals the presence of unsupported operators."""

    def __init__(
        self, args: Sequence[OpOverload | OpOverloadPacket | CustomOpDef]
    ) -> None:
        self.args = tuple(set(args))

    def __str__(self) -> str:
        message = (
            "The following operators are not supported\n{}\n"
            "Please make sure you are using the latest version of FlatFlow\n"
            "or file an issue to https://github.com/9rum/flatflow/issues\n"
            "The latest release can be found at https://github.com/9rum/flatflow/tags"
        )
        return message.format("\n".join(sorted(f"\t{arg}" for arg in self.args)))


def serialize(graph: torch.fx.Graph) -> bytes:
    """Serializes the given computational graph."""
    builder = flatbuffers.Builder()
    blocklist = []
    nodes = []

    for node in graph.nodes:
        if not is_accessor_node(node) and isinstance(
            node.target, (OpOverload, OpOverloadPacket, CustomOpDef)
        ):
            if node.target in _OPS_TABLE:
                target = _OPS_TABLE[node.target]
            elif (
                isinstance(node.target, OpOverload)
                and node.target.overloadpacket in _OPS_TABLE
            ):
                target = _OPS_TABLE[node.target.overloadpacket]
            else:
                blocklist.append(node.target)
                continue
            args = []

            for arg in node.args:
                if isinstance(arg, torch.fx.Node) and "tensor_meta" in arg.meta:
                    assert arg.meta["tensor_meta"].dtype in _DTYPE_TABLE
                    dtype = _DTYPE_TABLE[arg.meta["tensor_meta"].dtype]
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
                    TensorMetadataAddDtype(builder, dtype)
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
            if "tensor_meta" in node.meta:
                assert node.meta["tensor_meta"].dtype in _DTYPE_TABLE
                dtype = _DTYPE_TABLE[node.meta["tensor_meta"].dtype]
                TensorMetadataAddDtype(builder, dtype)
            TensorMetadataAddShape(builder, _shape)
            _meta = TensorMetadataEnd(builder)

            NodeStart(builder)
            NodeAddTarget(builder, target)
            NodeAddArgs(builder, _args)
            NodeAddMeta(builder, _meta)
            _node = NodeEnd(builder)
            nodes.append(_node)

    if blocklist:
        warnings.warn(UnsupportedOperatorWarning(blocklist), stacklevel=2)

    GraphStartNodesVector(builder, len(nodes))
    for node in reversed(nodes):
        builder.PrependUOffsetTRelative(node)
    _nodes = builder.EndVector()

    GraphStart(builder)
    GraphAddNodes(builder, _nodes)
    _graph = GraphEnd(builder)
    builder.Finish(_graph)
    return bytes(builder.Output())
