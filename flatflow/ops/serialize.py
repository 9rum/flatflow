# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Iterable, Iterator

import flatbuffers
import torch
from torch._ops import HigherOrderOperator, OpOverload
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

__all__ = ["serialize"]


class UnsupportedOperatorWarning(UserWarning):
    """Warning that signals the presence of unsupported operators."""

    def __init__(self, names: Iterable[str]) -> None:
        self.names = sorted(set(names))
        super().__init__(self.names)

    def __str__(self) -> str:
        names = "\n".join(f"\t{name}" for name in self.names)
        return (
            f"The following operators are not supported\n{names}\n"
            "Please make sure you are using the latest version of FlatFlow\n"
            "or file an issue to https://github.com/9rum/flatflow/issues\n"
            "The latest release can be found at https://github.com/9rum/flatflow/tags"
        )


def _walk(gm: torch.fx.GraphModule) -> Iterator[torch.fx.Node]:
    for node in gm.graph.nodes:
        if node.op == "call_function":
            if isinstance(node.target, HigherOrderOperator):
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
                        yield from _walk(gm.get_submodule(arg.target))  # type: ignore[arg-type]
            elif not is_accessor_node(node) and isinstance(node.target, OpOverload):
                yield node


def serialize(gm: torch.fx.GraphModule) -> bytes:
    """Serializes the computational graph of the given graph module."""
    builder = flatbuffers.Builder()
    blocklist = []

    nodes = []
    for node in _walk(gm):
        name = node.target.name().removeprefix("aten::").replace(".", "_").lower()  # type: ignore[union-attr]
        target = getattr(Operator, name, None)
        if target is None:
            blocklist.append(node.target.name())  # type: ignore[union-attr]
            continue

        args = []
        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and "tensor_meta" in arg.meta:
                name = str(arg.meta["tensor_meta"].dtype).removeprefix("torch.")
                dtype = getattr(ScalarType, name)

                shape = arg.meta["tensor_meta"].shape
                TensorMetadataStartShapeVector(builder, len(shape))
                for s in reversed(shape):
                    if isinstance(s, torch.SymInt):
                        expr = s.node.expr
                        (symbol,) = expr.free_symbols
                        data = (expr.coeff(symbol, 0), expr.coeff(symbol, 1))
                    else:
                        data = (s, 0)
                    CreateSymInt(builder, data)
                _shape = builder.EndVector()

                TensorMetadataStart(builder)
                TensorMetadataAddDtype(builder, dtype)
                TensorMetadataAddShape(builder, _shape)
                args.append(TensorMetadataEnd(builder))

        NodeStartArgsVector(builder, len(args))
        for arg in reversed(args):
            builder.PrependUOffsetTRelative(arg)
        _args = builder.EndVector()

        if "tensor_meta" in node.meta:
            name = str(node.meta["tensor_meta"].dtype).removeprefix("torch.")
            dtype = getattr(ScalarType, name)
            shape = node.meta["tensor_meta"].shape
        else:
            shape = torch.Size()

        TensorMetadataStartShapeVector(builder, len(shape))
        for s in reversed(shape):
            if isinstance(s, torch.SymInt):
                expr = s.node.expr
                (symbol,) = expr.free_symbols
                data = (expr.coeff(symbol, 0), expr.coeff(symbol, 1))
            else:
                data = (s, 0)
            CreateSymInt(builder, data)
        _shape = builder.EndVector()

        TensorMetadataStart(builder)
        if "tensor_meta" in node.meta:
            TensorMetadataAddDtype(builder, dtype)
        TensorMetadataAddShape(builder, _shape)
        _meta = TensorMetadataEnd(builder)

        NodeStart(builder)
        NodeAddTarget(builder, target)
        NodeAddArgs(builder, _args)
        NodeAddMeta(builder, _meta)
        nodes.append(NodeEnd(builder))

    if blocklist:
        warnings.warn(UnsupportedOperatorWarning(blocklist), stacklevel=2)

    GraphStartNodesVector(builder, len(nodes))
    for node in reversed(nodes):
        builder.PrependUOffsetTRelative(node)
    _nodes = builder.EndVector()

    GraphStart(builder)
    GraphAddNodes(builder, _nodes)
    builder.Finish(GraphEnd(builder))
    return bytes(builder.Output())
