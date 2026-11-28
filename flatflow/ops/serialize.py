# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Iterable, Iterator
from operator import attrgetter

import flatbuffers
import torch
from torch._ops import HigherOrderOperator, OpOverload
from torch.fx.experimental.symbolic_shapes import is_accessor_node, is_concrete_int
from torch.utils._pytree import tree_leaves

from flatflow.ops.graph_generated import (
    CreateSymInt,
    FakeTensorAddDtype,
    FakeTensorAddShape,
    FakeTensorEnd,
    FakeTensorStart,
    FakeTensorStartShapeVector,
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
    NodeStartMetaVector,
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
                        submodule = attrgetter(arg.target)(gm)  # type: ignore[arg-type]
                        if isinstance(submodule, torch.fx.GraphModule):
                            yield from _walk(submodule)
            elif not is_accessor_node(node) and isinstance(node.target, OpOverload):
                yield node


def _to_fake_tensor(builder: flatbuffers.Builder, tensor: torch.Tensor) -> int:
    name = str(tensor.dtype).removeprefix("torch.")
    dtype = getattr(ScalarType, name)

    FakeTensorStartShapeVector(builder, len(tensor.shape))
    for s in reversed(tensor.shape):
        if isinstance(s, torch.SymInt) and not is_concrete_int(s):
            expr = s.node.expr
            (symbol,) = expr.free_symbols
            data = (expr.coeff(symbol, 0), expr.coeff(symbol, 1))
        else:
            data = (s, 0)
        CreateSymInt(builder, data)
    shape = builder.EndVector()

    FakeTensorStart(builder)
    FakeTensorAddDtype(builder, dtype)
    FakeTensorAddShape(builder, shape)
    return FakeTensorEnd(builder)


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
        for arg in tree_leaves(node.args):
            if isinstance(arg, torch.fx.Node):
                val = arg.meta.get("val")
                if isinstance(val, torch.Tensor):
                    args.append(_to_fake_tensor(builder, val))

        NodeStartArgsVector(builder, len(args))
        for tensor in reversed(args):
            builder.PrependUOffsetTRelative(tensor)
        args = builder.EndVector()

        meta = []
        for val in tree_leaves(node.meta.get("val")):
            if isinstance(val, torch.Tensor):
                meta.append(_to_fake_tensor(builder, val))

        NodeStartMetaVector(builder, len(meta))
        for tensor in reversed(meta):
            builder.PrependUOffsetTRelative(tensor)
        meta = builder.EndVector()

        NodeStart(builder)
        NodeAddTarget(builder, target)
        NodeAddArgs(builder, args)
        NodeAddMeta(builder, meta)
        nodes.append(NodeEnd(builder))

    if blocklist:
        warnings.warn(UnsupportedOperatorWarning(blocklist), stacklevel=2)

    GraphStartNodesVector(builder, len(nodes))
    for node in reversed(nodes):
        builder.PrependUOffsetTRelative(node)
    nodes = builder.EndVector()

    GraphStart(builder)
    GraphAddNodes(builder, nodes)
    builder.Finish(GraphEnd(builder))
    return bytes(builder.Output())
