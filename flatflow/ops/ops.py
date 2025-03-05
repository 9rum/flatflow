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

from flatflow.ops.operator_generated import Operator

__all__ = ["_OPCODES"]

_OPCODES = {
    "aten::_softmax": Operator._SOFTMAX,
    "aten::_to_copy": Operator._TO_COPY,
    "aten::_unsafe_view": Operator._UNSAFE_VIEW,
    "aten::add.Tensor": Operator.ADD_TENSOR,
    "aten::arange": Operator.ARANGE,
    "aten::arange.start": Operator.ARANGE_START,
    "aten::bmm": Operator.BMM,
    "aten::cat": Operator.CAT,
    "aten::clone": Operator.CLONE,
    "aten::cos": Operator.COS,
    "aten::embedding": Operator.EMBEDDING,
    "aten::expand": Operator.EXPAND,
    "aten::full": Operator.FULL,
    "aten::gt.Tensor": Operator.GT_TENSOR,
    "aten::mean.dim": Operator.MEAN_DIM,
    "aten::mm": Operator.MM,
    "aten::mul.Scalar": Operator.MUL_SCALAR,
    "aten::mul.Tensor": Operator.MUL_TENSOR,
    "aten::neg": Operator.NEG,
    "aten::pow.Tensor_Scalar": Operator.POW_TENSOR_SCALAR,
    "aten::rsqrt": Operator.RSQRT,
    "aten::silu": Operator.SILU,
    "aten::sin": Operator.SIN,
    "aten::slice.Tensor": Operator.SLICE_TENSOR,
    "aten::sym_size.int": Operator.SYM_SIZE_INT,
    "aten::t": Operator.T,
    "aten::transpose.int": Operator.TRANSPOSE_INT,
    "aten::triu": Operator.TRIU,
    "aten::unsqueeze": Operator.UNSQUEEZE,
    "aten::view": Operator.VIEW,
}
