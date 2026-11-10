// SPDX-License-Identifier: Apache-2.0

#ifndef FLATFLOW_OPS_H_
#define FLATFLOW_OPS_H_

#include <chrono>
#include <format>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/time/time.h"
#include "flatbuffers/base.h"

#include "flatflow/ops/graph_generated.h"
#include "flatflow/ops/operator_generated.h"
#include "flatflow/ops/polynomial.h"
#include "flatflow/ops/scalar_type.h"
#include "flatflow/ops/scalar_type_generated.h"
#include "flatflow/ops/view.h"

namespace flatflow {

namespace detail {

// Returns a zero polynomial for operators that do not involve actual
// computations, such as tensor view operations, copies and tensor creations.
constexpr Polynomial noop() noexcept { return polynomial(); }

// Returns the total number of elements in a tensor with the given symbolic
// shape.
inline Polynomial numel(Vector<SymInt> shape) noexcept {
  auto numel = polynomial(1);
  for (Vector<SymInt>::size_type dim = 0; dim < shape.size(); ++dim) {
    numel *= polynomial(shape[dim]);
  }
  return numel;
}

// Helper function for element-wise operators.
inline Polynomial pointwise(
    Vector<SymInt> shape, ScalarType dtype,
    Polynomial::value_type ops_per_element = 1) noexcept {
  return ops_per_element * numel(shape) * polynomial(dtype);
}

inline Polynomial pointwise(
    TensorMetadata meta, Polynomial::value_type ops_per_element = 1) noexcept {
  return pointwise(meta.shape(), meta.dtype(), ops_per_element);
}

// Helper function for matrix products where each element of `meta` requires
// `inner` fused multiply-add operations.
inline Polynomial gemm(TensorMetadata meta, Polynomial inner) noexcept {
  return pointwise(meta, 2) * inner;
}

// This is a base template for implementing symbolic transformations for each
// node in a computational graph; calling this means that the program is
// ill-formed and should fail to compile.
template <Operator>
Polynomial transform_impl(Vector<TensorMetadata>,
                          TensorMetadata) noexcept = delete;

// Disable unused parameter warnings so we are allowed to avoid repeated use of
// the unused attribute (standard `[[maybe_unused]]` since C++17, otherwise
// `ABSL_ATTRIBUTE_UNUSED`), and turn unused enumerators in switch statements
// into errors so that `transform` below handles every operator.
#if defined(__GNUC__) || defined(__clang__)
// Clang also supports these GCC pragmas.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic error "-Wswitch"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)
#pragma warning(error : 4062)
#endif  // defined(__GNUC__) || defined(__clang__)

// func: _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None,
//                Device? device=None, bool? pin_memory=None,
//                bool non_blocking=False,
//                MemoryFormat? memory_format=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::_to_copy>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // _to_copy copies a tensor, so technically it has zero FLOPs.
  return noop();
}

// func: _unsafe_view(Tensor self, SymInt[] size) -> Tensor
template <>
inline Polynomial transform_impl<Operator::_unsafe_view>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // _unsafe_view is a tensor view operation, so technically it has zero FLOPs.
  return noop();
}

// func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
template <>
inline Polynomial transform_impl<Operator::add_tensor>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // add.Tensor adds `other`, scaled by `alpha`, to `self`.
  return pointwise(meta, 2);
}

// func: addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1,
//             Scalar alpha=1) -> Tensor
template <>
inline Polynomial transform_impl<Operator::addmm>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // addmm performs a matrix multiplication of the matrices `mat1` and `mat2`.
  // The matrix `self` is added to the final result. `alpha` and `beta` are
  // scaling factors on matrix product between `mat1` and `mat2` and the added
  // matrix `self` respectively. If `mat1` is a (n x m) tensor and `mat2` is a
  // (m x p) tensor, then it produces a (n x p) tensor with 2 x n x m x p FLOPs.
  // For scaling and addition, it requires additional 3 x n x p FLOPs.
  CHECK(std::cmp_equal(args.size(), 3));

  const auto mat1 = args[1].shape();
  CHECK(std::cmp_equal(mat1.size(), 2));

  return gemm(meta, mat1[1]) + pointwise(meta, 3);
}

// func: arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None,
//              Device? device=None, bool? pin_memory=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::arange>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // arange returns a tensor with values from the interval [0, `end`).
  return pointwise(meta, 2);
}

// func: arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None,
//                    Layout? layout=None, Device? device=None,
//                    bool? pin_memory=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::arange_start>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // arange.start returns a tensor with values from the interval
  // [`start`, `end`).
  return pointwise(meta, 2);
}

// func: bmm(Tensor self, Tensor mat2) -> Tensor
template <>
inline Polynomial transform_impl<Operator::bmm>(Vector<TensorMetadata> args,
                                                TensorMetadata meta) noexcept {
  // bmm performs a batch matrix-matrix product of matrices `self` and `mat2`.
  // `self` and `mat2` must be 3-D tensors each containing the same number of
  // matrices. If `self` is a (b x n x m) tensor and `mat2` is a (b x m x p)
  // tensor, then it produces a (b x n x p) tensor with 2 x b x n x m x p FLOPs.
  CHECK(std::cmp_equal(args.size(), 2));

  const auto self = args[0].shape();
  CHECK(std::cmp_equal(self.size(), 3));

  return gemm(meta, self[2]);
}

// func: cat(Tensor[] tensors, int dim=0) -> Tensor
template <>
inline Polynomial transform_impl<Operator::cat>(Vector<TensorMetadata> args,
                                                TensorMetadata meta) noexcept {
  // cat concatenates tensors in the given dimension, so it has zero FLOPs.
  return noop();
}

// func: clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::clone>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // clone copies a tensor, so technically it has zero FLOPs.
  return noop();
}

// func: cos(Tensor self) -> Tensor
template <>
inline Polynomial transform_impl<Operator::cos>(Vector<TensorMetadata> args,
                                                TensorMetadata meta) noexcept {
  // cos returns a new tensor with the cosine of the elements of `self`.
  CHECK(std::cmp_equal(args.size(), 1));

  return pointwise(args[0]);
}

// func: cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::cumsum>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // cumsum returns the cumulative sum of elements of `self` in the dimension
  // `dim`.
  //
  // NOTE: The exact number of FLOPs depends on `dim` which cannot be deduced
  // from the tensor metadata since cumsum always returns a tensor with the same
  // shape as `self`; the exact number of FLOPs can be obtained by subtracting
  // one from the corresponding dimension and then multiplying the size of the
  // remaining dimensions, but here we ignore that subtraction.
  return pointwise(meta);
}

// func: detach(Tensor(a) self) -> Tensor(a)
template <>
inline Polynomial transform_impl<Operator::detach>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // detach is a tensor view operation, so technically it has zero FLOPs.
  return noop();
}

// func: div.Tensor(Tensor self, Tensor other) -> Tensor
template <>
inline Polynomial transform_impl<Operator::div_tensor>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // div divides each element of `self` by the corresponding element of `other`.
  return pointwise(meta);
}

// func: dropout(Tensor input, float p, bool train) -> Tensor
template <>
inline Polynomial transform_impl<Operator::dropout>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // dropout randomly zeroes some elements of `input` with probability `p`.
  CHECK(std::cmp_equal(args.size(), 1));

  return pointwise(args[0], 4);
}

// func: embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1,
//                 bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
template <>
inline Polynomial transform_impl<Operator::embedding>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // embedding is a dictionary lookup, so technically it has zero FLOPs.
  return noop();
}

// func: expand(Tensor(a) self, SymInt[] size, *,
//              bool implicit=False) -> Tensor(a)
template <>
inline Polynomial transform_impl<Operator::expand>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // expand is a tensor view operation, so technically it has zero FLOPs.
  return noop();
}

// func: full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None,
//            Layout? layout=None, Device? device=None,
//            bool? pin_memory=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::full>(Vector<TensorMetadata> args,
                                                 TensorMetadata meta) noexcept {
  // full creates a tensor, so technically it has zero FLOPs.
  return noop();
}

// func: gelu(Tensor self, *, str approximate='none') -> Tensor
template <>
inline Polynomial transform_impl<Operator::gelu>(Vector<TensorMetadata> args,
                                                 TensorMetadata meta) noexcept {
  // gelu applies the gaussian error linear unit (GELU) function to `self` in
  // element-wise.
  CHECK(std::cmp_equal(args.size(), 1));

  return pointwise(args[0], 5);
}

// func: gt.Tensor(Tensor self, Tensor other) -> Tensor
template <>
inline Polynomial transform_impl<Operator::gt_tensor>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // gt.Tensor compares `self` and `other` in element-wise.
  CHECK(std::cmp_equal(args.size(), 2));

  const auto dtype = promote_types(args[0].dtype(), args[1].dtype());
  return pointwise(meta.shape(), dtype);
}

// func: layer_norm(Tensor input, SymInt[] normalized_shape,
//                  Tensor? weight=None, Tensor? bias=None, float eps=1e-05,
//                  bool cudnn_enable=True) -> Tensor
template <>
inline Polynomial transform_impl<Operator::layer_norm>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // layer_norm applies layer normalization over the last dimensions of `input`
  // given by `normalized_shape`, followed by an element-wise affine transform
  // with `weight` and `bias` if given.
  //
  // NOTE: layer_norm requires four FLOPs for each element to compute the mean
  // and the variance, i.e., one addition for the mean and one subtraction, one
  // multiplication and one addition for the variance, and two more to normalize
  // each element, i.e., one subtraction of the mean and one multiplication by
  // the reciprocal of the standard deviation. `weight` and `bias` take one
  // multiplication and one addition for each element, respectively.
  CHECK(!args.empty() && std::cmp_less_equal(args.size(), 3));

  return pointwise(args[0], 5 + args.size());
}

// func: lift_fresh_copy(Tensor self) -> Tensor
template <>
inline Polynomial transform_impl<Operator::lift_fresh_copy>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // lift_fresh_copy copies a tensor, so technically it has zero FLOPs.
  return noop();
}

// func: linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::linear>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // linear applies a linear transformation to `input`; i.e., the product of
  // `input` and the transpose of `weight`, plus `bias` if given.
  CHECK(std::cmp_equal(args.size(), 2) || std::cmp_equal(args.size(), 3));

  const auto weight = args[1].shape();
  CHECK(!weight.empty());
  const auto in_features = weight[weight.size() - 1];

  return std::cmp_equal(args.size(), 2)
             ? gemm(meta, in_features)
             : gemm(meta, in_features) + pointwise(meta);
}

// func: lt.Tensor(Tensor self, Tensor other) -> Tensor
template <>
inline Polynomial transform_impl<Operator::lt_tensor>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // lt.Tensor compares `self` and `other` in element-wise.
  CHECK(std::cmp_equal(args.size(), 2));

  const auto dtype = promote_types(args[0].dtype(), args[1].dtype());
  return pointwise(meta.shape(), dtype);
}

// func: masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
template <>
inline Polynomial transform_impl<Operator::masked_fill_scalar>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // masked_fill.Scalar is an element-wise selection between `self` and `value`
  // according to `mask`, so it has zero FLOPs.
  return noop();
}

// func: matmul(Tensor self, Tensor other) -> Tensor
template <>
inline Polynomial transform_impl<Operator::matmul>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // matmul computes the matrix product of two tensors `self` and `other`
  // including dot, matrix-vector, matrix-matrix and batched matrix-matrix
  // products with broadcasting of the batch dimensions.
  CHECK(std::cmp_equal(args.size(), 2));

  const auto self = args[0].shape();
  CHECK(!self.empty());

  return gemm(meta, self[self.size() - 1]);
}

// func: max.other(Tensor self, Tensor other) -> Tensor
template <>
inline Polynomial transform_impl<Operator::max_other>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // max.other computes the element-wise maximum of `self` and `other`.
  return pointwise(meta);
}

// func: mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *,
//                ScalarType? dtype=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::mean_dim>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // mean.dim returns the mean value of each row of `self` in the given
  // dimension `dim`.
  CHECK(std::cmp_equal(args.size(), 1));

  return pointwise(args[0].shape(), meta.dtype());
}

// func: mul.Tensor(Tensor self, Tensor other) -> Tensor
template <>
inline Polynomial transform_impl<Operator::mul_tensor>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // mul.Tensor multiplies `self` by `other` in element-wise.
  //
  // NOTE: As shown above, mul.Tensor gets two tensor arguments but we have
  // found that there is a bug during symbolic tracing where mul.Scalar is
  // replaced by mul.Tensor with a single argument due to constant folding.
  // Moreover, mul.Tensor supports broadcasting to a common shape, making it
  // difficult to trace FLOPs with input shapes alone; mul.Tensor has the same
  // FLOPs as mul.Scalar, and we leverage the output shape that reflects the
  // broadcasting.
  return pointwise(meta);
}

// func: neg(Tensor self) -> Tensor
template <>
inline Polynomial transform_impl<Operator::neg>(Vector<TensorMetadata> args,
                                                TensorMetadata meta) noexcept {
  // neg returns a new tensor with the negative of the elements of `self`.
  CHECK(std::cmp_equal(args.size(), 1));

  return pointwise(args[0]);
}

// func: ones(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None,
//            Device? device=None, bool? pin_memory=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::ones>(Vector<TensorMetadata> args,
                                                 TensorMetadata meta) noexcept {
  // ones creates a tensor, so technically it has zero FLOPs.
  return noop();
}

// func: ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None,
//                 Device? device=None, bool? pin_memory=None,
//                 MemoryFormat? memory_format=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::ones_like>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // ones_like creates a tensor, so technically it has zero FLOPs.
  return noop();
}

// func: permute(Tensor(a) self, int[] dims) -> Tensor(a)
template <>
inline Polynomial transform_impl<Operator::permute>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // permute is a tensor view operation, so technically it has zero FLOPs.
  return noop();
}

// func: pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
template <>
inline Polynomial transform_impl<Operator::pow_tensor_scalar>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // pow.Tensor_Scalar takes the power of each element in `self` with
  // `exponent`.
  return pointwise(meta);
}

// func: relu(Tensor self) -> Tensor
template <>
inline Polynomial transform_impl<Operator::relu>(Vector<TensorMetadata> args,
                                                 TensorMetadata meta) noexcept {
  // relu applies the rectified linear unit (ReLU) function to `self` in
  // element-wise.
  CHECK(std::cmp_equal(args.size(), 1));

  return pointwise(args[0]);
}

// func: rsqrt(Tensor self) -> Tensor
template <>
inline Polynomial transform_impl<Operator::rsqrt>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // rsqrt computes the element-wise reciprocal square root of `self`.
  CHECK(std::cmp_equal(args.size(), 1));

  return pointwise(args[0]);
}

// func: rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
template <>
inline Polynomial transform_impl<Operator::rsub_scalar>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // rsub.Scalar subtracts `self`, scaled by `alpha`, from `other`.
  return pointwise(meta, 2);
}

// func: silu(Tensor self) -> Tensor
template <>
inline Polynomial transform_impl<Operator::silu>(Vector<TensorMetadata> args,
                                                 TensorMetadata meta) noexcept {
  // silu applies the sigmoid linear unit (SiLU) function to `self` in
  // element-wise.
  CHECK(std::cmp_equal(args.size(), 1));

  return pointwise(args[0], 4);
}

// func: sin(Tensor self) -> Tensor
template <>
inline Polynomial transform_impl<Operator::sin>(Vector<TensorMetadata> args,
                                                TensorMetadata meta) noexcept {
  // sin returns a new tensor with the sine of the elements in `self`.
  CHECK(std::cmp_equal(args.size(), 1));

  return pointwise(args[0]);
}

// func: slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None,
//                    SymInt? end=None, SymInt step=1) -> Tensor(a)
template <>
inline Polynomial transform_impl<Operator::slice_tensor>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // slice.Tensor is a tensor view operation, so technically it has zero FLOPs.
  return noop();
}

// func: softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::softmax_int>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // softmax.int applies the softmax function to `self`.
  //
  // NOTE: softmax.int requires five FLOPs for each element.
  // Approximate FLOPs breakdown can be found at:
  // https://github.com/tensorflow/tensorflow/blob/v2.18.0/tensorflow/python/profiler/internal/flops_registry.py
  CHECK(std::cmp_equal(args.size(), 1));

  const auto half_to_float = args[0].dtype() == ScalarType::float16 &&
                             meta.dtype() == ScalarType::float32;

  return pointwise(args[0].shape(),
                   half_to_float ? args[0].dtype() : meta.dtype(), 5);
}

// func: split.Tensor(Tensor(a -> *) self, SymInt split_size,
//                    int dim=0) -> Tensor(a)[]
template <>
inline Polynomial transform_impl<Operator::split_tensor>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // split.Tensor splits a tensor into chunks, so technically it has zero FLOPs.
  return noop();
}

// func: sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
template <>
inline Polynomial transform_impl<Operator::sub_tensor>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // sub.Tensor subtracts `other`, scaled by `alpha`, from `self`.
  //
  // NOTE: As in the case of mul.Tensor, we have found that sub.Scalar is
  // replaced by sub.Tensor with a single argument due to constant folding.
  // sub.Tensor also supports broadcasting to a common shape, necessitating the
  // use of output shape.
  return pointwise(meta, 2);
}

// func: tanh(Tensor self) -> Tensor
template <>
inline Polynomial transform_impl<Operator::tanh>(Vector<TensorMetadata> args,
                                                 TensorMetadata meta) noexcept {
  // tanh applies the hyperbolic tangent (tanh) function to `self` in
  // element-wise.
  CHECK(std::cmp_equal(args.size(), 1));

  return pointwise(args[0]);
}

// func: transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
template <>
inline Polynomial transform_impl<Operator::transpose_int>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // transpose.int is a dimension swap, so technically it has zero FLOPs.
  return noop();
}

// func: tril(Tensor self, int diagonal=0) -> Tensor
template <>
inline Polynomial transform_impl<Operator::tril>(Vector<TensorMetadata> args,
                                                 TensorMetadata meta) noexcept {
  // tril is a masking operation, so technically it has zero FLOPs.
  return noop();
}

// func: triu(Tensor self, int diagonal=0) -> Tensor
template <>
inline Polynomial transform_impl<Operator::triu>(Vector<TensorMetadata> args,
                                                 TensorMetadata meta) noexcept {
  // triu is a masking operation, so technically it has zero FLOPs.
  return noop();
}

// func: type_as(Tensor self, Tensor other) -> Tensor
template <>
inline Polynomial transform_impl<Operator::type_as>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // type_as is a casting operation, so technically it has zero FLOPs.
  return noop();
}

// func: unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
template <>
inline Polynomial transform_impl<Operator::unsqueeze>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // unsqueeze inserts a singleton dimension at the specified position, so it
  // has zero FLOPs.
  return noop();
}

// func: view(Tensor(a) self, SymInt[] size) -> Tensor(a)
template <>
inline Polynomial transform_impl<Operator::view>(Vector<TensorMetadata> args,
                                                 TensorMetadata meta) noexcept {
  // view is a tensor view operation, so technically it has zero FLOPs.
  //
  // See https://pytorch.org/docs/stable/tensor_view.html.
  return noop();
}

// func: where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor
template <>
inline Polynomial transform_impl<Operator::where_scalarself>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // where.ScalarSelf is an element-wise selection between `self` and `other`
  // according to `condition`, so it has zero FLOPs.
  return noop();
}

// func: where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
template <>
inline Polynomial transform_impl<Operator::where_self>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // where.self is an element-wise selection between `self` and `other`
  // according to `condition`, so it has zero FLOPs.
  return noop();
}

// Executes the symbolic transformation corresponding to the given operator.
//
// NOTE: To register a new operator, please follow the instructions below:
//
//  - First, declare a new operator as an enumerator value of `Operator` in the
//    FlatBuffers schema, i.e., `flatflow/ops/operator.fbs`. The enumerator
//    values are recommended to be sorted in order of their original operator
//    names (not the enumerator values themselves) for searching convenience.
//  - Second, generate codes from the updated schema by running `make generate`
//    in the root directory of this source tree. This will create
//    `flatflow/ops/operator_generated.h`, etc.
//  - Third, add a template specialization of `transform_impl` for the new
//    operator; please refer to the implementations above for supported
//    operators. This step is checked at compile time.
//  - Finally, register the new operator as a case in the switch statement
//    below. Omitting this step may result in a compilation error by the pragmas
//    above. Do not add a default label, which disables this check. For the
//    Python frontend, add the ATen counterpart and enumerator value as key and
//    value respectively to the operator table in `flatflow/ops/serialize.py`.
//
// CAVEATS
//
// We provide only a handful of ATen operator set for now. The operator set is
// under development and more operators will be added in the future. For
// expanding the operator set, please refer to the note above.
inline Polynomial transform(Operator op, Vector<TensorMetadata> args,
                            TensorMetadata meta) noexcept {
  CHECK(flatbuffers::IsInRange(op, Operator::MIN, Operator::MAX));

  switch (op) {
    case Operator::_to_copy:
      return transform_impl<Operator::_to_copy>(args, meta);
    case Operator::_unsafe_view:
      return transform_impl<Operator::_unsafe_view>(args, meta);
    case Operator::add_tensor:
      return transform_impl<Operator::add_tensor>(args, meta);
    case Operator::addmm:
      return transform_impl<Operator::addmm>(args, meta);
    case Operator::arange:
      return transform_impl<Operator::arange>(args, meta);
    case Operator::arange_start:
      return transform_impl<Operator::arange_start>(args, meta);
    case Operator::bmm:
      return transform_impl<Operator::bmm>(args, meta);
    case Operator::cat:
      return transform_impl<Operator::cat>(args, meta);
    case Operator::clone:
      return transform_impl<Operator::clone>(args, meta);
    case Operator::cos:
      return transform_impl<Operator::cos>(args, meta);
    case Operator::cumsum:
      return transform_impl<Operator::cumsum>(args, meta);
    case Operator::detach:
      return transform_impl<Operator::detach>(args, meta);
    case Operator::div_tensor:
      return transform_impl<Operator::div_tensor>(args, meta);
    case Operator::dropout:
      return transform_impl<Operator::dropout>(args, meta);
    case Operator::embedding:
      return transform_impl<Operator::embedding>(args, meta);
    case Operator::expand:
      return transform_impl<Operator::expand>(args, meta);
    case Operator::full:
      return transform_impl<Operator::full>(args, meta);
    case Operator::gelu:
      return transform_impl<Operator::gelu>(args, meta);
    case Operator::gt_tensor:
      return transform_impl<Operator::gt_tensor>(args, meta);
    case Operator::layer_norm:
      return transform_impl<Operator::layer_norm>(args, meta);
    case Operator::lift_fresh_copy:
      return transform_impl<Operator::lift_fresh_copy>(args, meta);
    case Operator::linear:
      return transform_impl<Operator::linear>(args, meta);
    case Operator::lt_tensor:
      return transform_impl<Operator::lt_tensor>(args, meta);
    case Operator::masked_fill_scalar:
      return transform_impl<Operator::masked_fill_scalar>(args, meta);
    case Operator::matmul:
      return transform_impl<Operator::matmul>(args, meta);
    case Operator::max_other:
      return transform_impl<Operator::max_other>(args, meta);
    case Operator::mean_dim:
      return transform_impl<Operator::mean_dim>(args, meta);
    case Operator::mul_tensor:
      return transform_impl<Operator::mul_tensor>(args, meta);
    case Operator::neg:
      return transform_impl<Operator::neg>(args, meta);
    case Operator::ones:
      return transform_impl<Operator::ones>(args, meta);
    case Operator::ones_like:
      return transform_impl<Operator::ones_like>(args, meta);
    case Operator::permute:
      return transform_impl<Operator::permute>(args, meta);
    case Operator::pow_tensor_scalar:
      return transform_impl<Operator::pow_tensor_scalar>(args, meta);
    case Operator::relu:
      return transform_impl<Operator::relu>(args, meta);
    case Operator::rsqrt:
      return transform_impl<Operator::rsqrt>(args, meta);
    case Operator::rsub_scalar:
      return transform_impl<Operator::rsub_scalar>(args, meta);
    case Operator::silu:
      return transform_impl<Operator::silu>(args, meta);
    case Operator::sin:
      return transform_impl<Operator::sin>(args, meta);
    case Operator::slice_tensor:
      return transform_impl<Operator::slice_tensor>(args, meta);
    case Operator::softmax_int:
      return transform_impl<Operator::softmax_int>(args, meta);
    case Operator::split_tensor:
      return transform_impl<Operator::split_tensor>(args, meta);
    case Operator::sub_tensor:
      return transform_impl<Operator::sub_tensor>(args, meta);
    case Operator::tanh:
      return transform_impl<Operator::tanh>(args, meta);
    case Operator::transpose_int:
      return transform_impl<Operator::transpose_int>(args, meta);
    case Operator::tril:
      return transform_impl<Operator::tril>(args, meta);
    case Operator::triu:
      return transform_impl<Operator::triu>(args, meta);
    case Operator::type_as:
      return transform_impl<Operator::type_as>(args, meta);
    case Operator::unsqueeze:
      return transform_impl<Operator::unsqueeze>(args, meta);
    case Operator::view:
      return transform_impl<Operator::view>(args, meta);
    case Operator::where_scalarself:
      return transform_impl<Operator::where_scalarself>(args, meta);
    case Operator::where_self:
      return transform_impl<Operator::where_self>(args, meta);
  }

  ABSL_UNREACHABLE();
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif  // defined(__GNUC__) || defined(__clang__)

}  // namespace detail

#pragma omp declare reduction(+ : Polynomial : omp_out += omp_in) \
    initializer(omp_priv = polynomial())

// Transforms the given computational graph into a functor via symbolic
// transformation, then scaled for tensor and context parallelism.
//
// CAVEATS
//
// `graph` must be passed from a verified buffer; this function relies on the
// verification to omit null and bounds checks and it cannot verify the graph by
// itself since the size of underlying buffer is unknown.
inline Polynomial transform(
    const Graph *graph, Polynomial::value_type tensor_parallel_world_size,
    Polynomial::value_type context_parallel_world_size) {
  const auto now = std::chrono::steady_clock::now();
  const auto nodes = graph->nodes();

  auto expr = polynomial();

  // The nodes are traversed in parallel where the number of threads follows the
  // `OMP_NUM_THREADS` environment variable. Note that torchrun sets it to one
  // by default when running multiple processes per node; to use more threads,
  // set it explicitly upon launch, e.g., to the number of cores divided by the
  // number of processes per node.
#pragma omp parallel for reduction(+ : expr)
  for (const auto *node : *nodes) {
    expr += detail::transform(node->target(), node->args(), node->meta());
  }

  LOG(INFO) << std::format(
      "Traversing a graph with {} nodes took {}", nodes->size(),
      absl::FormatDuration(
          absl::FromChrono(std::chrono::steady_clock::now() - now)));

  return expr.with_tensor_parallel(tensor_parallel_world_size)
      .with_context_parallel(context_parallel_world_size)
      .normalized();
}

}  // namespace flatflow

#endif  // FLATFLOW_OPS_H_
