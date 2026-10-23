// SPDX-License-Identifier: Apache-2.0

#ifndef FLATFLOW_OPS_H_
#define FLATFLOW_OPS_H_

#include "flatflow/ops/operator_generated.h"
#include "flatflow/ops/polynomial.h"
#include "flatflow/ops/view.h"

namespace flatflow {

namespace detail {

// Returns a zero polynomial for operators that do not involve actual
// computations, such as tensor view operations, copies and tensor creations.
constexpr Polynomial noop() noexcept { return polynomial(); }

// This is a base template for implementing symbolic transformations for each
// node in a computational graph; calling this means that the program is
// ill-formed and should fail to compile.
template <Operator>
Polynomial transform_impl(Vector<TensorMetadata>,
                          TensorMetadata) noexcept = delete;

// Disable unused parameter warnings so we are allowed to avoid repeated use of
// the unused attribute (standard `[[maybe_unused]]` since C++17, otherwise
// `ABSL_ATTRIBUTE_UNUSED`).
#if defined(__GNUC__) || defined(__clang__)
// Clang also supports these GCC pragmas.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)
#endif  // defined(__GNUC__) || defined(__clang__)

// func: _softmax(Tensor self, int dim, bool half_to_float) -> Tensor
template <>
inline Polynomial transform_impl<Operator::_softmax>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
}

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
  // TODO
}

// func: addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1,
//             Scalar alpha=1) -> Tensor
template <>
inline Polynomial transform_impl<Operator::addmm>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
}

// func: alias(Tensor(a) self) -> Tensor(a)
template <>
inline Polynomial transform_impl<Operator::alias>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // alias is a tensor view operation, so technically it has zero FLOPs.
  return noop();
}

// func: all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
template <>
inline Polynomial transform_impl<Operator::all_dim>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
}

// func: arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None,
//              Device? device=None, bool? pin_memory=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::arange>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
}

// func: arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None,
//                    Layout? layout=None, Device? device=None,
//                    bool? pin_memory=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::arange_start>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
}

// func: bitwise_not(Tensor self) -> Tensor
template <>
inline Polynomial transform_impl<Operator::bitwise_not>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
}

// func: bmm(Tensor self, Tensor mat2) -> Tensor
template <>
inline Polynomial transform_impl<Operator::bmm>(Vector<TensorMetadata> args,
                                                TensorMetadata meta) noexcept {
  // TODO
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

// func: copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor
template <>
inline Polynomial transform_impl<Operator::copy>(Vector<TensorMetadata> args,
                                                 TensorMetadata meta) noexcept {
  // copy copies a tensor, so technically it has zero FLOPs.
  return noop();
}

// func: cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::cumsum>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
}

// func: embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1,
//                 bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
template <>
inline Polynomial transform_impl<Operator::embedding>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // embedding is a dictionary lookup, so technically it has zero FLOPs.
  return noop();
}

// func: eq.Scalar(Tensor self, Scalar other) -> Tensor
template <>
inline Polynomial transform_impl<Operator::eq_scalar>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
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
  // TODO
}

// func: gt.Tensor(Tensor self, Tensor other) -> Tensor
template <>
inline Polynomial transform_impl<Operator::gt_tensor>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
}

// func: lt.Tensor(Tensor self, Tensor other) -> Tensor
template <>
inline Polynomial transform_impl<Operator::lt_tensor>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
}

// func: masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
template <>
inline Polynomial transform_impl<Operator::masked_fill_scalar>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // masked_fill.Scalar is an element-wise selection between `self` and `value`
  // according to `mask`, so it has zero FLOPs.
  return noop();
}

// func: mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *,
//                ScalarType? dtype=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::mean_dim>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
}

// func: mm(Tensor self, Tensor mat2) -> Tensor
template <>
inline Polynomial transform_impl<Operator::mm>(Vector<TensorMetadata> args,
                                               TensorMetadata meta) noexcept {
  // TODO
}

// func: mul.Scalar(Tensor self, Scalar other) -> Tensor
template <>
inline Polynomial transform_impl<Operator::mul_scalar>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
}

// func: mul.Tensor(Tensor self, Tensor other) -> Tensor
template <>
inline Polynomial transform_impl<Operator::mul_tensor>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
}

// func: native_layer_norm(Tensor input, SymInt[] normalized_shape,
//                         Tensor? weight, Tensor? bias,
//                         float eps) -> (Tensor, Tensor, Tensor)
template <>
inline Polynomial transform_impl<Operator::native_layer_norm>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
}

// func: neg(Tensor self) -> Tensor
template <>
inline Polynomial transform_impl<Operator::neg>(Vector<TensorMetadata> args,
                                                TensorMetadata meta) noexcept {
  // TODO
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
  // TODO
}

// func: relu(Tensor self) -> Tensor
template <>
inline Polynomial transform_impl<Operator::relu>(Vector<TensorMetadata> args,
                                                 TensorMetadata meta) noexcept {
  // TODO
}

// func: rsqrt(Tensor self) -> Tensor
template <>
inline Polynomial transform_impl<Operator::rsqrt>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
}

// func: rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
template <>
inline Polynomial transform_impl<Operator::rsub_scalar>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // TODO
}

// func: scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None,
//                     Device? device=None, bool? pin_memory=None) -> Tensor
template <>
inline Polynomial transform_impl<Operator::scalar_tensor>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // scalar_tensor creates a tensor, so technically it has zero FLOPs.
  return noop();
}

// func: silu(Tensor self) -> Tensor
template <>
inline Polynomial transform_impl<Operator::silu>(Vector<TensorMetadata> args,
                                                 TensorMetadata meta) noexcept {
  // TODO
}

// func: slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None,
//                    SymInt? end=None, SymInt step=1) -> Tensor(a)
template <>
inline Polynomial transform_impl<Operator::slice_tensor>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // slice.Tensor is a tensor view operation, so technically it has zero FLOPs.
  return noop();
}

// func: slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None,
//                     SymInt? end=None, SymInt step=1) -> Tensor
template <>
inline Polynomial transform_impl<Operator::slice_scatter>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // slice_scatter embeds the values of `src` into `self` at the given
  // dimension, so technically it has zero FLOPs.
  return noop();
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
  // TODO
}

// func: t(Tensor(a) self) -> Tensor(a)
template <>
inline Polynomial transform_impl<Operator::t>(Vector<TensorMetadata> args,
                                              TensorMetadata meta) noexcept {
  // t is a dimension swap, so technically it has zero FLOPs.
  return noop();
}

// func: tanh(Tensor self) -> Tensor
template <>
inline Polynomial transform_impl<Operator::tanh>(Vector<TensorMetadata> args,
                                                 TensorMetadata meta) noexcept {
  // TODO
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

// func: where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
template <>
inline Polynomial transform_impl<Operator::where_self>(
    Vector<TensorMetadata> args, TensorMetadata meta) noexcept {
  // where.self is an element-wise selection between `self` and `other`
  // according to `condition`, so it has zero FLOPs.
  return noop();
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif  // defined(__GNUC__) || defined(__clang__)

}  // namespace detail

}  // namespace flatflow

#endif  // FLATFLOW_OPS_H_
