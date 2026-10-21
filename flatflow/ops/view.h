// SPDX-License-Identifier: Apache-2.0

#ifndef FLATFLOW_OPS_VIEW_H_
#define FLATFLOW_OPS_VIEW_H_

#include <cstdint>
#include <type_traits>

#include "flatbuffers/base.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/table.h"
#include "flatbuffers/vector.h"

#include "flatflow/ops/graph_generated.h"
#include "flatflow/ops/scalar_type_generated.h"

namespace flatflow {
namespace detail {

// `View` is the base of the views below, which provide an abstraction for
// read-only access to a serialized graph without copying the underlying data.
// The views are trivially copyable and meant to be passed by value.
//
// CAVEATS
//
// A view does not own the buffer it refers to, so the buffer must outlive the
// view. A view must also be constructed from a buffer that has been verified;
// every field in the schema is required and thus the verification ensures that
// no field within the buffer is null, allowing views to omit null checks.
template <typename T>
class View {
 public:
  using element_type = T;

  constexpr View(const T *ptr) noexcept : ptr_(ptr) {}

 protected:
  constexpr const T *get() const noexcept { return ptr_; }

 private:
  const T *ptr_;
};

// `Vector` is a view of a FlatBuffers vector whose elements are in turn
// accessed through the view of the element type.
//
// FlatBuffers vector stores structures inline and tables by offset; the
// template below covers vectors of structures while its partial specialization
// covers vectors of tables.
//
// Note that `std::is_base_of_v` which disregards access control is used instead
// of `std::derived_from` since generated tables inherit `flatbuffers::Table`
// privately.
template <typename T, bool = std::is_base_of_v<flatbuffers::Table,
                                               typename T::element_type>>
class Vector
    : public View<flatbuffers::Vector<const typename T::element_type *>> {
  using Base = typename Vector::View;

 public:
  using size_type = Base::element_type::size_type;

  using Base::Base;

  size_type size() const noexcept { return this->get()->size(); }

  bool empty() const noexcept { return this->get()->empty(); }

  T operator[](size_type pos) const noexcept { return this->get()->Get(pos); }
};

template <typename T>
class Vector<T, true>
    : public View<
          flatbuffers::Vector<flatbuffers::Offset<typename T::element_type>>> {
  using Base = typename Vector::View;

 public:
  using size_type = Base::element_type::size_type;

  using Base::Base;

  size_type size() const noexcept { return this->get()->size(); }

  bool empty() const noexcept { return this->get()->empty(); }

  T operator[](size_type pos) const noexcept { return this->get()->Get(pos); }
};

// `SymInt` provides a lightweight view into the serialized symbolic integer
// which records a value within the symbolic shape of a tensor.
class SymInt : public View<flatflow::SymInt> {
  using Base = typename SymInt::View;

 public:
  using Base::Base;

  int64_t operator[](flatbuffers::uoffset_t pos) const noexcept {
    return get()->data()->Get(pos);
  }
};

// `TensorMetadata` is a view of the serialized tensor metadata which contains
// pertinent information about a tensor within a PyTorch program.
class TensorMetadata : public View<flatflow::TensorMetadata> {
  using Base = typename TensorMetadata::View;

 public:
  using Base::Base;

  ScalarType dtype() const noexcept { return get()->dtype(); }

  Vector<SymInt> shape() const noexcept { return get()->shape(); }
};

}  // namespace detail
}  // namespace flatflow

#endif  // FLATFLOW_OPS_VIEW_H_
