// SPDX-License-Identifier: Apache-2.0

#ifndef FLATFLOW_TYPES_H_
#define FLATFLOW_TYPES_H_

#include <array>
#include <type_traits>

#include "flatbuffers/array.h"

namespace flatflow {

// If the type `T` is a pointer type, provides the member typedef `type` which
// is the type pointed to by `T` with its topmost cv-qualifiers removed.
// Otherwise `type` is `T` with its topmost cv-qualifiers removed.
template <typename T>
struct remove_cvptr {
  // The type pointed to by `T` or `T` itself if it is not a pointer, with
  // top-level cv-qualifiers removed.
  using type = std::remove_cv_t<std::remove_pointer_t<T>>;
};

// Alias template for `remove_cvptr`.
template <typename T>
using remove_cvptr_t = remove_cvptr<T>::type;

// If `T` is a FlatBuffers array or a (possibly cv-qualified) pointer to it,
// provides the member typedef `type` equal to its `std::array` counterpart,
// i.e., `std::array<T, N>` for `flatbuffers::Array<T, N>`. Otherwise `type` is
// not provided.
template <typename T, typename = remove_cvptr_t<T>>
struct make_array {};

template <typename _, typename T, auto N>
struct make_array<_, flatbuffers::Array<T, N>> {
  using type = std::array<T, N>;
};

// Alias template for `make_array`.
template <typename T>
using make_array_t = make_array<T>::type;

}  // namespace flatflow

#endif  // FLATFLOW_TYPES_H_
