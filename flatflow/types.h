// SPDX-License-Identifier: Apache-2.0

#ifndef FLATFLOW_TYPES_H_
#define FLATFLOW_TYPES_H_

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

// If `T` is an array or a FlatBuffers array, provides the member typedef `type`
// equal to its element type, otherwise `type` is `T`.
template <typename T, typename = remove_cvptr_t<T>>
struct remove_extent : public std::remove_extent<T> {};

template <typename _, typename T, auto N>
struct remove_extent<_, flatbuffers::Array<T, N>>
    : public std::type_identity<T> {};

// Alias template for `remove_extent`.
template <typename T>
using remove_extent_t = remove_extent<T>::type;

}  // namespace flatflow

#endif  // FLATFLOW_TYPES_H_
