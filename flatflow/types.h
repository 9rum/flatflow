// SPDX-License-Identifier: Apache-2.0

#ifndef FLATFLOW_TYPES_H_
#define FLATFLOW_TYPES_H_

#include <type_traits>

namespace flatflow {

// If the type `T` is a pointer type, provides the member typedef `type` which
// is the type pointed to by `T` with its topmost cv-qualifiers removed.
// Otherwise `type` is `T` with its topmost cv-qualifiers removed.
template <typename T>
struct remove_cvptr {
  // The type pointed by `T` or `T` itself if it is not a pointer, with
  // top-level cv-qualifiers removed.
  using type = std::remove_cv_t<std::remove_pointer_t<T>>;
};

// Alias template for `remove_cvptr`.
template <typename T>
using remove_cvptr_t = typename remove_cvptr<T>::type;

}  // namespace flatflow

#endif  // FLATFLOW_TYPES_H_
