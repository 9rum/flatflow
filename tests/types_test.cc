// SPDX-License-Identifier: Apache-2.0

#include "flatflow/types.h"

#include <array>
#include <cstdint>
#include <type_traits>

#include "flatbuffers/array.h"

using flatflow::make_array_t;
using flatflow::remove_cvptr_t;

// Tests whether `remove_cvptr` removes top-level cv-qualifiers from types other
// than pointers.
static_assert(std::is_same_v<remove_cvptr_t<int>, int>);
static_assert(std::is_same_v<remove_cvptr_t<const int>, int>);
static_assert(std::is_same_v<remove_cvptr_t<volatile int>, int>);
static_assert(std::is_same_v<remove_cvptr_t<const volatile int>, int>);

// Tests whether `remove_cvptr` removes the pointer along with cv-qualifiers of
// both the pointer and the pointed-to type.
static_assert(std::is_same_v<remove_cvptr_t<int *>, int>);
static_assert(std::is_same_v<remove_cvptr_t<const int *>, int>);
static_assert(std::is_same_v<remove_cvptr_t<volatile int *>, int>);
static_assert(std::is_same_v<remove_cvptr_t<const volatile int *>, int>);
static_assert(std::is_same_v<remove_cvptr_t<int *const>, int>);
static_assert(std::is_same_v<remove_cvptr_t<const int *const volatile>, int>);
static_assert(
    std::is_same_v<remove_cvptr_t<const flatbuffers::Array<int64_t, 2> *>,
                   flatbuffers::Array<int64_t, 2>>);

// Tests whether `remove_cvptr` removes only one level of indirection.
static_assert(std::is_same_v<remove_cvptr_t<int **>, int *>);
static_assert(std::is_same_v<remove_cvptr_t<const int *const *>, const int *>);

// Tests whether `make_array` provides the member typedef `type` equal to the
// `std::array` counterpart of FlatBuffers arrays, with or without (possibly
// cv-qualified) pointers.
static_assert(std::is_same_v<make_array_t<flatbuffers::Array<int64_t, 2>>,
                             std::array<int64_t, 2>>);
static_assert(std::is_same_v<make_array_t<const flatbuffers::Array<int64_t, 2>>,
                             std::array<int64_t, 2>>);
static_assert(std::is_same_v<make_array_t<flatbuffers::Array<int64_t, 2> *>,
                             std::array<int64_t, 2>>);
static_assert(
    std::is_same_v<make_array_t<const flatbuffers::Array<int64_t, 2> *>,
                   std::array<int64_t, 2>>);
static_assert(
    std::is_same_v<make_array_t<const flatbuffers::Array<int64_t, 2> *const>,
                   std::array<int64_t, 2>>);
