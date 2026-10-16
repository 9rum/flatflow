// SPDX-License-Identifier: Apache-2.0

#include "flatflow/types.h"

#include <cstdint>
#include <type_traits>

#include "flatbuffers/array.h"

namespace flatflow {
namespace {

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

// Tests whether `remove_extent` is consistent with `std::remove_extent` for
// types other than FlatBuffers arrays.
static_assert(std::is_same_v<remove_extent_t<int>, std::remove_extent_t<int>>);
static_assert(std::is_same_v<remove_extent_t<const int>,
                             std::remove_extent_t<const int>>);
static_assert(
    std::is_same_v<remove_extent_t<int *>, std::remove_extent_t<int *>>);
static_assert(std::is_same_v<remove_extent_t<const int *>,
                             std::remove_extent_t<const int *>>);
static_assert(
    std::is_same_v<remove_extent_t<int[3]>, std::remove_extent_t<int[3]>>);
static_assert(std::is_same_v<remove_extent_t<const int[3]>,
                             std::remove_extent_t<const int[3]>>);
static_assert(
    std::is_same_v<remove_extent_t<int[]>, std::remove_extent_t<int[]>>);
static_assert(std::is_same_v<remove_extent_t<int[2][3]>,
                             std::remove_extent_t<int[2][3]>>);
static_assert(std::is_same_v<remove_extent_t<int (*)[3]>,
                             std::remove_extent_t<int (*)[3]>>);

// Tests whether `remove_extent` yields the element type of FlatBuffers arrays,
// with or without (possibly cv-qualified) pointers.
static_assert(
    std::is_same_v<remove_extent_t<flatbuffers::Array<int64_t, 2>>, int64_t>);
static_assert(std::is_same_v<
              remove_extent_t<const flatbuffers::Array<int64_t, 2>>, int64_t>);
static_assert(
    std::is_same_v<remove_extent_t<flatbuffers::Array<int64_t, 2> *>, int64_t>);
static_assert(
    std::is_same_v<remove_extent_t<const flatbuffers::Array<int64_t, 2> *>,
                   int64_t>);
static_assert(
    std::is_same_v<remove_extent_t<const flatbuffers::Array<int64_t, 2> *const>,
                   int64_t>);

}  // namespace
}  // namespace flatflow
