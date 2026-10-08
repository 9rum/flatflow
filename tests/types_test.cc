// SPDX-License-Identifier: Apache-2.0

#include "flatflow/types.h"

#include <array>
#include <cstdint>
#include <type_traits>

#include "flatbuffers/array.h"

using flatflow::extent_v;
using flatflow::remove_cvptr_t;
using flatflow::remove_extent_t;

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

// Tests whether `remove_extent` leaves pointers to pointers to FlatBuffers
// arrays unchanged, as `std::remove_extent` does.
static_assert(std::is_same_v<
              remove_extent_t<const flatbuffers::Array<int64_t, 2> **>,
              std::remove_extent_t<const flatbuffers::Array<int64_t, 2> **>>);

// Tests whether `extent` is consistent with `std::extent` for types other than
// FlatBuffers arrays.
static_assert(extent_v<int> == std::extent_v<int>);
static_assert(extent_v<int *> == std::extent_v<int *>);
static_assert(extent_v<int[3]> == std::extent_v<int[3]>);
static_assert(extent_v<const int[3]> == std::extent_v<const int[3]>);
static_assert(extent_v<int[]> == std::extent_v<int[]>);
static_assert(extent_v<int[2][3]> == std::extent_v<int[2][3]>);
static_assert(extent_v<int[2][3], 1> == std::extent_v<int[2][3], 1>);
static_assert(extent_v<int[2][3], 2> == std::extent_v<int[2][3], 2>);
static_assert(extent_v<int[][3], 1> == std::extent_v<int[][3], 1>);
static_assert(extent_v<int (*)[3]> == std::extent_v<int (*)[3]>);

// Tests whether `extent` yields the number of elements of FlatBuffers arrays,
// with or without (possibly cv-qualified) pointers.
static_assert(extent_v<flatbuffers::Array<int64_t, 2>> == 2);
static_assert(extent_v<const flatbuffers::Array<int64_t, 2>> == 2);
static_assert(extent_v<flatbuffers::Array<int64_t, 2> *> == 2);
static_assert(extent_v<const flatbuffers::Array<int64_t, 2> *> == 2);
static_assert(extent_v<const flatbuffers::Array<int64_t, 2> *const> == 2);

// Tests whether `extent` yields zero for dimensions other than the first of
// FlatBuffers arrays.
static_assert(extent_v<flatbuffers::Array<int64_t, 2>, 1> == 0);
static_assert(extent_v<const flatbuffers::Array<int64_t, 2> *, 1> == 0);

// Tests whether `extent` leaves pointers to pointers to FlatBuffers arrays
// unchanged, as `std::extent` does.
static_assert(extent_v<const flatbuffers::Array<int64_t, 2> **> ==
              std::extent_v<const flatbuffers::Array<int64_t, 2> **>);

// Tests whether `remove_extent` and `extent` together reconstruct the shape of
// a FlatBuffers array as a `std::array`, which is how `SymInt` declares its
// underlying storage.
static_assert(
    std::is_same_v<
        std::array<remove_extent_t<const flatbuffers::Array<int64_t, 2> *>,
                   extent_v<const flatbuffers::Array<int64_t, 2> *>>,
        std::array<int64_t, 2>>);
