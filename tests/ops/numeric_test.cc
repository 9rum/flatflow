// SPDX-License-Identifier: Apache-2.0

#include "flatflow/ops/numeric.h"

#include <cstdint>
#include <type_traits>
#include <utility>

using flatflow::gcd;
using flatflow::signum;

// Tests whether `gcd` returns the unsigned counterpart of the common type.
static_assert(std::is_same_v<decltype(gcd(std::declval<int8_t>(),
                                          std::declval<int8_t>())),
                             uint8_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<uint8_t>(),
                                          std::declval<uint8_t>())),
                             uint8_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<int8_t>(),
                                          std::declval<uint8_t>())),
                             unsigned int>);

static_assert(std::is_same_v<decltype(gcd(std::declval<int16_t>(),
                                          std::declval<int16_t>())),
                             uint16_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<uint16_t>(),
                                          std::declval<uint16_t>())),
                             uint16_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<int16_t>(),
                                          std::declval<uint16_t>())),
                             unsigned int>);

static_assert(std::is_same_v<decltype(gcd(std::declval<int32_t>(),
                                          std::declval<int32_t>())),
                             uint32_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<uint32_t>(),
                                          std::declval<uint32_t>())),
                             uint32_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<int32_t>(),
                                          std::declval<uint32_t>())),
                             uint32_t>);

static_assert(std::is_same_v<decltype(gcd(std::declval<int64_t>(),
                                          std::declval<int64_t>())),
                             uint64_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<uint64_t>(),
                                          std::declval<uint64_t>())),
                             uint64_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<int64_t>(),
                                          std::declval<uint64_t>())),
                             uint64_t>);

// Tests whether `gcd` returns the greatest common divisor where the behavior of
// `std::gcd` is defined.
static_assert(gcd(12, 18) == 6);
static_assert(gcd(18, 12) == 6);
static_assert(gcd(6, 10) == 2);
static_assert(gcd(10, 6) == 2);
static_assert(gcd(6, -10) == 2);
static_assert(gcd(-10, 6) == 2);
static_assert(gcd(-6, -10) == 2);
static_assert(gcd(-10, -6) == 2);
static_assert(gcd(24, 0) == 24);
static_assert(gcd(0, 24) == 24);
static_assert(gcd(-24, 0) == 24);
static_assert(gcd(0, -24) == 24);
static_assert(gcd(0, 0) == 0);

static_assert(gcd(INT64_MAX, 0) == INT64_MAX);
static_assert(gcd(INT64_MAX, 1) == 1);
static_assert(gcd(INT64_MAX, INT64_MAX) == INT64_MAX);

// Tests whether `gcd` returns the greatest common divisor where the behavior of
// `std::gcd` is undefined. This happens if and only if either argument is the
// minimum representable value of the common type.
static_assert(gcd(INT64_MIN, 0) == 0x8000000000000000);
static_assert(gcd(INT64_MIN, 1) == 1);
static_assert(gcd(INT64_MIN, INT64_MIN) == 0x8000000000000000);
static_assert(gcd(INT64_MIN, INT64_MAX) == 1);

// Tests whether `signum` returns the expected sign of the given integer.
static_assert(signum(0) == 0);
static_assert(signum(1) == 1);
static_assert(signum(-1) == -1);
static_assert(signum(INT64_MAX) == 1);
static_assert(signum(INT64_MIN) == -1);
