// SPDX-License-Identifier: Apache-2.0

#include "flatflow/ops/numeric.h"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

using flatflow::gcd;
using flatflow::signum;

// Tests whether `gcd` returns the unsigned counterpart of the common type.
static_assert(std::is_same_v<decltype(gcd(std::declval<std::int8_t>(),
                                          std::declval<std::int8_t>())),
                             std::uint8_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<std::uint8_t>(),
                                          std::declval<std::uint8_t>())),
                             std::uint8_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<std::int8_t>(),
                                          std::declval<std::uint8_t>())),
                             unsigned int>);

static_assert(std::is_same_v<decltype(gcd(std::declval<std::int16_t>(),
                                          std::declval<std::int16_t>())),
                             std::uint16_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<std::uint16_t>(),
                                          std::declval<std::uint16_t>())),
                             std::uint16_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<std::int16_t>(),
                                          std::declval<std::uint16_t>())),
                             unsigned int>);

static_assert(std::is_same_v<decltype(gcd(std::declval<std::int32_t>(),
                                          std::declval<std::int32_t>())),
                             std::uint32_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<std::uint32_t>(),
                                          std::declval<std::uint32_t>())),
                             std::uint32_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<std::int32_t>(),
                                          std::declval<std::uint32_t>())),
                             std::uint32_t>);

static_assert(std::is_same_v<decltype(gcd(std::declval<std::int64_t>(),
                                          std::declval<std::int64_t>())),
                             std::uint64_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<std::uint64_t>(),
                                          std::declval<std::uint64_t>())),
                             std::uint64_t>);
static_assert(std::is_same_v<decltype(gcd(std::declval<std::int64_t>(),
                                          std::declval<std::uint64_t>())),
                             std::uint64_t>);

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

static_assert(gcd(std::numeric_limits<std::int64_t>::max(), 0) ==
              std::numeric_limits<std::int64_t>::max());
static_assert(gcd(std::numeric_limits<std::int64_t>::max(), 1) == 1);
static_assert(gcd(std::numeric_limits<std::int64_t>::max(),
                  std::numeric_limits<std::int64_t>::max()) ==
              std::numeric_limits<std::int64_t>::max());
static_assert(gcd(-std::numeric_limits<std::int64_t>::max(),
                  std::numeric_limits<std::int64_t>::max()) ==
              std::numeric_limits<std::int64_t>::max());

// Tests whether `gcd` returns the greatest common divisor where the behavior of
// `std::gcd` is undefined. This happens if and only if either argument is the
// minimum representable value of the common type.
static_assert(gcd(std::numeric_limits<std::int64_t>::min(), 0) ==
              std::numeric_limits<std::int64_t>::min());
static_assert(gcd(std::numeric_limits<std::int64_t>::min(), 1) == 1);
static_assert(gcd(std::numeric_limits<std::int64_t>::min(),
                  std::numeric_limits<std::int64_t>::min()) ==
              std::numeric_limits<std::int64_t>::min());
static_assert(gcd(std::numeric_limits<std::int64_t>::min(),
                  std::numeric_limits<std::int64_t>::max()) == 1);
static_assert(gcd(std::numeric_limits<std::int64_t>::min(),
                  -std::numeric_limits<std::int64_t>::max()) == 1);
static_assert(gcd(std::numeric_limits<std::int64_t>::min(),
                  0x4000000000000000) == 0x4000000000000000);

// Tests whether `signum` returns the expected sign of the given integer.
static_assert(signum(0) == 0);
static_assert(signum(0x4000000000000000) == 1);
static_assert(signum(-0x4000000000000000) == -1);
static_assert(signum(std::numeric_limits<std::int64_t>::max()) == 1);
static_assert(signum(std::numeric_limits<std::int64_t>::min()) == -1);
