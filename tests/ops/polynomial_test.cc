// SPDX-License-Identifier: Apache-2.0

#include "flatflow/ops/polynomial.h"

#include <cstdint>

#include "flatflow/ops/graph.h"
#include "flatflow/ops/scalar_type_generated.h"

using flatflow::Polynomial;
using flatflow::polynomial;
using flatflow::ScalarType;
using flatflow::SymInt;

// Tests whether `Polynomial` is constructed as intended.
static_assert(Polynomial(1, 2, 3) == Polynomial{1, 2, 3});

static_assert(Polynomial() == Polynomial{0, 0, 0});
static_assert(Polynomial() == Polynomial(0, 0, 0));

static_assert(Polynomial(2) == Polynomial{2, 0, 0});
static_assert(Polynomial(3) == Polynomial(3, 0, 0));
static_assert(Polynomial(0) == Polynomial());

static_assert(polynomial() == Polynomial());
static_assert(polynomial(1) == Polynomial(1));
static_assert(polynomial(1, 2) == Polynomial{1, 2, 0});
static_assert(polynomial(1, 2, 3) == Polynomial{1, 2, 3});

static_assert(polynomial(1, 2, 4).with_tensor_parallel(2) ==
              polynomial(1, 1, 4));
static_assert(polynomial(1, 2, 4).with_tensor_parallel(8) ==
              polynomial(1, 2, 32));
static_assert(polynomial().with_tensor_parallel(8) == polynomial());

static_assert(polynomial(1, 2, 4).with_context_parallel(2) ==
              polynomial(1, 2, 2));
static_assert(polynomial(1, 2, 4).with_context_parallel(8) ==
              polynomial(1, 16, 4));
static_assert(polynomial().with_context_parallel(8) == polynomial());

// Tests whether a polynomial constructed from a symbolic integer has the same
// coefficients taken from it.
static_assert(polynomial(SymInt()) == 0);
static_assert(polynomial(SymInt(4096)) == 4096);
static_assert(polynomial(SymInt(4096, 0)) == 4096);
static_assert(polynomial(SymInt{0, 1}) == polynomial(0, 1, 0));
static_assert(polynomial(SymInt(-1, 2)) == polynomial(-1, 2, 0));
static_assert(polynomial(SymInt(0, 1)) == polynomial(0, 1, 0));

// Tests whether `polynomial` returns a constant polynomial of the expected
// scale factor for each data type.
static_assert(polynomial(ScalarType::float32) == 4);
static_assert(polynomial(ScalarType::float64) == 64);
static_assert(polynomial(ScalarType::float16) == 2);
static_assert(polynomial(ScalarType::bfloat16) == 2);
static_assert(polynomial(ScalarType::bool_) == 64);
static_assert(polynomial(ScalarType::int8) == 1);
static_assert(polynomial(ScalarType::int16) == 64);
static_assert(polynomial(ScalarType::int32) == 64);
static_assert(polynomial(ScalarType::int64) == 128);
static_assert(polynomial(ScalarType::uint8) == 1);
static_assert(polynomial(ScalarType::uint16) == 64);
static_assert(polynomial(ScalarType::uint32) == 64);
static_assert(polynomial(ScalarType::uint64) == 128);
static_assert(polynomial(ScalarType::float8_e4m3fn) == 1);
static_assert(polynomial(ScalarType::float8_e4m3fnuz) == 1);
static_assert(polynomial(ScalarType::float8_e5m2) == 1);
static_assert(polynomial(ScalarType::float8_e5m2fnuz) == 1);

// Tests arithmetic between a polynomial and a scalar.
static_assert(polynomial(12, 18, 30) + 0 == polynomial(12, 18, 30));
static_assert(0 + polynomial(12, 18, 30) == polynomial(12, 18, 30));
static_assert(polynomial(12, 18, 30) + 3 == polynomial(15, 18, 30));
static_assert(3 + polynomial(12, 18, 30) == polynomial(12, 18, 30) + 3);
static_assert(polynomial(12, 18, 30) - 0 == polynomial(12, 18, 30));
static_assert(0 - polynomial(12, 18, 30) == -polynomial(12, 18, 30));
static_assert(polynomial(12, 18, 30) - 3 == polynomial(9, 18, 30));
static_assert(3 - polynomial(12, 18, 30) == polynomial(-9, -18, -30));

static_assert(polynomial(12, 18, 30) * 1 == polynomial(12, 18, 30));
static_assert(1 * polynomial(12, 18, 30) == polynomial(12, 18, 30));
static_assert(polynomial(12, 18, 30) * 2 == polynomial(24, 36, 60));
static_assert(2 * polynomial(12, 18, 30) == polynomial(12, 18, 30) * 2);
static_assert(polynomial(12, 18, 30) / 1 == polynomial(12, 18, 30));
static_assert(polynomial(12, 18, 30) / 3 == polynomial(4, 6, 10));

inline constexpr auto kScalarAdded = [](Polynomial expr) {
  expr += 1;
  return expr;
}(polynomial(16, 24, 32));
static_assert(kScalarAdded == polynomial(17, 24, 32));

inline constexpr auto kScalarSubtracted = [](Polynomial expr) {
  expr -= 1;
  return expr;
}(kScalarAdded);
static_assert(kScalarSubtracted == polynomial(16, 24, 32));

inline constexpr auto kScalarMultiplied = [](Polynomial expr) {
  expr *= 3;
  return expr;
}(kScalarSubtracted);
static_assert(kScalarMultiplied == polynomial(48, 72, 96));

inline constexpr auto kScalarDivided = [](Polynomial expr) {
  expr /= 4;
  return expr;
}(kScalarMultiplied);
static_assert(kScalarDivided == polynomial(12, 18, 24));

// Tests arithmetic between polynomials.
static_assert(polynomial(3, 2, 1) + polynomial() == polynomial(3, 2, 1));
static_assert(polynomial(3, 2, 1) + -polynomial(3, 2, 1) == polynomial());
static_assert(polynomial(3, 2, 1) + polynomial(1, 2, 0) == polynomial(4, 4, 1));

static_assert(polynomial(3, 2, 1) - polynomial() == polynomial(3, 2, 1));
static_assert(polynomial(3, 2, 1) - polynomial(3, 2, 1) == polynomial());
static_assert(polynomial(3, 2, 1) - polynomial(1, 2, 0) == polynomial(2, 0, 1));

static_assert(-polynomial() == polynomial());
static_assert(-polynomial(3, 2, 1) == polynomial(-3, -2, -1));

inline constexpr auto kAdded = [](Polynomial expr) {
  expr += polynomial(6, 9, 15);
  return expr;
}(polynomial(16, 24, 32));
static_assert(kAdded == polynomial(22, 33, 47));

inline constexpr auto kSubtracted = [](Polynomial expr) {
  expr -= polynomial(4, 6, 10);
  return expr;
}(kAdded);
static_assert(kSubtracted == polynomial(18, 27, 37));

static_assert(polynomial(5, 8, 13) * polynomial() == polynomial());
static_assert(polynomial(5, 8, 13) * polynomial(1) == polynomial(5, 8, 13));
static_assert(polynomial(5, 8, 13) * polynomial(1, 2) == polynomial(5, 18, 29));
static_assert(polynomial(5, 8, 13) * polynomial(1, 2, 3) ==
              polynomial(5, 18, 44));

inline constexpr auto kMultiplied = [](Polynomial expr) {
  expr *= polynomial(1, 2, 3);
  return expr;
}(polynomial(5, 8, 13));
static_assert(kMultiplied == polynomial(5, 8, 13) * polynomial(1, 2, 3));

inline constexpr auto kSquared = [](Polynomial expr) {
  expr *= expr;
  return expr;
}(polynomial(5, 8, 13));
static_assert(kSquared == polynomial(5, 8, 13) * polynomial(5, 8, 13));

// Tests whether `normalized` makes the constant term zero and the rest
// relatively prime.
static_assert(polynomial(5, 10, 20).normalized() == polynomial(0, 1, 2));
static_assert(polynomial().normalized() == polynomial());
static_assert(polynomial(-1, -2, -4).normalized() == polynomial(0, -1, -2));
static_assert(polynomial(INT64_MIN, INT64_MIN, INT64_MIN).normalized() ==
              polynomial(0, -1, -1));
static_assert(polynomial(INT64_MIN, INT64_MIN).normalized() ==
              polynomial(0, -1));
static_assert(polynomial(0, INT64_MIN, INT64_MAX).normalized() ==
              polynomial(0, INT64_MIN, INT64_MAX));

// Tests whether a polynomial is evaluated at the given value.
static_assert(polynomial(0, 2, 1)(0) == 0);
static_assert(polynomial(0, 2, 1)(1) == 3);
static_assert(polynomial(0, 2, 1)(2) == 8);
static_assert(polynomial(INT64_MIN, INT64_MIN, INT64_MIN).normalized()(0) == 0);
static_assert(polynomial(INT64_MIN, INT64_MIN, INT64_MIN).normalized()(1) ==
              -2);
static_assert(polynomial(INT64_MIN, INT64_MIN, INT64_MIN).normalized()(2) ==
              -6);
static_assert(polynomial()(INT64_MAX) == 0);
