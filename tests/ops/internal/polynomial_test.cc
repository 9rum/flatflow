// Copyright 2025 The FlatFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "flatflow/ops/internal/polynomial.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace {

TEST(PolynomialTest, Addition) {
  auto sum = flatflow::internal::polynomial<int64_t>(3, 2, 1) + 3;
  EXPECT_EQ(sum, flatflow::internal::polynomial<int64_t>(6, 2, 1));

  sum = flatflow::internal::polynomial<int64_t>(3, 2, 1) +
        flatflow::internal::polynomial<int64_t>(1, 2, 0);
  EXPECT_EQ(sum, flatflow::internal::polynomial<int64_t>(4, 4, 1));
}

TEST(PolynomialTest, AdditionIdentity) {
  const auto poly = flatflow::internal::polynomial<int64_t>(1, 2, 3);
  EXPECT_EQ(poly + 0, poly);

  const auto zero = flatflow::internal::polynomial<int64_t>();
  EXPECT_EQ(poly + zero, poly);
}

TEST(PolynomialTest, AdditionInverse) {
  const auto poly = flatflow::internal::polynomial<int64_t>(1, 2, 3);
  const auto inverse = flatflow::internal::polynomial<int64_t>(-1, -2, -3);
  const auto zero = flatflow::internal::polynomial<int64_t>();
  EXPECT_EQ(poly + inverse, zero);
}

TEST(PolynomialTest, Subtraction) {
  auto diff = flatflow::internal::polynomial<int64_t>(4, 3, 2) - 3;
  EXPECT_EQ(diff, flatflow::internal::polynomial<int64_t>(1, 3, 2));

  diff = flatflow::internal::polynomial<int64_t>(4, 3, 2) -
         flatflow::internal::polynomial<int64_t>(1, 2, 0);
  EXPECT_EQ(diff, flatflow::internal::polynomial<int64_t>(3, 1, 2));
}

TEST(PolynomialTest, SubtractionIdentity) {
  const auto poly = flatflow::internal::polynomial<int64_t>(1, 2, 3);
  EXPECT_EQ(poly - 0, poly);

  const auto zero = flatflow::internal::polynomial<int64_t>();
  EXPECT_EQ(poly - zero, poly);
}

TEST(PolynomialTest, SubtractionInverse) {
  const auto poly = flatflow::internal::polynomial<int64_t>(1, 2, 3);
  const auto inverse = poly;
  const auto zero = flatflow::internal::polynomial<int64_t>();
  EXPECT_EQ(poly - inverse, zero);
}

TEST(PolynomialTest, Multiplication) {
  auto product = flatflow::internal::polynomial<int64_t>(5, 8, 13) * 3;
  EXPECT_EQ(product, flatflow::internal::polynomial<int64_t>(15, 24, 39));

  product = flatflow::internal::polynomial<int64_t>(5, 8, 13) *
            flatflow::internal::polynomial<int64_t>(1, 2, 3);
  EXPECT_EQ(product, flatflow::internal::polynomial<int64_t>(5, 18, 44));
}

TEST(PolynomialTest, MultiplicationIdentity) {
  const auto poly = flatflow::internal::polynomial<int64_t>(5, 8, 13);
  EXPECT_EQ(poly * 1, poly);

  const auto one = flatflow::internal::polynomial<int64_t>(1);
  EXPECT_EQ(poly * one, poly);
}

TEST(PolynomialTest, Division) {
  const auto quotient = flatflow::internal::polynomial<int64_t>(15, 24, 39) / 3;
  EXPECT_EQ(quotient, flatflow::internal::polynomial<int64_t>(5, 8, 13));
}

TEST(PolynomialTest, DivisionIdentity) {
  const auto poly = flatflow::internal::polynomial<int64_t>(15, 24, 39);
  EXPECT_EQ(poly / 1, poly);
}

TEST(PolynomialTest, ShiftLeft) {
  const auto poly = flatflow::internal::polynomial<int64_t>(5, 8, 13);
  EXPECT_EQ(poly << 1, poly * 2);
  EXPECT_EQ(poly << 2, poly * 4);
  EXPECT_EQ(poly << 3, poly * 8);
  EXPECT_EQ(poly << 4, poly * 16);
}

TEST(PolynomialTest, ShiftLeftIdentity) {
  const auto poly = flatflow::internal::polynomial<int64_t>(5, 8, 13);
  EXPECT_EQ(poly << 0, poly);
}

TEST(PolynomialTest, ShiftRight) {
  const auto poly = flatflow::internal::polynomial<int64_t>(80, 128, 208);
  EXPECT_EQ(poly >> 1, poly / 2);
  EXPECT_EQ(poly >> 2, poly / 4);
  EXPECT_EQ(poly >> 3, poly / 8);
  EXPECT_EQ(poly >> 4, poly / 16);
}

TEST(PolynomialTest, ShiftRightIdentity) {
  const auto poly = flatflow::internal::polynomial<int64_t>(80, 128, 208);
  EXPECT_EQ(poly >> 0, poly);
}

TEST(PolynomialTest, Size) {
  const auto poly = flatflow::internal::polynomial<int64_t>();
  EXPECT_EQ(poly.size(), 3);
}

TEST(PolynomialTest, Degree) {
  const auto poly = flatflow::internal::polynomial<int64_t>();
  EXPECT_EQ(poly.degree(), 2);
}

TEST(PolynomialTest, EvaluatePolynomial) {
  const auto poly = flatflow::internal::polynomial<int64_t>(80, 128, 208);
  EXPECT_EQ(flatflow::internal::evaluate_polynomial(poly, 0), 80);
  EXPECT_EQ(poly(0), 80);
  EXPECT_EQ(flatflow::internal::evaluate_polynomial(poly, 1), 416);
  EXPECT_EQ(poly(1), 416);
  EXPECT_EQ(flatflow::internal::evaluate_polynomial(poly, 34), 244880);
  EXPECT_EQ(poly(34), 244880);
}

TEST(PolynomialTest, EvaluatePolynomialIdentity) {
  const auto identity = flatflow::internal::polynomial<int64_t>(0, 1, 0);
  EXPECT_EQ(flatflow::internal::evaluate_polynomial(identity, 55), 55);
  EXPECT_EQ(identity(55), 55);
}

}  // namespace
