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
  auto sum = flatflow::internal::polynomial<int64_t>(4, 3, 2, 1) + 3;
  EXPECT_EQ(sum, flatflow::internal::polynomial<int64_t>(7, 3, 2, 1));

  sum = flatflow::internal::polynomial<int64_t>(4, 3, 2, 1) +
        flatflow::internal::polynomial<int64_t>(1, 2, 0, 0);
  EXPECT_EQ(sum, flatflow::internal::polynomial<int64_t>(5, 5, 2, 1));
}

TEST(PolynomialTest, AdditionIdentity) {
  const auto p = flatflow::internal::polynomial<int64_t>(1, 2, 3, 4);
  EXPECT_EQ(p + 0, p);

  const auto zero = flatflow::internal::polynomial<int64_t>();
  EXPECT_EQ(p + zero, p);
}

TEST(PolynomialTest, AdditionInverse) {
  const auto p = flatflow::internal::polynomial<int64_t>(1, 2, 3, 4);
  const auto inverse = flatflow::internal::polynomial<int64_t>(-1, -2, -3, -4);
  const auto zero = flatflow::internal::polynomial<int64_t>();
  EXPECT_EQ(p + inverse, zero);
}

TEST(PolynomialTest, Subtraction) {
  auto diff = flatflow::internal::polynomial<int64_t>(4, 3, 2, 1) - 3;
  EXPECT_EQ(diff, flatflow::internal::polynomial<int64_t>(1, 3, 2, 1));

  diff = flatflow::internal::polynomial<int64_t>(4, 3, 2, 1) -
         flatflow::internal::polynomial<int64_t>(1, 2, 0, 0);
  EXPECT_EQ(diff, flatflow::internal::polynomial<int64_t>(3, 1, 2, 1));
}

TEST(PolynomialTest, SubtractionIdentity) {
  const auto p = flatflow::internal::polynomial<int64_t>(1, 2, 3, 4);
  EXPECT_EQ(p - 0, p);

  const auto zero = flatflow::internal::polynomial<int64_t>();
  EXPECT_EQ(p - zero, p);
}

TEST(PolynomialTest, SubtractionInverse) {
  const auto p = flatflow::internal::polynomial<int64_t>(1, 2, 3, 4);
  const auto inverse = p;
  const auto zero = flatflow::internal::polynomial<int64_t>();
  EXPECT_EQ(p - inverse, zero);
}

TEST(PolynomialTest, Multiplication) {
  auto product = flatflow::internal::polynomial<int64_t>(5, 8, 13, 21) * 3;
  EXPECT_EQ(product, flatflow::internal::polynomial<int64_t>(15, 24, 39, 63));

  product = flatflow::internal::polynomial<int64_t>(5, 8, 13, 21) *
            flatflow::internal::polynomial<int64_t>(1, 1, 2, 3);
  EXPECT_EQ(product, flatflow::internal::polynomial<int64_t>(5, 13, 31, 65));
}

TEST(PolynomialTest, MultiplicationIdentity) {
  const auto p = flatflow::internal::polynomial<int64_t>(5, 8, 13, 21);
  EXPECT_EQ(p * 1, p);

  const auto one = flatflow::internal::polynomial<int64_t>(1);
  EXPECT_EQ(p * one, p);
}

TEST(PolynomialTest, Division) {
  const auto quotient =
      flatflow::internal::polynomial<int64_t>(15, 24, 39, 63) / 3;
  EXPECT_EQ(quotient, flatflow::internal::polynomial<int64_t>(5, 8, 13, 21));
}

TEST(PolynomialTest, DivisionIdentity) {
  const auto p = flatflow::internal::polynomial<int64_t>(15, 24, 39, 63);
  EXPECT_EQ(p / 1, p);
}

TEST(PolynomialTest, ShiftLeft) {
  const auto p = flatflow::internal::polynomial<int64_t>(5, 8, 13, 21);
  EXPECT_EQ(p << 1, p * 2);
  EXPECT_EQ(p << 2, p * 4);
  EXPECT_EQ(p << 3, p * 8);
  EXPECT_EQ(p << 4, p * 16);
}

TEST(PolynomialTest, ShiftLeftIdentity) {
  const auto p = flatflow::internal::polynomial<int64_t>(5, 8, 13, 21);
  EXPECT_EQ(p << 0, p);
}

TEST(PolynomialTest, ShiftRight) {
  const auto p = flatflow::internal::polynomial<int64_t>(80, 128, 208, 336);
  EXPECT_EQ(p >> 1, p / 2);
  EXPECT_EQ(p >> 2, p / 4);
  EXPECT_EQ(p >> 3, p / 8);
  EXPECT_EQ(p >> 4, p / 16);
}

TEST(PolynomialTest, ShiftRightIdentity) {
  const auto p = flatflow::internal::polynomial<int64_t>(80, 128, 208, 336);
  EXPECT_EQ(p >> 0, p);
}

TEST(PolynomialTest, Size) {
  const auto p = flatflow::internal::polynomial<int64_t>();
  EXPECT_EQ(p.size(), 4);
}

TEST(PolynomialTest, Degree) {
  const auto p = flatflow::internal::polynomial<int64_t>();
  EXPECT_EQ(p.degree(), 3);
}

TEST(PolynomialTest, EvaluatePolynomial) {
  const auto p = flatflow::internal::polynomial<int64_t>(80, 128, 208, 336);
  EXPECT_EQ(flatflow::internal::evaluate_polynomial(p, 0), 80);
  EXPECT_EQ(p(0), 80);
  EXPECT_EQ(flatflow::internal::evaluate_polynomial(p, 1), 752);
  EXPECT_EQ(p(1), 752);
  EXPECT_EQ(flatflow::internal::evaluate_polynomial(p, 34), 13451024);
  EXPECT_EQ(p(34), 13451024);
}

TEST(PolynomialTest, EvaluatePolynomialIdentity) {
  const auto identity = flatflow::internal::polynomial<int64_t>(0, 1, 0, 0);
  EXPECT_EQ(flatflow::internal::evaluate_polynomial(identity, 55), 55);
  EXPECT_EQ(identity(55), 55);
}

}  // namespace
