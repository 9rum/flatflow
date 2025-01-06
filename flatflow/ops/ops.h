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

#ifndef FLATFLOW_OPS_OPS_H_
#define FLATFLOW_OPS_OPS_H_

#include <cstdint>
#include <execution>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "flatbuffers/base.h"
#include "flatbuffers/vector.h"

#include "flatflow/ops/operator_generated.h"

namespace flatflow {

// flatflow::OpFLOPs
//
// A `flatflow::OpFLOPs` represents the specification of operator FLOPs
// (floating point operations). These specifications are created by measuring
// the absolute number of floating point operations required by each operator
// or MACs (multiply-accumulates) for FMA instructions, for a given size such
// as sequence length.
class OpFLOPs {
 public:
  using value_type = uint64_t;

  // Constructors and assignment operators
  //
  // `flatflow::OpFLOPs` is required to be MoveConstructible since
  // `flatflow::OpFLOPsRegistry` stores it as a value of `absl::flat_hash_map`.
  explicit OpFLOPs(value_type coef0 = 0, value_type coef1 = 0,
                   value_type coef2 = 0, value_type coef3 = 0)
      : coef0_(coef0), coef1_(coef1), coef2_(coef2), coef3_(coef3) {}

  explicit OpFLOPs(const OpFLOPs &other) = default;

  OpFLOPs &operator=(const OpFLOPs &other) = default;

  explicit OpFLOPs(OpFLOPs &&other) = default;

  OpFLOPs &operator=(OpFLOPs &&other) = default;

  // OpFLOPs::operator+()
  //
  // Combines the two given operator FLOPs specifications in coefficient-wise.
  OpFLOPs &operator+(const OpFLOPs &other) {
    auto spec = OpFLOPs();
    spec.coef0_ = coef0_ + other.coef0_;
    spec.coef1_ = coef1_ + other.coef1_;
    spec.coef2_ = coef2_ + other.coef2_;
    spec.coef3_ = coef3_ + other.coef3_;
    return spec;
  }

  value_type coef0_;  // constant
  value_type coef1_;  // linear
  value_type coef2_;  // quadratic
  value_type coef3_;  // cubic
};

// flatflow::OpFLOPsRegistry
//
// A `flatflow::OpFLOPsRegistry` holds the key information to identify operators
// and generate optimized computation plans. It has an operator FLOPs
// specifications table in a form of `absl::flat_hash_map<Operator, OpFLOPs>`,
// where each specification contains the FLOPs formula of the corresponding
// operator for a given size.
//
// To register a new operator, please follow the steps below:
//
// * First, declare a new operator as an enumerator value of `Operator` in the
//   FlatBuffers schema - i.e., `flatflow/ops/operator.fbs`. The enumerator
//   values are recommended to be sorted in order of their original operator
//   names (not the enumerator values) for searching convenience.
// * Second, generate codes from the updated schema by running `make generate`
//   in the root directory of this source tree. This will create
//   `flatflow/ops/operator_generated.h`, etc.
// * Third, add a template specialization of `OpFLOPsRegistry::RegisterOpFLOPs`
//   for the new operator; please refer to the below implementations for
//   supported operators. Note that omitting this step will call the base
//   template of `OpFLOPsRegistry::RegisterOpFLOPs`, which raises assertion
//   failure at compile-time.
// * Finally, register the new operator to the operator FLOPs specifications
//   table by calling its specialized `OpFLOPsRegistry::RegisterOpFLOPs` in the
//   constructor below.
class OpFLOPsRegistry {
 public:
  using value_type = OpFLOPs::value_type;

  explicit OpFLOPsRegistry(value_type hidden_size) {
    constexpr auto kOpTableSpace =
        sizeof(EnumValuesOperator()) / sizeof(Operator);
    table_.reserve(kOpTableSpace);

    RegisterOpFLOPs<Operator::_SOFTMAX>(hidden_size);
    RegisterOpFLOPs<Operator::_TO_COPY>(hidden_size);
    RegisterOpFLOPs<Operator::_UNSAFE_VIEW>(hidden_size);
    RegisterOpFLOPs<Operator::ADD_TENSOR>(hidden_size);
    RegisterOpFLOPs<Operator::ARANGE_START>(hidden_size);
    RegisterOpFLOPs<Operator::BMM>(hidden_size);
    RegisterOpFLOPs<Operator::CAT>(hidden_size);
    RegisterOpFLOPs<Operator::CLONE>(hidden_size);
    RegisterOpFLOPs<Operator::COS>(hidden_size);
    RegisterOpFLOPs<Operator::EMBEDDING>(hidden_size);
    RegisterOpFLOPs<Operator::EXPAND>(hidden_size);
    RegisterOpFLOPs<Operator::INDEX_TENSOR>(hidden_size);
    RegisterOpFLOPs<Operator::MEAN_DIM>(hidden_size);
    RegisterOpFLOPs<Operator::MM>(hidden_size);
    RegisterOpFLOPs<Operator::MUL_SCALAR>(hidden_size);
    RegisterOpFLOPs<Operator::MUL_TENSOR>(hidden_size);
    RegisterOpFLOPs<Operator::NEG>(hidden_size);
    RegisterOpFLOPs<Operator::POW_TENSOR_SCALAR>(hidden_size);
    RegisterOpFLOPs<Operator::REPEAT>(hidden_size);
    RegisterOpFLOPs<Operator::RSQRT>(hidden_size);
    RegisterOpFLOPs<Operator::SILU>(hidden_size);
    RegisterOpFLOPs<Operator::SIN>(hidden_size);
    RegisterOpFLOPs<Operator::SLICE_TENSOR>(hidden_size);
    RegisterOpFLOPs<Operator::SYM_SIZE_INT>(hidden_size);
    RegisterOpFLOPs<Operator::T>(hidden_size);
    RegisterOpFLOPs<Operator::TRANSPOSE_INT>(hidden_size);
    RegisterOpFLOPs<Operator::UNSQUEEZE>(hidden_size);
    RegisterOpFLOPs<Operator::VIEW>(hidden_size);
  }

  explicit OpFLOPsRegistry(const OpFLOPsRegistry &other) = default;

  OpFLOPsRegistry &operator=(const OpFLOPsRegistry &other) = default;

  explicit OpFLOPsRegistry(OpFLOPsRegistry &&other) = default;

  OpFLOPsRegistry &operator=(OpFLOPsRegistry &&other) = default;

  // OpFLOPsRegistry::RegisterOpFLOPs<>()
  //
  // This is a base template for registering operator FLOPs specifications;
  // calling this means that the program is ill-formed and should fail to
  // compile.
  template <Operator>
  void RegisterOpFLOPs(value_type hidden_size) {
    static_assert(std::false_type::value);
  }

  // OpFLOPsRegistry::MapReduce()
  //
  // Transforms the given operators to the corresponding operator FLOPs
  // specifications, then returns their reduction.
  OpFLOPs MapReduce(const flatbuffers::Vector<Operator> *operators) {
    auto specs = std::vector<OpFLOPs>(operators->size());

    // clang-format off
    #pragma omp parallel for
    for (flatbuffers::uoffset_t index = 0; index < operators->size(); ++index) {
      specs[index] = table_.at(operators->Get(index));
    }
    // clang-format on

    return std::reduce(std::execution::par, specs.cbegin(), specs.cend(),
                       OpFLOPs());
  }

  absl::flat_hash_map<Operator, OpFLOPs> table_;
};

////////////////////////////////////////////////////////////////////////////////
/////////////// OPERATOR FLOPS SPECIFICATIONS REGISTRATION BEGIN ///////////////
////////////////////////////////////////////////////////////////////////////////

// OpFLOPsRegistry::RegisterOpFLOPs<_SOFTMAX>()
//
// Registers operator FLOPs specification for `_softmax`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::_SOFTMAX>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::_SOFTMAX, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<_TO_COPY>()
//
// Registers operator FLOPs specification for `_to_copy`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::_TO_COPY>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::_TO_COPY, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<_UNSAFE_VIEW>()
//
// Registers operator FLOPs specification for `_unsafe_view`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::_UNSAFE_VIEW>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::_UNSAFE_VIEW, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<ADD_TENSOR>()
//
// Registers operator FLOPs specification for `add.Tensor`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::ADD_TENSOR>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::ADD_TENSOR, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<ARANGE_START>()
//
// Registers operator FLOPs specification for `arange.start`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::ARANGE_START>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::ARANGE_START, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<BMM>()
//
// Registers operator FLOPs specification for `bmm`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::BMM>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::BMM, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<CAT>()
//
// Registers operator FLOPs specification for `cat`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::CAT>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::CAT, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<CLONE>()
//
// Registers operator FLOPs specification for `clone`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::CLONE>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::CLONE, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<COS>()
//
// Registers operator FLOPs specification for `cos`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::COS>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::COS, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<EMBEDDING>()
//
// Registers operator FLOPs specification for `embedding`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::EMBEDDING>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::EMBEDDING, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<EXPAND>()
//
// Registers operator FLOPs specification for `expand`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::EXPAND>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::EXPAND, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<INDEX_TENSOR>()
//
// Registers operator FLOPs specification for `index.Tensor`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::INDEX_TENSOR>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::INDEX_TENSOR, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<MEAN_DIM>()
//
// Registers operator FLOPs specification for `mean.dim`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::MEAN_DIM>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::MEAN_DIM, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<MM>()
//
// Registers operator FLOPs specification for `mm`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::MM>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::MM, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<MUL_SCALAR>()
//
// Registers operator FLOPs specification for `mul.Scalar`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::MUL_SCALAR>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::MUL_SCALAR, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<MUL_TENSOR>()
//
// Registers operator FLOPs specification for `mul.Tensor`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::MUL_TENSOR>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::MUL_TENSOR, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<NEG>()
//
// Registers operator FLOPs specification for `neg`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::NEG>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::NEG, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<POW_TENSOR_SCALAR>()
//
// Registers operator FLOPs specification for `pow.Tensor_Scalar`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::POW_TENSOR_SCALAR>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(
      std::make_pair(Operator::POW_TENSOR_SCALAR, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<REPEAT>()
//
// Registers operator FLOPs specification for `repeat`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::REPEAT>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::REPEAT, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<RSQRT>()
//
// Registers operator FLOPs specification for `rsqrt`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::RSQRT>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::RSQRT, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<SILU>()
//
// Registers operator FLOPs specification for `silu`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::SILU>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::SILU, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<SIN>()
//
// Registers operator FLOPs specification for `sin`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::SIN>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::SIN, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<SLICE_TENSOR>()
//
// Registers operator FLOPs specification for `slice.Tensor`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::SLICE_TENSOR>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::SLICE_TENSOR, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<SYM_SIZE_INT>()
//
// Registers operator FLOPs specification for `sym_size.int`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::SYM_SIZE_INT>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::SYM_SIZE_INT, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<T>()
//
// Registers operator FLOPs specification for `t`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::T>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::T, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<TRANSPOSE_INT>()
//
// Registers operator FLOPs specification for `transpose.int`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::TRANSPOSE_INT>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::TRANSPOSE_INT, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<UNSQUEEZE>()
//
// Registers operator FLOPs specification for `unsqueeze`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::UNSQUEEZE>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::UNSQUEEZE, OpFLOPs(0, 0, 0, 0)));
}

// OpFLOPsRegistry::RegisterOpFLOPs<VIEW>()
//
// Registers operator FLOPs specification for `view`.
template <>
void OpFLOPsRegistry::RegisterOpFLOPs<Operator::VIEW>(
    OpFLOPsRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::VIEW, OpFLOPs(0, 0, 0, 0)));
}

////////////////////////////////////////////////////////////////////////////////
//////////////// OPERATOR FLOPS SPECIFICATIONS REGISTRATION END ////////////////
////////////////////////////////////////////////////////////////////////////////

}  // namespace flatflow

#endif  // FLATFLOW_OPS_OPS_H_
