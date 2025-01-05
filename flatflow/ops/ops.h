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
#include <type_traits>
#include <utility>

#include "absl/container/flat_hash_map.h"

#include "flatflow/ops/operator_generated.h"

namespace flatflow {

class OpProf {
 public:
  using value_type = uint64_t;

  OpProf() {}

  OpProf(value_type coef0, value_type coef1, value_type coef2, value_type coef3)
      : coef0_(coef0), coef1_(coef1), coef2_(coef2), coef3_(coef3) {}

  OpProf(const OpProf &other) = default;

  OpProf &operator=(const OpProf &other) = default;

  OpProf(OpProf &&other) = default;

  OpProf &operator=(OpProf &&other) = default;

  OpProf &operator+(const OpProf &other) {
    auto prof = OpProf();
    prof.coef0_ = coef0_ + other.coef0_;
    prof.coef1_ = coef1_ + other.coef1_;
    prof.coef2_ = coef2_ + other.coef2_;
    prof.coef3_ = coef3_ + other.coef3_;
    return prof;
  }

  value_type coef0_;
  value_type coef1_;
  value_type coef2_;
  value_type coef3_;
};

class OpProfRegistry {
 public:
  using value_type = OpProf::value_type;

  OpProfRegistry(value_type hidden_size) {
    constexpr auto kOpTableSpace =
        sizeof(EnumValuesOperator()) / sizeof(Operator);
    table_.reserve(kOpTableSpace);

    RegisterOpProf<Operator::_SOFTMAX>(hidden_size);
    RegisterOpProf<Operator::_TO_COPY>(hidden_size);
    RegisterOpProf<Operator::_UNSAFE_VIEW>(hidden_size);
    RegisterOpProf<Operator::ADD_TENSOR>(hidden_size);
    RegisterOpProf<Operator::ARANGE_START>(hidden_size);
    RegisterOpProf<Operator::BMM>(hidden_size);
    RegisterOpProf<Operator::CAT>(hidden_size);
    RegisterOpProf<Operator::CLONE>(hidden_size);
    RegisterOpProf<Operator::COS>(hidden_size);
    RegisterOpProf<Operator::EMBEDDING>(hidden_size);
    RegisterOpProf<Operator::EXPAND>(hidden_size);
    RegisterOpProf<Operator::INDEX_TENSOR>(hidden_size);
    RegisterOpProf<Operator::MEAN_DIM>(hidden_size);
    RegisterOpProf<Operator::MM>(hidden_size);
    RegisterOpProf<Operator::MUL_SCALAR>(hidden_size);
    RegisterOpProf<Operator::MUL_TENSOR>(hidden_size);
    RegisterOpProf<Operator::NEG>(hidden_size);
    RegisterOpProf<Operator::POW_TENSOR_SCALAR>(hidden_size);
    RegisterOpProf<Operator::REPEAT>(hidden_size);
    RegisterOpProf<Operator::RSQRT>(hidden_size);
    RegisterOpProf<Operator::SILU>(hidden_size);
    RegisterOpProf<Operator::SIN>(hidden_size);
    RegisterOpProf<Operator::SLICE_TENSOR>(hidden_size);
    RegisterOpProf<Operator::SYM_SIZE_INT>(hidden_size);
    RegisterOpProf<Operator::T>(hidden_size);
    RegisterOpProf<Operator::TRANSPOSE_INT>(hidden_size);
    RegisterOpProf<Operator::UNSQUEEZE>(hidden_size);
    RegisterOpProf<Operator::VIEW>(hidden_size);
  }

  // OpProfRegistry::RegisterOpProf<>()
  //
  // This is a base template for operator profile registration; calling this
  // means that the program is ill-formed and should fail to compile.
  template <Operator>
  void RegisterOpProf(value_type hidden_size) {
    static_assert(std::false_type::value);
  }

  absl::flat_hash_map<Operator, OpProf> table_;
};

///////////////////////////////////////////////////////////////////////////////
///////////////// OPERATOR PROFILE REGISTRATION SECTION BEGIN /////////////////
///////////////////////////////////////////////////////////////////////////////

// OpProfRegistry::RegisterOpProf<_SOFTMAX>()
//
// Registers operator profile for `_softmax`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::_SOFTMAX>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::_SOFTMAX, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<_TO_COPY>()
//
// Registers operator profile for `_to_copy`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::_TO_COPY>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::_TO_COPY, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<_UNSAFE_VIEW>()
//
// Registers operator profile for `_unsafe_view`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::_UNSAFE_VIEW>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::_UNSAFE_VIEW, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<ADD_TENSOR>()
//
// Registers operator profile for `add.Tensor`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::ADD_TENSOR>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::ADD_TENSOR, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<ARANGE_START>()
//
// Registers operator profile for `arange.start`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::ARANGE_START>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::ARANGE_START, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<BMM>()
//
// Registers operator profile for `bmm`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::BMM>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::BMM, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<CAT>()
//
// Registers operator profile for `cat`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::CAT>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::CAT, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<CLONE>()
//
// Registers operator profile for `clone`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::CLONE>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::CLONE, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<COS>()
//
// Registers operator profile for `cos`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::COS>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::COS, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<EMBEDDING>()
//
// Registers operator profile for `embedding`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::EMBEDDING>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::EMBEDDING, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<EXPAND>()
//
// Registers operator profile for `expand`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::EXPAND>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::EXPAND, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<INDEX_TENSOR>()
//
// Registers operator profile for `index.Tensor`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::INDEX_TENSOR>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::INDEX_TENSOR, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<MEAN_DIM>()
//
// Registers operator profile for `mean.dim`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::MEAN_DIM>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::MEAN_DIM, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<MM>()
//
// Registers operator profile for `mm`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::MM>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::MM, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<MUL_SCALAR>()
//
// Registers operator profile for `mul.Scalar`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::MUL_SCALAR>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::MUL_SCALAR, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<MUL_TENSOR>()
//
// Registers operator profile for `mul.Tensor`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::MUL_TENSOR>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::MUL_TENSOR, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<NEG>()
//
// Registers operator profile for `neg`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::NEG>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::NEG, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<POW_TENSOR_SCALAR>()
//
// Registers operator profile for `pow.Tensor_Scalar`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::POW_TENSOR_SCALAR>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(
      std::make_pair(Operator::POW_TENSOR_SCALAR, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<REPEAT>()
//
// Registers operator profile for `repeat`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::REPEAT>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::REPEAT, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<RSQRT>()
//
// Registers operator profile for `rsqrt`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::RSQRT>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::RSQRT, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<SILU>()
//
// Registers operator profile for `silu`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::SILU>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::SILU, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<SIN>()
//
// Registers operator profile for `sin`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::SIN>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::SIN, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<SLICE_TENSOR>()
//
// Registers operator profile for `slice.Tensor`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::SLICE_TENSOR>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::SLICE_TENSOR, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<SYM_SIZE_INT>()
//
// Registers operator profile for `sym_size.int`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::SYM_SIZE_INT>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::SYM_SIZE_INT, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<T>()
//
// Registers operator profile for `t`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::T>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::T, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<TRANSPOSE_INT>()
//
// Registers operator profile for `transpose.int`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::TRANSPOSE_INT>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::TRANSPOSE_INT, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<UNSQUEEZE>()
//
// Registers operator profile for `unsqueeze`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::UNSQUEEZE>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::UNSQUEEZE, OpProf(0, 0, 0, 0)));
}

// OpProfRegistry::RegisterOpProf<VIEW>()
//
// Registers operator profile for `view`.
template <>
void OpProfRegistry::RegisterOpProf<Operator::VIEW>(
    OpProfRegistry::value_type hidden_size) {
  table_.insert(std::make_pair(Operator::VIEW, OpProf(0, 0, 0, 0)));
}

///////////////////////////////////////////////////////////////////////////////
////////////////// OPERATOR PROFILE REGISTRATION SECTION END //////////////////
///////////////////////////////////////////////////////////////////////////////

}  // namespace flatflow

#endif  // FLATFLOW_OPS_OPS_H_
