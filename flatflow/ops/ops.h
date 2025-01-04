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

// OpProfRegistry::RegisterOpProf<_SOFTMAX>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::_SOFTMAX>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<_TO_COPY>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::_TO_COPY>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<_UNSAFE_VIEW>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::_UNSAFE_VIEW>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<ADD_TENSOR>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::ADD_TENSOR>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<ARANGE_START>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::ARANGE_START>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<BMM>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::BMM>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<CAT>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::CAT>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<CLONE>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::CLONE>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<COS>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::COS>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<EMBEDDING>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::EMBEDDING>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<EXPAND>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::EXPAND>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<INDEX_TENSOR>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::INDEX_TENSOR>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<MEAN_DIM>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::MEAN_DIM>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<MM>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::MM>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<MUL_SCALAR>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::MUL_SCALAR>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<MUL_TENSOR>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::MUL_TENSOR>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<NEG>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::NEG>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<POW_TENSOR_SCALAR>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::POW_TENSOR_SCALAR>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<REPEAT>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::REPEAT>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<RSQRT>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::RSQRT>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<SILU>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::SILU>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<SIN>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::SIN>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<SLICE_TENSOR>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::SLICE_TENSOR>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<SYM_SIZE_INT>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::SYM_SIZE_INT>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<T>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::T>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<TRANSPOSE_INT>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::TRANSPOSE_INT>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<UNSQUEEZE>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::UNSQUEEZE>(
    OpProfRegistry::value_type hidden_size) {}

// OpProfRegistry::RegisterOpProf<VIEW>()
//
template <>
void OpProfRegistry::RegisterOpProf<Operator::VIEW>(
    OpProfRegistry::value_type hidden_size) {}

}  // namespace flatflow

#endif  // FLATFLOW_OPS_OPS_H_
