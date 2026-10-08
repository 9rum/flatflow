// SPDX-License-Identifier: Apache-2.0

#ifndef FLATFLOW_OPS_GRAPH_H_
#define FLATFLOW_OPS_GRAPH_H_

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "absl/log/die_if_null.h"

#include "flatflow/ops/graph_generated.h"
#include "flatflow/ops/scalar_type_generated.h"
#include "flatflow/types.h"

namespace flatflow {

// `SymInt` records a value within the symbolic shape of a tensor.
struct SymInt {
  using value_type =
      std::array<remove_extent_t<decltype(internal::SymInt().data())>,
                 extent_v<decltype(internal::SymInt().data())>>::value_type;
  using size_type =
      std::array<remove_extent_t<decltype(internal::SymInt().data())>,
                 extent_v<decltype(internal::SymInt().data())>>::size_type;

  template <typename... Args>
  constexpr SymInt(Args... args) noexcept : data{args...} {}

  constexpr SymInt(const SymInt &) noexcept = default;

  constexpr SymInt &operator=(const SymInt &) noexcept = default;

  constexpr SymInt(SymInt &&) noexcept = default;

  constexpr SymInt &operator=(SymInt &&) noexcept = default;

  SymInt(const internal::SymInt *s) noexcept {
    std::ranges::copy(*ABSL_DIE_IF_NULL(s)->data(), data.begin());
  }

  constexpr value_type operator[](size_type pos) const noexcept {
    return data[pos];
  }

  constexpr bool operator==(const SymInt &) const noexcept = default;

  std::array<remove_extent_t<decltype(internal::SymInt().data())>,
             extent_v<decltype(internal::SymInt().data())>>
      data;
};

// `TensorMetadata` is a structure containing pertinent information about a
// tensor within a PyTorch program.
struct TensorMetadata {
  constexpr TensorMetadata() noexcept = default;

  constexpr TensorMetadata(ScalarType dtype,
                           const std::vector<SymInt> &shape) noexcept
      : dtype(dtype), shape(shape) {}

  constexpr TensorMetadata(ScalarType dtype,
                           std::vector<SymInt> &&shape) noexcept
      : dtype(dtype), shape(std::move(shape)) {}

  constexpr TensorMetadata(const TensorMetadata &) noexcept = default;

  constexpr TensorMetadata &operator=(const TensorMetadata &) noexcept =
      default;

  constexpr TensorMetadata(TensorMetadata &&) noexcept = default;

  constexpr TensorMetadata &operator=(TensorMetadata &&) noexcept = default;

  TensorMetadata(const internal::TensorMetadata *meta) noexcept
      : dtype(ABSL_DIE_IF_NULL(meta)->dtype()),
        shape(meta->shape()->begin(), meta->shape()->end()) {}

  constexpr bool operator==(const TensorMetadata &) const noexcept = default;

  ScalarType dtype;
  std::vector<SymInt> shape;
};

}  // namespace flatflow

#endif  // FLATFLOW_OPS_GRAPH_H_
