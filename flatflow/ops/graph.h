// SPDX-License-Identifier: Apache-2.0

#ifndef FLATFLOW_OPS_GRAPH_H_
#define FLATFLOW_OPS_GRAPH_H_

#include <algorithm>
#include <array>

#include "absl/log/die_if_null.h"

#include "flatflow/ops/graph_generated.h"
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

}  // namespace flatflow

#endif  // FLATFLOW_OPS_GRAPH_H_
