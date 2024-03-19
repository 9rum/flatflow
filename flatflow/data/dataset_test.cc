// Copyright 2024 The FlatFlow Authors
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

#include "flatflow/data/dataset.h"

#include <cstdlib>
#include <ctime>
#include <map>
#include <vector>

#include <absl/container/inlined_vector.h>
#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include "flatflow/data/dataset_test.h"

namespace {

class DatasetTest final : private flatflow::data::Dataset<uint64_t, uint16_t> {
 public:
  inline explicit DatasetTest(const flatbuffers::Vector64<uint16_t> *sizes,
                              uint64_t seed)
      : flatflow::data::Dataset<uint64_t, uint16_t>(sizes, seed) {}

  inline bool contains(uint16_t size) const noexcept {
    return items.contains(size);
  }

  inline std::size_t size(uint16_t size) const noexcept {
    return items.at(size).size();
  }

  inline std::size_t capacity(uint16_t size) const noexcept {
    return items.at(size).capacity();
  }

  inline bool is_sorted(uint16_t size) const noexcept {
    return std::is_sorted(items.at(size).cbegin(), items.at(size).cend());
  }

  inline bool empty() const noexcept { return recyclebin.empty(); }

  inline void on_epoch_begin(uint64_t epoch) { on_epoch_begin(epoch); }

  inline std::vector<uint64_t> at(uint16_t size) const noexcept {
    return items.at(size);
  }
};

TEST(DatasetTest, Constructor) {
  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  auto items = std::map<uint16_t, std::size_t>();
  for (uint16_t size = 1; size <= 1 << 12; ++size) {
    items.emplace(size, static_cast<std::size_t>(std::rand() % (1 << 15)));
  }

  auto sizes = std::vector<uint16_t>();
  for (const auto item : items) {
    const auto size = item.first;
    auto count = item.second;
    for (; 0 < count; --count) {
      sizes.push_back(size);
    }
  }
  sizes.shrink_to_fit();

  // As of FlatBuffers v24.3.7, it is not possible to initialize a 64-bit
  // vector directly; use generated code from the FlatBuffers schema.
  auto builder = flatbuffers::FlatBufferBuilder64();
  auto sizes__ = builder.CreateVector64(sizes);
  auto offset = CreateSizes(builder, sizes__);
  builder.Finish(offset);
  auto sizes_ = GetSizes(builder.GetBufferPointer());
  auto dataset = DatasetTest(sizes_->sizes(), 0UL);

  for (const auto item : items) {
    const auto size = item.first;
    auto count = item.second;
    if (0 < count) {
      EXPECT_EQ(dataset.size(size), count);
      EXPECT_EQ(dataset.capacity(size), count);
      EXPECT_TRUE(dataset.is_sorted(size));
    } else {
      EXPECT_FALSE(dataset.contains(size));
    }
  }
  EXPECT_TRUE(dataset.empty());
}

TEST(DatasetTest, IntraBatchShuffling) {
  // Test case for IntraBatchShuffling
  //
  // NOTE:
  //
  // * We first check if the dataset vectors are sorted.
  // * After shuffling, we check if dataset vectors are not sorted.
  // * Then using the same input vector, we shuffle the slots.
  // * Last, we check if the slots and dataset vectors are equal. 

  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  thread_local auto generator = std::ranlux48();
  auto items = std::map<uint16_t, std::size_t>();
  uint16_t epoch = 0;
  for (uint16_t size = 1; size <= 1 << 12; ++size) {
    items.emplace(size, static_cast<std::size_t>(std::rand() % (1 << 15)));
  }

  auto sizes = std::vector<uint16_t>();
  for (const auto item : items) {
    const auto size = item.first;
    auto count = item.second;
    for (; 0 < count; --count) {
      sizes.push_back(size);
    }
  }
  sizes.shrink_to_fit();

  auto builder = flatbuffers::FlatBufferBuilder64();
  auto sizes__ = builder.CreateVector64(sizes);
  auto offset = CreateSizes(builder, sizes__);
  builder.Finish(offset);

  auto sizes_ = GetSizes(builder.GetBufferPointer());
  auto dataset = DatasetTest(sizes_->sizes(), 0UL);

  // Expects all vectors are sorted.
  for (const auto &item : items) {
    const auto &size = item.first;
    EXPECT_TRUE(dataset.is_sorted(size));
  }

  // call on_epoch_begin for shuffle.
  dataset.on_epoch_begin(epoch);

  // Expects all vectors are not sorted.
  for (const auto &item : items) {
    const auto &size = item.first;
    EXPECT_FALSE(dataset.is_sorted(size));
  }

  constexpr auto kIndexSlotSpace =
      static_cast<std::size_t>(1 << std::numeric_limits<uint16_t>::digits);
  auto counts =
      absl::InlinedVector<uint64_t, kIndexSlotSpace>(kIndexSlotSpace, 0);

// clang-format off
  #pragma omp unroll partial
  // clang-format on
  for (uint64_t index = 0; index < sizes.size(); ++index) {
    const auto size = static_cast<std::size_t>(sizes[index]);
    ++counts.at(size);
  }

  auto slots = absl::InlinedVector<std::vector<uint64_t>, kIndexSlotSpace>(
      kIndexSlotSpace);

// clang-format off
  #pragma omp parallel for
  // clang-format on
  for (std::size_t size = 0; size < counts.size(); ++size) {
    const auto count = counts.at(size);
    if (0 < count) {
      slots.at(size).reserve(static_cast<std::size_t>(count));
    }
  }

// clang-format off
  #pragma omp unroll partial
  // clang-format on
  for (uint64_t index = 0; index < sizes.size(); ++index) {
    const auto size = static_cast<std::size_t>(sizes[index]);
    slots.at(size).emplace_back(index);
  }

  // Expects dataset and slots are not equal.
  // Since, dataset is shuffled and slots are not.
  for (std::size_t size = 0; size < counts.size(); ++size) {
    const auto count = counts.at(size);
    if (0 < count) {
      const auto &dataset_vector = dataset.at(size);
      const auto &current_vector = slots.at(size);
      EXPECT_FALSE(std::equal(dataset_vector.begin(), dataset_vector.end(),
                              current_vector.begin()));
    }
  }
// clang-format off
  #pragma omp parallel for
  // clang-format on
  for (auto &item : slots) {
    generator.seed(static_cast<uint_fast64_t>(0UL + epoch));
    std::shuffle(item.begin(), item.end(), generator);
  }

  // Expects dataset and slots are equal.
  // Since, slots are shuffled.
  for (std::size_t size = 0; size < counts.size(); ++size) {
    const auto count = counts.at(size);
    if (0 < count) {
      const auto &dataset_vector = dataset.access_vector(size);
      const auto &current_vector = slots.at(size);
      EXPECT_TRUE(std::equal(dataset_vector.begin(), dataset_vector.end(),
                             current_vector.begin()));
    }
  }
}

}  // namespace
