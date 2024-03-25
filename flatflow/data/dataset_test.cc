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

  inline std::size_t size(uint16_t size, bool checkItem = true) const noexcept {
    if (checkItem) {
      return items.at(size).size();
    } else {
      return recyclebin.at(size).size();
    }
  }

  inline std::size_t capacity(uint16_t size,
                              bool checkItem = true) const noexcept {
    if (checkItem) {
      return items.at(size).capacity();
    } else {
      return recyclebin.at(size).capacity();
    }
  }

  inline bool is_sorted(uint16_t size) const noexcept {
    return std::is_sorted(items.at(size).cbegin(), items.at(size).cend());
  }

  inline bool empty(bool checkItem = true) const noexcept {
    if (checkItem) {
      return items.empty();
    } else {
      return recyclebin.empty();
    }
  }

  inline void shuffle(uint64_t epoch) { on_epoch_begin(epoch); }

  inline std::vector<uint64_t> &at(uint16_t size,
                                   bool checkItem = true) noexcept {
    if (checkItem) {
      return items.at(size);
    } else {
      return recyclebin.at(size);
    }
  }

  inline std::pair<uint64_t, uint16_t> retrieveItem(uint16_t size) {
    return (*this)[size];
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
  EXPECT_TRUE(dataset.empty(false));
}

TEST(DatasetTest, IntraBatchShuffling) {
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
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

  // call on_epoch_begin for shuffle.
  dataset.shuffle(epoch);

  constexpr auto kIndexSlotSpace =
      static_cast<std::size_t>(1 << std::numeric_limits<uint16_t>::digits);
  auto counts =
      absl::InlinedVector<uint64_t, kIndexSlotSpace>(kIndexSlotSpace, 0);

  #pragma omp unroll partial
  for (uint64_t index = 0; index < sizes.size(); ++index) {
    const auto size = static_cast<std::size_t>(sizes[index]);
    ++counts.at(size);
  }

  auto slots = absl::InlinedVector<std::vector<uint64_t>, kIndexSlotSpace>(
      kIndexSlotSpace);

  #pragma omp parallel for
  for (std::size_t size = 0; size < counts.size(); ++size) {
    const auto count = counts.at(size);
    if (0 < count) {
      slots.at(size).reserve(static_cast<std::size_t>(count));
    }
  }

  #pragma omp unroll partial
  for (uint64_t index = 0; index < sizes.size(); ++index) {
    const auto size = static_cast<std::size_t>(sizes[index]);
    slots.at(size).emplace_back(index);
  }

  thread_local auto generator = std::ranlux48();

  #pragma omp parallel for
  for (auto &item : slots) {
    generator.seed(static_cast<uint_fast64_t>(0UL + epoch));
    std::shuffle(item.begin(), item.end(), generator);
  }

  // Expects dataset and slots are equal.
  // Since, slots are shuffled.
  for (std::size_t size = 0; size < counts.size(); ++size) {
    const auto count = counts.at(size);
    if (0 < count) {
      const auto &dataset_vector = dataset.at(size);
      const auto &current_vector = slots.at(size);
      EXPECT_FALSE(dataset.is_sorted(size));
      EXPECT_TRUE(std::equal(dataset_vector.begin(), dataset_vector.end(),
                             current_vector.begin()));
    }
  }
}

TEST(DatasetTest, IndexRetriever) {
  // Note:
  //
  // Test sample looks like below items.
  // std::unordered_map<int, std::vector<int>> items = {
  //       {10, {0, 1, 2}},
  //       {20, {3, 4}},
  //       {30, {5, 6, 7, 8}}
  //       {40, {9}}
  // };
  //
  // There are three test case.
  // Test case 1 : When the size is in the dataset.
  // Test case 2 : When the size is not in the dataset.
  // Test case 3 : When only one index of the size is left.
  //
  // After retrieving all indexes from the items, check if the dataset is empty
  // and the size of the recyclebin is correct. At the end, check if the
  // recyclebin vectors are all in reverted order.

  std::vector<uint16_t> sizes = {10, 10, 10, 20, 20, 30, 30, 30, 30, 40};
  auto builder = flatbuffers::FlatBufferBuilder64();
  auto sizes__ = builder.CreateVector64(sizes);
  auto offset = CreateSizes(builder, sizes__);
  builder.Finish(offset);

  auto sizes_ = GetSizes(builder.GetBufferPointer());
  auto dataset = DatasetTest(sizes_->sizes(), 0UL);

  // Test case 1
  // Expected behavior : Dataset will retrieve a pair of index and size from the
  // dataset.
  uint16_t size = 20;
  auto result = dataset.retrieveItem(size);
  EXPECT_EQ(result.first, 4);
  EXPECT_EQ(result.second, size);
  EXPECT_EQ(dataset.size(size), 1);
  EXPECT_EQ(dataset.size(size, false), 1);

  // Test case 2
  // Expected behaivor : Dataset will retrieve closest size's pair for size
  // which will be 10.
  size = 8;
  result = dataset.retrieveItem(size);
  EXPECT_EQ(result.first, 2);
  EXPECT_EQ(result.second, 10);
  EXPECT_EQ(dataset.size(10, false), 1);
  EXPECT_EQ(dataset.size(10), 2);
  EXPECT_EQ(dataset.capacity(10, false), 3);

  // Test case 3
  // Expected behavior : items will erase size 40 from the dataset.
  size = 40;
  result = dataset.retrieveItem(size);
  EXPECT_EQ(result.first, 9);
  EXPECT_EQ(result.second, size);
  EXPECT_EQ(dataset.size(size, false), 1);
  EXPECT_EQ(dataset.capacity(size, false), 1);

  // Test case 3
  // Expected behavior : Size 20 will be removed from the dataset.
  size = 20;
  result = dataset.retrieveItem(size);
  EXPECT_EQ(result.first, 3);
  EXPECT_EQ(result.second, size);
  EXPECT_EQ(dataset.size(size, false), 2);
  EXPECT_EQ(dataset.capacity(size, false), 2);

  // Test case 2
  // Expected behavior : Dataset will retrieve closest size's pair for size
  // which will be 30.
  size = 28;
  result = dataset.retrieveItem(size);
  EXPECT_EQ(result.first, 8);
  EXPECT_EQ(result.second, 30);
  EXPECT_EQ(dataset.size(30, false), 1);
  EXPECT_EQ(dataset.size(30), 3);
  EXPECT_EQ(dataset.capacity(30, false), 4);

  // Test case 2
  // Expected behavior : Dataset will retrieve closest size's pair for size
  // which will be 30.
  size = 31;
  result = dataset.retrieveItem(size);
  EXPECT_EQ(result.first, 7);
  EXPECT_EQ(result.second, 30);
  EXPECT_EQ(dataset.size(30, false), 2);
  EXPECT_EQ(dataset.size(30), 2);
  EXPECT_EQ(dataset.capacity(30, false), 4);

  // Test case 1
  // Expected behavior : Dataset will retrieve a pair of index and size from the
  // dataset.
  size = 30;
  result = dataset.retrieveItem(size);
  EXPECT_EQ(result.first, 6);
  EXPECT_EQ(result.second, size);
  EXPECT_EQ(dataset.size(size, false), 3);
  EXPECT_EQ(dataset.size(size), 1);
  EXPECT_EQ(dataset.capacity(size, false), 4);

  // Test case 2 & 3
  // Expected behavior : Size 30 will be removed from the dataset.
  size = 34;
  result = dataset.retrieveItem(size);
  EXPECT_EQ(result.first, 5);
  EXPECT_EQ(result.second, 30);
  EXPECT_EQ(dataset.size(30, false), 4);
  EXPECT_FALSE(dataset.contains(30));
  EXPECT_EQ(dataset.capacity(30, false), 4);

  // Test case 1
  // Expected behavior : Dataset will retrieve a pair of index and size from the
  // dataset.
  size = 10;
  result = dataset.retrieveItem(size);
  EXPECT_EQ(result.first, 1);
  EXPECT_EQ(result.second, size);
  EXPECT_EQ(dataset.size(size), 1);
  EXPECT_EQ(dataset.size(size, false), 2);
  EXPECT_EQ(dataset.capacity(size, false), 3);

  // Test case 3
  // Expected behavior : Size 10 will be removed from the dataset.
  size = 10;
  result = dataset.retrieveItem(size);
  EXPECT_EQ(result.first, 0);
  EXPECT_EQ(result.second, size);
  EXPECT_EQ(dataset.size(size, false), 3);
  EXPECT_EQ(dataset.capacity(size, false), 3);

  // Check if items is empty.
  EXPECT_TRUE(dataset.empty(true));

  // check size of recyclebin.
  EXPECT_EQ(dataset.size(10, false), 3);
  EXPECT_EQ(dataset.size(20, false), 2);
  EXPECT_EQ(dataset.size(30, false), 4);
  EXPECT_EQ(dataset.size(40, false), 1);

  // Check Capacity of recyclebin.
  EXPECT_EQ(dataset.capacity(10, false), 3);
  EXPECT_EQ(dataset.capacity(20, false), 2);
  EXPECT_EQ(dataset.capacity(30, false), 4);
  EXPECT_EQ(dataset.capacity(40, false), 1);

  // check if recyclebin vectors are all reverted.
  std::reverse(dataset.at(10, false).begin(), dataset.at(10, false).end());
  EXPECT_TRUE(std::is_sorted(dataset.at(10, false).begin(),
                             dataset.at(10, false).end()));
  std::reverse(dataset.at(20, false).begin(), dataset.at(20, false).end());
  EXPECT_TRUE(std::is_sorted(dataset.at(20, false).begin(),
                             dataset.at(20, false).end()));
  std::reverse(dataset.at(30, false).begin(), dataset.at(30, false).end());
  EXPECT_TRUE(std::is_sorted(dataset.at(30, false).begin(),
                             dataset.at(30, false).end()));
  std::reverse(dataset.at(40, false).begin(), dataset.at(40, false).end());
  EXPECT_TRUE(std::is_sorted(dataset.at(40, false).begin(),
                             dataset.at(40, false).end()));
}

}  // namespace
