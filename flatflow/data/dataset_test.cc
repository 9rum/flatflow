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

#include <algorithm>
#include <execution>
#include <map>
#include <random>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/internal/globals.h"
#include "flatbuffers/flatbuffers.h"
#include "gtest/gtest.h"

#include "flatflow/data/dataset_test.h"

namespace {

// A read-only data set used only for testing purpose.
class Dataset : public flatflow::data::Dataset<uint64_t, uint16_t> {
 public:
  inline explicit Dataset() : flatflow::data::Dataset<uint64_t, uint16_t>() {}

  inline explicit Dataset(const flatbuffers::Vector64<uint16_t> *sizes,
                          const uint64_t &seed)
      : flatflow::data::Dataset<uint64_t, uint16_t>(sizes, seed) {}

  inline bool empty(const bool &items = true) const {
    return items ? items_.empty() : recyclebin_.empty();
  }

  inline std::size_t size(const uint16_t &size,
                          const bool &items = true) const {
    return items ? items_.at(size).size() : recyclebin_.at(size).size();
  }

  inline std::size_t capacity(const uint16_t &size,
                              const bool &items = true) const {
    return items ? items_.at(size).capacity() : recyclebin_.at(size).capacity();
  }

  inline bool contains(const uint16_t &size, const bool &items = true) const {
    return items ? items_.contains(size) : recyclebin_.contains(size);
  }

  inline bool is_sorted(const uint16_t &size, const bool &items = true) const {
    return items ? std::is_sorted(items_.at(size).cbegin(),
                                  items_.at(size).cend())
                 : std::is_sorted(recyclebin_.at(size).crbegin(),
                                  recyclebin_.at(size).crend());
  }

  inline void copy(const uint16_t &size, std::vector<uint64_t> &slot) const {
    std::copy(items_.at(size).cbegin(), items_.at(size).cend(), slot.begin());
  }

  inline bool equal(const uint16_t &size,
                    const std::vector<uint64_t> &slot) const {
    return std::equal(slot.cbegin(), slot.cend(), items_.at(size).cbegin());
  }
};

class DatasetTest : public testing::Test {
 protected:
  void SetUp() override {
    constexpr auto kMaxSize = static_cast<uint16_t>(1 << 12);
    constexpr auto kMaxCount = static_cast<std::size_t>(1 << 15);

    if (!absl::log_internal::IsInitialized()) {
      absl::InitializeLog();
      absl::SetStderrThreshold(absl::LogSeverity::kInfo);
    }

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    auto capacity = static_cast<std::size_t>(0);
    for (uint16_t size = 1; size <= kMaxSize; ++size) {
      const auto count = static_cast<std::size_t>(std::rand() % kMaxCount);
      capacity += count;
      counts_.try_emplace(size, count);
    }

    auto data = std::vector<uint16_t>();
    data.reserve(capacity);
    for (auto [size, count] : counts_) {
      for (; 0 < count; --count) {
        data.emplace_back(size);
      }
    }

    // As of FlatBuffers v24.3.7, it is not possible to initialize a 64-bit
    // vector directly; use generated code from the FlatBuffers schema.
    auto builder = flatbuffers::FlatBufferBuilder64();
    auto data__ = builder.CreateVector64(data);
    auto offset = CreateSizes(builder, data__);
    builder.Finish(offset);

    auto sizes = GetSizes(builder.GetBufferPointer());
    dataset_ = Dataset(sizes->data(), 0);
  }

  std::map<uint16_t, std::size_t> counts_;
  Dataset dataset_;
};

// This test answers the following questions to see that the inverted index is
// constructed as intended:
//
// * Are there any redundant keys stored in the inverted index?
// * Does each index slot occupy exactly as much memory footprint as required?
// * Is each index slot initially sorted?
// * Is the recycle bin initially empty?
TEST_F(DatasetTest, Constructor) {
  EXPECT_TRUE(dataset_.empty(false));

  for (const auto [size, count] : counts_) {
    if (0 < count) {
      EXPECT_EQ(dataset_.size(size), count);
      EXPECT_EQ(dataset_.capacity(size), count);
      EXPECT_TRUE(dataset_.is_sorted(size));
    } else {
      EXPECT_FALSE(dataset_.contains(size));
    }
  }
}

// This test checks whether intra-batch shuffling occurs deterministically for
// an arbitrary value of random seed.
TEST_F(DatasetTest, IntraBatchShuffling) {
  const auto epoch = static_cast<uint64_t>(std::rand());

  auto slots = std::map<uint16_t, std::vector<uint64_t>>();
  for (const auto [size, count] : counts_) {
    if (0 < count) {
      auto slot = std::vector<uint64_t>(count);
      dataset_.copy(size, slot);
      slots.try_emplace(size, std::move(slot));
    }
  }

  std::for_each(
      std::execution::par, slots.begin(), slots.end(), [&](auto &slot) {
        auto generator = std::ranlux48();
        generator.seed(static_cast<uint_fast64_t>(epoch));
        std::shuffle(slot.second.begin(), slot.second.end(), generator);
      });

  dataset_.on_epoch_begin(epoch);

  for (const auto &[size, slot] : slots) {
    EXPECT_TRUE(dataset_.equal(size, slot));
  }
}

// This test checks whether the retrieval process of data sample finds one with
// the nearest size to the requested value. It also verifies that the retrieved
// data samples are properly recovered in the recycle bin.
TEST_F(DatasetTest, At) {
  const auto epoch = static_cast<uint64_t>(std::rand());

  for (auto [size, count] : counts_) {
    for (; 0 < count; --count) {
      EXPECT_EQ(dataset_.at(size).first, size);
    }
  }

  EXPECT_TRUE(dataset_.empty());

  for (const auto [size, count] : counts_) {
    if (0 < count) {
      EXPECT_EQ(dataset_.size(size, false), count);
      EXPECT_EQ(dataset_.capacity(size, false), count);
      EXPECT_TRUE(dataset_.is_sorted(size, false));
    } else {
      EXPECT_FALSE(dataset_.contains(size, false));
    }
  }

  dataset_.on_epoch_end(epoch);
  EXPECT_TRUE(dataset_.empty(false));

  dataset_.on_epoch_begin(epoch);
  for (auto [size, count] : counts_) {
    for (; 0 < count; --count) {
      EXPECT_EQ(dataset_.at(size).first, size);
    }
  }

  EXPECT_TRUE(dataset_.empty());
}

}  // namespace
