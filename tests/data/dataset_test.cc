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
#include <cstdlib>
#include <execution>
#include <map>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/internal/globals.h"
#include "flatbuffers/flatbuffers.h"
#include "gtest/gtest.h"
#include "tests/data/dataset_test_generated.h"

#include "flatflow/aten/generator.h"

namespace {

// A read-only data set used only for testing purpose.
class Dataset : public flatflow::data::Dataset<uint64_t, uint16_t> {
  using super_type = flatflow::data::Dataset<uint64_t, uint16_t>;

 public:
  using super_type::super_type;

  inline bool empty(bool items = true) const {
    return items ? items_.empty() : recyclebin_.empty();
  }

  inline std::size_t size() const noexcept { return size_; }

  inline std::size_t size(uint16_t size, bool items = true) const {
    return items ? items_.at(size).size() : recyclebin_.at(size).size();
  }

  inline std::size_t capacity(uint16_t size, bool items = true) const {
    return items ? items_.at(size).capacity() : recyclebin_.at(size).capacity();
  }

  inline bool contains(uint16_t size, bool items = true) const {
    return items ? items_.contains(size) : recyclebin_.contains(size);
  }

  inline bool is_sorted(uint16_t size, bool items = true) const {
    return items ? std::is_sorted(items_.at(size).cbegin(),
                                  items_.at(size).cend())
                 : std::is_sorted(recyclebin_.at(size).crbegin(),
                                  recyclebin_.at(size).crend());
  }

  inline std::vector<uint64_t> copy(uint16_t size) const {
    auto slot = std::vector<uint64_t>(items_.at(size).size());
    std::copy(items_.at(size).cbegin(), items_.at(size).cend(), slot.begin());
    return slot;
  }

  inline bool equal(uint16_t size, const std::vector<uint64_t> &slot) const {
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

    auto builder = flatbuffers::FlatBufferBuilder();
    auto data__ = builder.CreateVector(data);
    auto offset = CreateSizes(builder, data__);
    builder.Finish(offset);

    auto sizes = GetSizes(builder.GetBufferPointer());
    dataset_ = Dataset(sizes->data(), 0);
  }

  std::map<uint16_t, std::size_t> counts_;
  Dataset dataset_;
};

// This test answers the following questions to see that the inverted indices
// are constructed as intended:
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

  EXPECT_EQ(dataset_.size(), dataset_.max_size());
}

// This test checks whether intra-batch shuffling occurs deterministically for
// an arbitrary value of random seed.
TEST_F(DatasetTest, IntraBatchShuffling) {
  const auto epoch = static_cast<uint64_t>(std::rand());

  auto slots = std::map<uint16_t, std::vector<uint64_t>>();
  for (const auto [size, count] : counts_) {
    if (0 < count) {
      slots.try_emplace(size, std::move(dataset_.copy(size)));
    }
  }

  std::for_each(
      std::execution::par, slots.begin(), slots.end(), [&](auto &slot) {
        auto generator = flatflow::aten::Generator(epoch);
        std::shuffle(slot.second.begin(), slot.second.end(), generator);
      });

  dataset_.on_epoch_begin(epoch);

  for (const auto &[size, slot] : slots) {
    EXPECT_TRUE(dataset_.equal(size, slot));
  }
}

// This test checks whether the basic insertion routine works as intended.
TEST_F(DatasetTest, Insert) {
  std::size_t count = 0;

  const auto samples = dataset_.take<true>(dataset_.size());
  EXPECT_EQ(samples.size(), dataset_.max_size());
  EXPECT_TRUE(dataset_.empty(true));
  EXPECT_FALSE(dataset_.empty(false));
  std::for_each(std::execution::seq, samples.cbegin(), samples.cend(),
                [&](const auto &sample) {
                  dataset_.insert<false>(sample);
                  ++count;
                  EXPECT_EQ(dataset_.size(), count);
                });
  EXPECT_EQ(dataset_.size(), dataset_.max_size());
  EXPECT_TRUE(dataset_.empty(false));

  for (const auto [size, count] : counts_) {
    if (0 < count) {
      EXPECT_EQ(dataset_.size(size), count);
      EXPECT_EQ(dataset_.capacity(size), count);
    } else {
      EXPECT_FALSE(dataset_.contains(size));
    }
  }
}

// This test checks whether the reverse insertion routine works as intended.
TEST_F(DatasetTest, InsertIntoRecycleBin) {
  const auto samples = dataset_.take<true>(dataset_.size());
  EXPECT_EQ(samples.size(), dataset_.max_size());
  EXPECT_TRUE(dataset_.empty(true));
  EXPECT_FALSE(dataset_.empty(false));
  std::for_each(std::execution::seq, samples.cbegin(), samples.cend(),
                [&](const auto &sample) {
                  dataset_.insert<true>(sample);
                  EXPECT_TRUE(dataset_.empty(true));
                });
  EXPECT_TRUE(dataset_.empty(true));
  EXPECT_FALSE(dataset_.empty(false));

  for (const auto [size, count] : counts_) {
    if (0 < count) {
      EXPECT_EQ(dataset_.size(size, false), count);
      EXPECT_EQ(dataset_.capacity(size, false), count);
    } else {
      EXPECT_FALSE(dataset_.contains(size, false));
    }
  }
}

// This test checks whether the bulk insertion routine works as intended.
TEST_F(DatasetTest, InsertRange) {
  const auto samples = dataset_.take<true>(dataset_.size());
  EXPECT_EQ(samples.size(), dataset_.max_size());
  EXPECT_TRUE(dataset_.empty(true));
  EXPECT_FALSE(dataset_.empty(false));
  dataset_.insert_range<false>(samples);
  EXPECT_EQ(dataset_.size(), dataset_.max_size());
  EXPECT_TRUE(dataset_.empty(false));

  for (const auto [size, count] : counts_) {
    if (0 < count) {
      EXPECT_EQ(dataset_.size(size), count);
      EXPECT_EQ(dataset_.capacity(size), count);
    } else {
      EXPECT_FALSE(dataset_.contains(size));
    }
  }
}

// This test checks whether the reverse bulk insertion routine works as
// intended.
TEST_F(DatasetTest, InsertRangeIntoRecycleBin) {
  const auto samples = dataset_.take<true>(dataset_.size());
  EXPECT_EQ(samples.size(), dataset_.max_size());
  EXPECT_TRUE(dataset_.empty(true));
  EXPECT_FALSE(dataset_.empty(false));
  dataset_.insert_range<true>(samples);
  EXPECT_TRUE(dataset_.empty(true));
  EXPECT_FALSE(dataset_.empty(false));

  for (const auto [size, count] : counts_) {
    if (0 < count) {
      EXPECT_EQ(dataset_.size(size, false), count);
      EXPECT_EQ(dataset_.capacity(size, false), count);
    } else {
      EXPECT_FALSE(dataset_.contains(size, false));
    }
  }
}

// This test checks whether the bulk loading routine retrieves data samples as
// intended. It also verifies that the retrieved data samples are properly
// recovered in the recycle bin.
TEST_F(DatasetTest, Take) {
  auto samples = dataset_.take<false>(dataset_.size());
  EXPECT_EQ(samples.size(), dataset_.max_size());
  EXPECT_TRUE(std::is_sorted(samples.cbegin(), samples.cend(),
                             [](const auto &sample, const auto &other) {
                               return sample.first < other.first;
                             }));
  EXPECT_TRUE(dataset_.empty(true));
  EXPECT_FALSE(dataset_.empty(false));

  for (const auto [size, count] : counts_) {
    if (0 < count) {
      EXPECT_EQ(dataset_.size(size, false), count);
      EXPECT_EQ(dataset_.capacity(size, false), count);
    } else {
      EXPECT_FALSE(dataset_.contains(size, false));
    }
  }

  const auto epoch = static_cast<uint64_t>(std::rand());

  dataset_.on_epoch_end(epoch);
  EXPECT_EQ(dataset_.size(), dataset_.max_size());
  dataset_.on_epoch_begin(epoch);

  samples = dataset_.take<false>(dataset_.size());
  EXPECT_EQ(samples.size(), dataset_.max_size());
  EXPECT_TRUE(std::is_sorted(samples.cbegin(), samples.cend(),
                             [](const auto &sample, const auto &other) {
                               return sample.first < other.first;
                             }));
  EXPECT_TRUE(dataset_.empty(true));
  EXPECT_FALSE(dataset_.empty(false));

  for (const auto [size, count] : counts_) {
    if (0 < count) {
      EXPECT_EQ(dataset_.size(size, false), count);
      EXPECT_EQ(dataset_.capacity(size, false), count);
    } else {
      EXPECT_FALSE(dataset_.contains(size, false));
    }
  }
}

// This test checks whether the reverse bulk loading routine retrieves data
// samples as intended.
TEST_F(DatasetTest, TakeWithoutRestoration) {
  auto samples = dataset_.take<true>(dataset_.size());
  EXPECT_EQ(samples.size(), dataset_.max_size());
  EXPECT_TRUE(std::is_sorted(samples.cbegin(), samples.cend(),
                             [](const auto &sample, const auto &other) {
                               return sample.first < other.first;
                             }));
  EXPECT_TRUE(dataset_.empty(true));
  EXPECT_FALSE(dataset_.empty(false));

  for (const auto [size, count] : counts_) {
    if (0 < count) {
      EXPECT_EQ(dataset_.size(size, false), 0);
      EXPECT_EQ(dataset_.capacity(size, false), count);
    } else {
      EXPECT_FALSE(dataset_.contains(size, false));
    }
  }

  dataset_.insert_range<false>(samples);
  EXPECT_EQ(dataset_.size(), dataset_.max_size());
  EXPECT_FALSE(dataset_.empty(true));
  EXPECT_TRUE(dataset_.empty(false));

  samples = dataset_.take<true>(dataset_.size());
  EXPECT_EQ(samples.size(), dataset_.max_size());
  EXPECT_TRUE(std::is_sorted(samples.cbegin(), samples.cend(),
                             [](const auto &sample, const auto &other) {
                               return sample.first < other.first;
                             }));
  EXPECT_TRUE(dataset_.empty(true));
  EXPECT_FALSE(dataset_.empty(false));

  for (const auto [size, count] : counts_) {
    if (0 < count) {
      EXPECT_EQ(dataset_.size(size, false), 0);
      EXPECT_EQ(dataset_.capacity(size, false), count);
    } else {
      EXPECT_FALSE(dataset_.contains(size, false));
    }
  }
}

}  // namespace
