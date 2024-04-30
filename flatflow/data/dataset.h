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

#ifndef FLATFLOW_DATA_DATASET_H_
#define FLATFLOW_DATA_DATASET_H_

#include <omp.h>

#include <algorithm>
#include <execution>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/vector.h"

#include "flatflow/base/types.h"
#include "flatflow/data/internal/container/btree_map.h"

namespace flatflow {
namespace data {

// flatflow::data::Dataset<>
//
// A `flatflow::data::Dataset<I, S>` stores metadata about the index and size of
// data samples in a given data set. For fast execution of scheduling, a
// `flatflow::data::Dataset<I, S>` constructs an inverted index in a form of
// `btree_map<S, std::vector<I>>` and stores the scheduled data samples in
// another inverted index; the two inverted indices are swapped at the end of
// each training epoch so that the data samples are recovered without any data
// movement overhead.
//
// This exposes several callbacks which are invoked at the beginning and end of
// each batch, epoch, and training; these are similar to the callback interface
// provided by Keras and PyTorch Lightning:
//
// * Keras callbacks: https://keras.io/guides/writing_your_own_callbacks/
// * PyTorch Lightning callbacks: https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
template <typename Index, typename Size>
  requires(flatflow::base::Unsigned<Index> && flatflow::base::Unsigned<Size>)
class Dataset {
  using key_type = Size;
  using value_type = Index;

 public:
  // Constructors and assignment operators
  //
  // In addition to a constructor to build an inverted index,
  // a `flatflow::data::Dataset<>` supports a default constructor for
  // declaration, as well as copy and move constructors and assignment
  // operators.
  //
  // Note that even if a copy/move constructor or assignment operator is called,
  // the data set is actually direct-initialized by copy elision.
  // See https://en.cppreference.com/w/cpp/language/copy_elision.
  inline explicit Dataset() {}

  // Constructor to build an inverted index from the relative sizes for each
  // data sample delivered from the Python frontend.
  //
  // CAVEATS
  //
  // This constructor is optimized for key types under 32 bits. It stores index
  // slots in a temporary inlined vector using offsets and constructs the
  // inverted index at once, which is fast but memory-intensive. For key types
  // over 16 bits, this may bring too much memory pressure and the constructor
  // needs to be specialized.
  inline explicit Dataset(
      const flatbuffers::Vector<key_type, value_type> *sizes,
      const value_type &seed)
      : seed_(seed) {
    const auto now = omp_get_wtime();

    // The construction of inverted index goes as follows:
    //
    // * First, count the number of values for each key to avoid copying of
    //   underlying array within each index slot.
    // * Second, initialize and reserve index slots in an inlined vector using
    //   the count for each key, since B-trees are inherently hard to be
    //   parallelized; such ahead-of-time construction of index slots allows us
    //   to parallelize the reservation phase and access an index slot in
    //   constant time.
    // * Third, insert indices into the index slots.
    // * Finally, construct an inverted index by inserting the index slots into
    //   a B-tree.
    constexpr auto kIndexSlotSpace =
        static_cast<std::size_t>(1 << std::numeric_limits<key_type>::digits);
    auto counts =
        absl::InlinedVector<std::size_t, kIndexSlotSpace>(kIndexSlotSpace, 0);

    // Unlike counts and slots whose lengths are known at compile time (e.g.,
    // 65536 for 16-bit key type), the length of sizes is unpredictable so we
    // partially unroll loops over sizes.
    //
    // CAVEATS
    //
    // As of GCC 11.4.0, the unroll construct of OpenMP is ignored with an
    // unknown pragma warning on compilation, regardless of whether its clause
    // is full or partial. That is, we have to define our own portable loop
    // unrolling macros.
    #pragma omp unroll partial
    for (value_type index = 0; index < sizes->size(); ++index) {
      const auto size = static_cast<std::size_t>(sizes->Get(index));
      ++counts.at(size);
    }

    auto slots = absl::InlinedVector<std::vector<value_type>, kIndexSlotSpace>(
        kIndexSlotSpace);

    #pragma omp parallel for
    for (std::size_t size = 0; size < counts.size(); ++size) {
      const auto count = counts.at(size);
      if (0 < count) {
        slots.at(size).reserve(count);
      }
    }

    #pragma omp unroll partial
    for (value_type index = 0; index < sizes->size(); ++index) {
      const auto size = static_cast<std::size_t>(sizes->Get(index));
      slots.at(size).emplace_back(index);
    }

    #pragma omp unroll full
    for (std::size_t size = 0; size < slots.size(); ++size) {
      auto &slot = slots.at(size);
      if (0 < slot.size()) {
        items_.try_emplace(static_cast<key_type>(size), std::move(slot));
      }
    }

    LOG(INFO) << absl::StrFormat("Construction of inverted index took %fs", omp_get_wtime() - now);
  }

  inline explicit Dataset(const Dataset &other) = default;

  inline Dataset &operator=(const Dataset &other) = default;

  inline explicit Dataset(Dataset &&other) = default;

  inline Dataset &operator=(Dataset &&other) = default;

  // Dataset::operator[]()
  //
  // Returns a data sample with the nearest size to the given size from inverted
  // index. This is equivalent to call `find()`.
  inline std::pair<value_type, key_type> operator[](const key_type &size) {
    return find(size);
  }

  // Dataset::find()
  //
  // Finds a data sample with the same, or at least nearest size to the given
  // size from inverted index.
  inline std::pair<value_type, key_type> find(const key_type &size) {
    // The retrieval process of a data sample is described below:
    //
    // * First, find lower bound for the given size from inverted index. To find
    //   the item with the nearest size, compare the size of the found item with
    //   its precedence if necessary. Since the actual size of the found item
    //   may be different from the given size, it returns the index of the found
    //   item along with its size.
    // * Once the item has been found, select and remove an index from its index
    //   slot. For efficient removal, it always choose the last index in the
    //   index slot. Even though the choice of index within a given index slot
    //   is deterministic, the training sequence is guaranteed to be randomized
    //   since each index slot is shuffled at the beginning of each training
    //   epoch. If the index has been removed, store the index in another
    //   inverted index (i.e., the `recycle bin`) for efficient recovery of
    //   inverted index at the end of training epoch. There are four possible
    //   cases for this:
    //
    //   * If there is no equivalent size in recycle bin, create a new index
    //     slot with the index and reserve its capacity as in the constructor to
    //     avoid copying of the underlying array. Note that it is the same to
    //     check whether the size and capacity of an index slot in items are
    //     equal to each other as to search for the equivalent size in recycle
    //     bin, since any index slots with the same size in items and recycle
    //     bin are mutually exclusive.
    //   * A special case occurs when the capacity of the index slot is one,
    //     where its node handle can be extracted from items and moved to
    //     recycle bin without allocating a new index slot.
    //   * If the size already exists in recycle bin, then simply append the
    //     index to the corresponding index slot.
    //   * Finally, if the index slot in items becomes empty, delete it from
    //     items.
    auto item = items_.lower_bound(size);
    if (item != items_.begin()) {
      const auto prev = std::prev(item);
      if (item == items_.end() || size - prev->first < item->first - size) {
        item = prev;
      }
    }

    const auto found = item->first;
    const auto index = item->second.back();

    if (item->second.capacity() == 1) {
      recyclebin_.insert(std::move(items_.extract(item)));
    } else if (item->second.size() == item->second.capacity()) {
      item->second.pop_back();
      auto slot = std::vector<value_type>();
      slot.reserve(item->second.capacity());
      slot.emplace_back(index);
      recyclebin_.try_emplace(found, std::move(slot));
    } else {
      item->second.pop_back();
      if (item->second.empty()) {
        items_.erase(item);
      }
      recyclebin_.at(found).emplace_back(index);
    }

    return std::make_pair(index, found);
  }

  // Dataset::on_batch_begin()
  //
  // A callback to be called at the beginning of a training batch.
  inline void on_batch_begin(
      [[maybe_unused]] const value_type &batch) const noexcept {}

  // Dataset::on_batch_end()
  //
  // A callback to be called at the end of a training batch.
  inline void on_batch_end(
      [[maybe_unused]] const value_type &batch) const noexcept {}

  // Dataset::on_epoch_begin()
  //
  // A callback to be called at the beginning of an epoch.
  inline void on_epoch_begin(const value_type &epoch) {
    const auto now = omp_get_wtime();

    // At the beginning of each epoch, a `flatflow::data::Dataset<>`
    // shuffles between data samples with the same size, which we call
    // intra-batch shuffling. The details are as follows:
    //
    // * First, access each index slot in the inverted index. This can be
    //   parallelized since there is no data dependency between any couple of
    //   index slots.
    // * Second, deterministically shuffle each index slot to ensure the
    //   reproducibility of training. As PyTorch's distributed sampler does, set
    //   the random seed to the sum of seed and epoch; a pseudorandom number
    //   generator based on subtract-with-carry is adopted to produce
    //   high-quality random numbers.
    std::for_each(
        std::execution::par, items_.begin(), items_.end(), [&](auto &item) {
          auto generator = std::ranlux48();
          generator.seed(static_cast<uint_fast64_t>(seed_ + epoch));
          std::shuffle(item.second.begin(), item.second.end(), generator);
        });

    LOG(INFO) << absl::StrFormat("Epoch: %u intra-batch shuffling took %fs", epoch, omp_get_wtime() - now);
  }

  // Dataset::on_epoch_end()
  //
  // A callback to be called at the end of an epoch.
  inline void on_epoch_end([[maybe_unused]] const value_type &epoch) {
    internal::container::swap(items_, recyclebin_);
  }

  // Dataset::on_train_begin()
  //
  // A callback to be called at the beginning of training.
  inline void on_train_begin() const noexcept {}

  // Dataset::on_train_end()
  //
  // A callback to be called at the end of training.
  inline void on_train_end() const noexcept {}

 protected:
  value_type seed_;
  internal::container::btree_map<
      key_type, std::vector<value_type>, std::less<key_type>,
      std::allocator<std::pair<const key_type, std::vector<value_type>>>,
      /*TargetNodeSize=*/512>
      items_, recyclebin_;
};

}  // namespace data
}  // namespace flatflow

#endif  // FLATFLOW_DATA_DATASET_H_
