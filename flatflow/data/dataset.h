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

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <absl/container/inlined_vector.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <flatbuffers/vector.h>
#include <omp.h>

#include "flatflow/data/internal/container/btree_map.h"

namespace flatflow {
namespace data {

/// \brief A `flatflow::data::Dataset<I, S>` stores metadata about the index
/// and size of data samples in a given data set. For fast execution of
/// scheduling, a `flatflow::data::Dataset<I, S>` constructs an inverted index
/// in a form of `btree_map<S, std::vector<I>>` and stores the scheduled
/// data samples in another inverted index; the two inverted indices are
/// swapped at the end of each training epoch so that the data samples are
/// restored without any data movement overhead.
///
/// A `flatflow::data::Dataset<I, S>` exposes several callbacks which are
/// invoked at the beginning and end of each batch, epoch and training; these
/// are similar to the callback interface provided by Keras and PyTorch
/// Lightning:
///
///   * Keras callbacks: https://keras.io/guides/writing_your_own_callbacks/
///   * PyTorch Lightning callbacks: https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
///
/// \tparam Index The data type of the values in the inverted index.
/// \tparam Size The data type of the keys in the inverted index.
/// \tparam Compare An (optional) comparison function to sort inverted index,
/// which defaults to `std::less<Size>`.
template <typename Index, typename Size, typename Compare = std::less<Size>>
class Dataset {
 public:
  using key_type    = Size;
  using value_type  = Index;
  using key_compare = Compare;

  /// \brief Constructor to build an inverted index from the relative sizes for
  /// each data sample delivered from the Python frontend.
  /// \param sizes A mapping from an index to the relative size of the
  /// corresponding data sample.
  /// \param seed A random seed used for selective shuffling.
  inline explicit Dataset(const flatbuffers::Vector<key_type, value_type> *sizes, value_type seed) : seed(seed) {
    const auto now = omp_get_wtime();

    // The construction of inverted index goes as follows:
    //
    //   * First, count the number of values for each key to avoid copying of
    //     underlying array within each vector.
    //   * Second, initialize and reserve index slots in an inlined vector
    //     using the count for each key, since B-trees are inherently hard to
    //     be parallelized; such ahead-of-time construction of index slots
    //     allows us to parallelize the reservation phase and access an index
    //     slot in constant time.
    //   * Third, insert indices into the index slots.
    //   * Finally, construct an inverted index by inserting the index slots
    //     into a B-tree.
    constexpr auto kIndexSlotSpace = static_cast<std::size_t>(1 << std::numeric_limits<key_type>::digits);
              auto counts          = absl::InlinedVector<value_type, kIndexSlotSpace>(kIndexSlotSpace, 0);

    // Unlike counts and slots whose lengths are known at compile time (e.g.,
    // 65536 for 16-bit key type), the length of sizes is unpredictable so we
    // partially unroll loops over sizes.
    #pragma omp unroll partial
    for (value_type index = 0; index < sizes->size(); ++index) {
      const auto size = static_cast<std::size_t>(sizes->Get(index));
      ++counts.at(size);
    }

    auto slots = absl::InlinedVector<std::vector<value_type>, kIndexSlotSpace>(kIndexSlotSpace);

    #pragma omp parallel for
    for (std::size_t size = 0; size < counts.size(); ++size) {
      const auto count = counts.at(size);
      if (0 < count) {
        slots.at(size).reserve(static_cast<std::size_t>(count));
      }
    }

    #pragma omp unroll partial
    for (value_type index = 0; index < sizes->size(); ++index) {
      const auto size = static_cast<std::size_t>(sizes->Get(index));
      slots.at(size).emplace_back(index);
    }

    #pragma omp unroll full
    for (std::size_t size = 0; size < slots.size(); ++size) {
      const auto slot = slots.at(size);
      if (0 < slot.size()) {
        items.emplace(static_cast<key_type>(size), std::move(slot));
      }
    }

    LOG(INFO) << absl::StrFormat("Construction of inverted index took %f seconds", omp_get_wtime() - now);
  }

  /// \brief A callback to be called at the beginning of a training batch.
  /// \param batch The index of batch within the current epoch.
  inline void on_batch_begin(value_type batch) const noexcept {}

  /// \brief A callback to be called at the end of a training batch.
  /// \param batch The index of batch within the current epoch.
  inline void on_batch_end(value_type batch) const noexcept {}

  /// \brief A callback to be called at the beginning of an epoch.
  /// \param epoch The index of epoch.
  inline void on_epoch_begin(value_type epoch) {
    const auto now = omp_get_wtime();

    // At the beginning of each epoch, a `flatflow::data::Dataset<I, S>`
    // shuffles between data samples with the same size, which we call
    // intra-batch shuffling. The details of intra-batch shuffling are as
    // follows:
    //
    //   * First, access each index slot in the inverted index. This can be
    //     parallelized since there is no data dependency between any couple of
    //     index slots.
    //   * Second, deterministically shuffle each index slot to ensure the
    //     reproducibility of training. As PyTorch's distributed sampler does,
    //     set the random seed to the sum of seed and epoch; a pseudorandom
    //     number generator based on subtract-with-carry is adopted to produce
    //     high-quality random numbers.
    thread_local auto generator = std::ranlux48();

    #pragma omp parallel for
    for (auto &item : items) {
      generator.seed(static_cast<uint_fast64_t>(seed + epoch));
      std::shuffle(item.second.begin(), item.second.end(), generator);
    }

    LOG(INFO) << absl::StrFormat("Epoch: %d intra-batch shuffling took %f seconds", epoch, omp_get_wtime() - now);
  }

  /// \brief A callback to be called at the end of an epoch.
  /// \param epoch The index of epoch.
  inline void on_epoch_end(value_type epoch) {
    internal::container::swap(items, recyclebin);
  }

  /// \brief A callback to be called at the beginning of training.
  inline void on_train_begin() const noexcept {}

  /// \brief A callback to be called at the end of training.
  inline void on_train_end() const noexcept {}

 protected:
  internal::container::btree_map<
      key_type, std::vector<value_type>, key_compare,
      std::allocator<std::pair<const key_type, std::vector<value_type>>>,
      /*TargetNodeSize=*/512> items, recyclebin;
  value_type seed;
};

}  // namespace data
}  // namespace flatflow

#endif  // FLATFLOW_DATA_DATASET_H_
