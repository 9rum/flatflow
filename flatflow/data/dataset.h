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

#include <cblas.h>
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
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/vector.h"

#include "flatflow/data/internal/container/btree_map.h"
#include "flatflow/data/internal/types.h"

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
  requires(internal::Unsigned<Index> && internal::Unsigned<Size>)
class Dataset {
 public:
  using container_type = internal::container::btree_map<
      Size, std::vector<Index>, std::less<Size>,
      std::allocator<std::pair<const Size, std::vector<Index>>>,
      /*TargetNodeSize=*/512>;
  using key_type = Size;
  using mapped_type = Index;
  using value_type = std::pair<Size, Index>;
  using size_type = std::size_t;

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
  explicit Dataset() {}

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
  explicit Dataset(const flatbuffers::Vector<key_type, mapped_type> *sizes,
                   const mapped_type &seed)
      : seed_(seed) {
    CHECK_NE(sizes, nullptr);

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
    auto counts = std::vector<double>(kIndexSlotSpace, 0.0);

    #pragma omp declare reduction(vadd : std::vector<double> : cblas_daxpy( \
            static_cast<int>(omp_in.size()), 1.0, omp_in.data(), 1,         \
                omp_out.data(), 1)) initializer(omp_priv = omp_orig)

    #pragma omp parallel for reduction(vadd : counts)
    for (mapped_type index = 0; index < sizes->size(); ++index) {
      const auto size = static_cast<std::size_t>(sizes->Get(index));
      ++counts[size];
    }

    auto slots = absl::InlinedVector<std::vector<mapped_type>, kIndexSlotSpace>(
        kIndexSlotSpace);

    #pragma omp parallel for
    for (std::size_t size = 0; size < counts.size(); ++size) {
      const auto count = static_cast<std::size_t>(counts[size]);
      if (0 < count) {
        slots[size].reserve(count);
      }
    }

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
    for (mapped_type index = 0; index < sizes->size(); ++index) {
      const auto size = static_cast<std::size_t>(sizes->Get(index));
      slots[size].emplace_back(index);
    }

    #pragma omp unroll full
    for (std::size_t size = 0; size < slots.size(); ++size) {
      auto &slot = slots[size];
      if (0 < slot.size()) {
        items_.try_emplace(static_cast<key_type>(size), std::move(slot));
      }
    }

    max_size_ = static_cast<size_type>(sizes->size());
    size_ = static_cast<size_type>(sizes->size());

    LOG(INFO) << absl::StrFormat("Construction of inverted index took %fs", omp_get_wtime() - now);
  }

  explicit Dataset(const Dataset &other) = default;

  Dataset &operator=(const Dataset &other) = default;

  explicit Dataset(Dataset &&other) = default;

  Dataset &operator=(Dataset &&other) = default;

  // Dataset::operator[]()
  //
  // Returns a data sample with the nearest size to the given size from inverted
  // index. This is equivalent to call `at()`.
  inline value_type operator[](const key_type &size) { return at(size); }

  // Dataset::at()
  //
  // Finds a data sample with the same, or at least nearest size to the given
  // size from inverted index.
  inline value_type at(const key_type &size) {
    CHECK_NE(size_, 0);

    // The retrieval process of a data sample is described below:
    //
    // * First, find lower bound for the given size from inverted index. To find
    //   the item with the nearest size, compare the size of the found item with
    //   its precedence if necessary. Since the actual size of the found item
    //   may be different from the given size, the index of the found item is
    //   returned along with its size.
    // * Once the item has been found, select and remove an index from its index
    //   slot. For efficient removal, it always choose the last index in the
    //   index slot. Even though the choice of index within a given index slot
    //   is deterministic, the training sequence is guaranteed to be randomized
    //   since each index slot is shuffled at the beginning of each training
    //   epoch. If the index has been removed, store the index in another
    //   inverted index (i.e., the `recycle bin`) for efficient recovery of
    //   inverted index at the end of training epoch.
    auto item = items_.lower_bound(size);
    if (item != items_.begin()) {
      const auto prev = std::prev(item);
      if (item == items_.end() || size - prev->first < item->first - size) {
        item = prev;
      }
    }

    return take(item);
  }

  // Dataset::take()
  //
  // Takes the first `n` data samples from the inverted index with bounds
  // checking. This ensures that the retrieved data samples are sorted in
  // order of size.
  std::vector<value_type> take(size_type n) {
    CHECK_LE(n, size_);

    auto items = std::vector<value_type>();
    items.reserve(n);

    for (auto item = items_.begin(); 0 < n; item = items_.begin()) {
      if (n < item->second.size()) {
        for (; 0 < n; --n) {
          items.emplace_back(take(item));
        }
      } else {
        for (; 1 < item->second.size(); --n) {
          items.emplace_back(take(item));
        }
        items.emplace_back(take(item));
        --n;
      }
    }

    return items;
  }

  // Dataset::size()
  //
  // Returns the number of data samples in the inverted index.
  inline size_type size() const noexcept { return size_; }

  // Dataset::max_size()
  //
  // Returns the maximum possible number of data samples in the inverted index.
  inline size_type max_size() const noexcept { return max_size_; }

  // Dataset::on_batch_begin()
  //
  // A callback to be called at the beginning of a training batch.
  void on_batch_begin([[maybe_unused]] const mapped_type &batch) const noexcept {}

  // Dataset::on_batch_end()
  //
  // A callback to be called at the end of a training batch.
  void on_batch_end([[maybe_unused]] const mapped_type &batch) const noexcept {}

  // Dataset::on_epoch_begin()
  //
  // A callback to be called at the beginning of an epoch.
  inline void on_epoch_begin(const mapped_type &epoch) {
    const auto now = omp_get_wtime();

    // At the beginning of each epoch, a `flatflow::data::Dataset<>`
    // shuffles between data samples with the same size, which we call
    // intra-batch shuffling. The details are as follows:
    //
    // * First, access each index slot in the inverted index. This can be
    //   parallelized since there is no data dependency between any couple of
    //   index slots.
    // * Second, deterministically shuffle each index slot to ensure
    //   reproducibility of training. As PyTorch's distributed sampler does,
    //   set the random seed to the sum of seed and epoch. A pseudo-random
    //   number generator based on 32-bit Mersenne Twister algorithm is adopted,
    //   just like in PyTorch.
    std::for_each(
        std::execution::par, items_.begin(), items_.end(), [&](auto &item) {
          auto generator = std::mt19937();
          generator.seed(static_cast<uint_fast32_t>(seed_ + epoch));
          std::shuffle(item.second.begin(), item.second.end(), generator);
        });

    LOG(INFO) << absl::StrFormat("Epoch: %u intra-batch shuffling took %fs", epoch, omp_get_wtime() - now);
  }

  // Dataset::on_epoch_end()
  //
  // A callback to be called at the end of an epoch.
  inline void on_epoch_end([[maybe_unused]] const mapped_type &epoch) {
    // At the end of an epoch, the inverted index must be empty.
    CHECK_EQ(size_, 0);

    internal::container::swap(items_, recyclebin_);
    size_ = max_size_;
  }

  // Dataset::on_train_begin()
  //
  // A callback to be called at the beginning of training.
  void on_train_begin() const noexcept {}

  // Dataset::on_train_end()
  //
  // A callback to be called at the end of training.
  void on_train_end() const noexcept {}

 protected:
  // Dataset::take()
  //
  // Takes a data sample from item at `position`.
  value_type take(container_type::iterator position) {
    // Take the last index from item and restore it to recycle bin. There are
    // four possible cases for this:
    //
    // * If there is no equivalent size in recycle bin, create a new index slot
    //   with the index and reserve its capacity as in the constructor to avoid
    //   copying of the underlying array. Note that it is the same to check
    //   whether the size and capacity of an index slot in items are equal to
    //   each other as to search for the equivalent size in recycle bin, since
    //   any index slots with the same size in items and recycle bin are
    //   mutually exclusive.
    // * A special case occurs when the capacity of the index slot is one, where
    //   its node handle can be extracted from items and moved to recycle bin
    //   without allocating a new index slot.
    // * If the size already exists in recycle bin, then simply append the index
    //   to the corresponding index slot.
    // * Finally, if the index slot becomes empty, delete it from items.
    const auto size = position->first;
    const auto index = position->second.back();

    if (position->second.capacity() == 1) {
      recyclebin_.insert(std::move(items_.extract(position)));
    } else if (position->second.size() == position->second.capacity()) {
      position->second.pop_back();
      auto slot = std::vector<mapped_type>();
      slot.reserve(position->second.capacity());
      slot.emplace_back(index);
      recyclebin_.try_emplace(size, std::move(slot));
    } else {
      position->second.pop_back();
      if (position->second.empty()) {
        items_.erase(position);
      }
      recyclebin_.at(size).emplace_back(index);
    }

    --size_;

    return std::make_pair(size, index);
  }

  inline value_type take(container_type::reverse_iterator position) {
    return take(std::next(position).base());
  }

  size_type max_size_;
  size_type size_;
  mapped_type seed_;
  container_type items_;
  container_type recyclebin_;
};

}  // namespace data
}  // namespace flatflow

#endif  // FLATFLOW_DATA_DATASET_H_
