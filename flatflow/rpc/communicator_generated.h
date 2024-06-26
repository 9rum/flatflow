// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_COMMUNICATOR_FLATFLOW_RPC_H_
#define FLATBUFFERS_GENERATED_COMMUNICATOR_FLATFLOW_RPC_H_

#include "flatbuffers/flatbuffers.h"

// Ensure the included flatbuffers.h is the same version as when this file was
// generated, otherwise it may not be compatible.
static_assert(FLATBUFFERS_VERSION_MAJOR == 24 &&
              FLATBUFFERS_VERSION_MINOR == 3 &&
              FLATBUFFERS_VERSION_REVISION == 25,
             "Non-compatible flatbuffers version included");

#include "flatflow/rpc/empty_generated.h"

namespace flatflow {
namespace rpc {

struct InitRequest;
struct InitRequestBuilder;

struct BroadcastRequest;
struct BroadcastRequestBuilder;

struct BroadcastResponse;
struct BroadcastResponseBuilder;

struct InitRequest FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef InitRequestBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_GLOBAL_BATCH_SIZE = 4,
    VT_HIDDEN_SIZE = 6,
    VT_MICRO_BATCH_SIZE = 8,
    VT_ORDER = 10,
    VT_RANK = 12,
    VT_SEED = 14,
    VT_SIZES = 16,
    VT_HETEROGENEOUS = 18,
    VT_USE_FLAT_SHUFFLE = 20
  };
  uint64_t global_batch_size() const {
    return GetField<uint64_t>(VT_GLOBAL_BATCH_SIZE, 0);
  }
  uint64_t hidden_size() const {
    return GetField<uint64_t>(VT_HIDDEN_SIZE, 0);
  }
  uint64_t micro_batch_size() const {
    return GetField<uint64_t>(VT_MICRO_BATCH_SIZE, 0);
  }
  uint64_t order() const {
    return GetField<uint64_t>(VT_ORDER, 0);
  }
  uint64_t rank() const {
    return GetField<uint64_t>(VT_RANK, 0);
  }
  uint64_t seed() const {
    return GetField<uint64_t>(VT_SEED, 0);
  }
  const ::flatbuffers::Vector64<uint16_t> *sizes() const {
    return GetPointer64<const ::flatbuffers::Vector64<uint16_t> *>(VT_SIZES);
  }
  bool heterogeneous() const {
    return GetField<uint8_t>(VT_HETEROGENEOUS, 0) != 0;
  }
  bool use_flat_shuffle() const {
    return GetField<uint8_t>(VT_USE_FLAT_SHUFFLE, 0) != 0;
  }
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_GLOBAL_BATCH_SIZE, 8) &&
           VerifyField<uint64_t>(verifier, VT_HIDDEN_SIZE, 8) &&
           VerifyField<uint64_t>(verifier, VT_MICRO_BATCH_SIZE, 8) &&
           VerifyField<uint64_t>(verifier, VT_ORDER, 8) &&
           VerifyField<uint64_t>(verifier, VT_RANK, 8) &&
           VerifyField<uint64_t>(verifier, VT_SEED, 8) &&
           VerifyOffset64(verifier, VT_SIZES) &&
           verifier.VerifyVector(sizes()) &&
           VerifyField<uint8_t>(verifier, VT_HETEROGENEOUS, 1) &&
           VerifyField<uint8_t>(verifier, VT_USE_FLAT_SHUFFLE, 1) &&
           verifier.EndTable();
  }
};

struct InitRequestBuilder {
  typedef InitRequest Table;
  ::flatbuffers::FlatBufferBuilder64 &fbb_;
  ::flatbuffers::uoffset_t start_;
  void add_global_batch_size(uint64_t global_batch_size) {
    fbb_.AddElement<uint64_t>(InitRequest::VT_GLOBAL_BATCH_SIZE, global_batch_size, 0);
  }
  void add_hidden_size(uint64_t hidden_size) {
    fbb_.AddElement<uint64_t>(InitRequest::VT_HIDDEN_SIZE, hidden_size, 0);
  }
  void add_micro_batch_size(uint64_t micro_batch_size) {
    fbb_.AddElement<uint64_t>(InitRequest::VT_MICRO_BATCH_SIZE, micro_batch_size, 0);
  }
  void add_order(uint64_t order) {
    fbb_.AddElement<uint64_t>(InitRequest::VT_ORDER, order, 0);
  }
  void add_rank(uint64_t rank) {
    fbb_.AddElement<uint64_t>(InitRequest::VT_RANK, rank, 0);
  }
  void add_seed(uint64_t seed) {
    fbb_.AddElement<uint64_t>(InitRequest::VT_SEED, seed, 0);
  }
  void add_sizes(::flatbuffers::Offset64<::flatbuffers::Vector64<uint16_t>> sizes) {
    fbb_.AddOffset(InitRequest::VT_SIZES, sizes);
  }
  void add_heterogeneous(bool heterogeneous) {
    fbb_.AddElement<uint8_t>(InitRequest::VT_HETEROGENEOUS, static_cast<uint8_t>(heterogeneous), 0);
  }
  void add_use_flat_shuffle(bool use_flat_shuffle) {
    fbb_.AddElement<uint8_t>(InitRequest::VT_USE_FLAT_SHUFFLE, static_cast<uint8_t>(use_flat_shuffle), 0);
  }
  explicit InitRequestBuilder(::flatbuffers::FlatBufferBuilder64 &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ::flatbuffers::Offset<InitRequest> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<InitRequest>(end);
    return o;
  }
};

inline ::flatbuffers::Offset<InitRequest> CreateInitRequest(
    ::flatbuffers::FlatBufferBuilder64 &_fbb,
    uint64_t global_batch_size = 0,
    uint64_t hidden_size = 0,
    uint64_t micro_batch_size = 0,
    uint64_t order = 0,
    uint64_t rank = 0,
    uint64_t seed = 0,
    ::flatbuffers::Offset64<::flatbuffers::Vector64<uint16_t>> sizes = 0,
    bool heterogeneous = false,
    bool use_flat_shuffle = false) {
  InitRequestBuilder builder_(_fbb);
  builder_.add_sizes(sizes);
  builder_.add_seed(seed);
  builder_.add_rank(rank);
  builder_.add_order(order);
  builder_.add_micro_batch_size(micro_batch_size);
  builder_.add_hidden_size(hidden_size);
  builder_.add_global_batch_size(global_batch_size);
  builder_.add_use_flat_shuffle(use_flat_shuffle);
  builder_.add_heterogeneous(heterogeneous);
  return builder_.Finish();
}

inline ::flatbuffers::Offset<InitRequest> CreateInitRequestDirect(
    ::flatbuffers::FlatBufferBuilder64 &_fbb,
    uint64_t global_batch_size = 0,
    uint64_t hidden_size = 0,
    uint64_t micro_batch_size = 0,
    uint64_t order = 0,
    uint64_t rank = 0,
    uint64_t seed = 0,
    const std::vector<uint16_t> *sizes = nullptr,
    bool heterogeneous = false,
    bool use_flat_shuffle = false) {
  auto sizes__ = sizes ? _fbb.CreateVector64(*sizes) : 0;
  return flatflow::rpc::CreateInitRequest(
      _fbb,
      global_batch_size,
      hidden_size,
      micro_batch_size,
      order,
      rank,
      seed,
      sizes__,
      heterogeneous,
      use_flat_shuffle);
}

struct BroadcastRequest FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef BroadcastRequestBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_EPOCH = 4,
    VT_RANK = 6,
    VT_COSTS = 8
  };
  uint64_t epoch() const {
    return GetField<uint64_t>(VT_EPOCH, 0);
  }
  uint64_t rank() const {
    return GetField<uint64_t>(VT_RANK, 0);
  }
  const ::flatbuffers::Vector64<double> *costs() const {
    return GetPointer64<const ::flatbuffers::Vector64<double> *>(VT_COSTS);
  }
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_EPOCH, 8) &&
           VerifyField<uint64_t>(verifier, VT_RANK, 8) &&
           VerifyOffset64(verifier, VT_COSTS) &&
           verifier.VerifyVector(costs()) &&
           verifier.EndTable();
  }
};

struct BroadcastRequestBuilder {
  typedef BroadcastRequest Table;
  ::flatbuffers::FlatBufferBuilder64 &fbb_;
  ::flatbuffers::uoffset_t start_;
  void add_epoch(uint64_t epoch) {
    fbb_.AddElement<uint64_t>(BroadcastRequest::VT_EPOCH, epoch, 0);
  }
  void add_rank(uint64_t rank) {
    fbb_.AddElement<uint64_t>(BroadcastRequest::VT_RANK, rank, 0);
  }
  void add_costs(::flatbuffers::Offset64<::flatbuffers::Vector64<double>> costs) {
    fbb_.AddOffset(BroadcastRequest::VT_COSTS, costs);
  }
  explicit BroadcastRequestBuilder(::flatbuffers::FlatBufferBuilder64 &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ::flatbuffers::Offset<BroadcastRequest> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<BroadcastRequest>(end);
    return o;
  }
};

inline ::flatbuffers::Offset<BroadcastRequest> CreateBroadcastRequest(
    ::flatbuffers::FlatBufferBuilder64 &_fbb,
    uint64_t epoch = 0,
    uint64_t rank = 0,
    ::flatbuffers::Offset64<::flatbuffers::Vector64<double>> costs = 0) {
  BroadcastRequestBuilder builder_(_fbb);
  builder_.add_costs(costs);
  builder_.add_rank(rank);
  builder_.add_epoch(epoch);
  return builder_.Finish();
}

inline ::flatbuffers::Offset<BroadcastRequest> CreateBroadcastRequestDirect(
    ::flatbuffers::FlatBufferBuilder64 &_fbb,
    uint64_t epoch = 0,
    uint64_t rank = 0,
    const std::vector<double> *costs = nullptr) {
  auto costs__ = costs ? _fbb.CreateVector64(*costs) : 0;
  return flatflow::rpc::CreateBroadcastRequest(
      _fbb,
      epoch,
      rank,
      costs__);
}

struct BroadcastResponse FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef BroadcastResponseBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_INDICES = 4,
    VT_CONVERGED = 6
  };
  const ::flatbuffers::Vector64<uint64_t> *indices() const {
    return GetPointer64<const ::flatbuffers::Vector64<uint64_t> *>(VT_INDICES);
  }
  bool converged() const {
    return GetField<uint8_t>(VT_CONVERGED, 0) != 0;
  }
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset64Required(verifier, VT_INDICES) &&
           verifier.VerifyVector(indices()) &&
           VerifyField<uint8_t>(verifier, VT_CONVERGED, 1) &&
           verifier.EndTable();
  }
};

struct BroadcastResponseBuilder {
  typedef BroadcastResponse Table;
  ::flatbuffers::FlatBufferBuilder64 &fbb_;
  ::flatbuffers::uoffset_t start_;
  void add_indices(::flatbuffers::Offset64<::flatbuffers::Vector64<uint64_t>> indices) {
    fbb_.AddOffset(BroadcastResponse::VT_INDICES, indices);
  }
  void add_converged(bool converged) {
    fbb_.AddElement<uint8_t>(BroadcastResponse::VT_CONVERGED, static_cast<uint8_t>(converged), 0);
  }
  explicit BroadcastResponseBuilder(::flatbuffers::FlatBufferBuilder64 &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ::flatbuffers::Offset<BroadcastResponse> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<BroadcastResponse>(end);
    fbb_.Required(o, BroadcastResponse::VT_INDICES);
    return o;
  }
};

inline ::flatbuffers::Offset<BroadcastResponse> CreateBroadcastResponse(
    ::flatbuffers::FlatBufferBuilder64 &_fbb,
    ::flatbuffers::Offset64<::flatbuffers::Vector64<uint64_t>> indices = 0,
    bool converged = false) {
  BroadcastResponseBuilder builder_(_fbb);
  builder_.add_indices(indices);
  builder_.add_converged(converged);
  return builder_.Finish();
}

inline ::flatbuffers::Offset<BroadcastResponse> CreateBroadcastResponseDirect(
    ::flatbuffers::FlatBufferBuilder64 &_fbb,
    const std::vector<uint64_t> *indices = nullptr,
    bool converged = false) {
  auto indices__ = indices ? _fbb.CreateVector64(*indices) : 0;
  return flatflow::rpc::CreateBroadcastResponse(
      _fbb,
      indices__,
      converged);
}

}  // namespace rpc
}  // namespace flatflow

#endif  // FLATBUFFERS_GENERATED_COMMUNICATOR_FLATFLOW_RPC_H_
