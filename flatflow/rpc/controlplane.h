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

#ifndef FLATFLOW_RPC_CONTROLPLANE_H_
#define FLATFLOW_RPC_CONTROLPLANE_H_

#include <grpcpp/grpcpp.h>

#include <csignal>
#include <cstdint>
#include <future>
#include <iterator>
#include <tuple>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/internal/globals.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/grpc.h"

#include "flatflow/rpc/controlplane.grpc.fb.h"
#include "flatflow/rpc/controlplane_generated.h"
#include "flatflow/rpc/empty_generated.h"
#include "flatflow/scheduler/internal/scatter.h"
#include "flatflow/scheduler/scheduler.h"

namespace flatflow {

// flatflow::ControlPlaneServiceImpl
//
// A `flatflow::ControlPlaneServiceImpl` is an intermediary to communicate
// between the scheduler and data plane. It is responsible for exchanging the
// original computation schedule from the data plane with the reordered
// computation schedule and invoking callbacks exposed by the scheduler.
//
// Its primitives are based on the syntax of message passing interface (MPI);
// the control plane always starts with `Init` and ends with `Finalize`.
// At the beginning of each training epoch, `Scatter` is called to reorder
// the computation schedule of the data plane.
class ControlPlaneServiceImpl final : public ControlPlane::Service {
 public:
  using size_type = typename Scheduler::size_type;

  // Constructors and assignment operators
  //
  // There are only basic constructors and assignment operators to allow copy
  // elision, except for the one to open communication channels to synchronize
  // the data plane upon initialization. The actual initialization is handled
  // through `Init`.
  ControlPlaneServiceImpl() {}

  ControlPlaneServiceImpl(size_type data_parallel_world_size)
      : data_parallel_world_size_(data_parallel_world_size) {
    producers_.reserve(data_parallel_world_size);
    consumers_.reserve(data_parallel_world_size);

    for (size_type rank = 0; rank < data_parallel_world_size; ++rank) {
      producers_.emplace_back();
      consumers_.emplace_back(producers_[rank].get_future());
    }
  }

  ControlPlaneServiceImpl(const ControlPlaneServiceImpl &other) = default;

  ControlPlaneServiceImpl &operator=(const ControlPlaneServiceImpl &other) =
      default;

  ControlPlaneServiceImpl(ControlPlaneServiceImpl &&other) = default;

  ControlPlaneServiceImpl &operator=(ControlPlaneServiceImpl &&other) = default;

  ~ControlPlaneServiceImpl() override {
    // The signal should be first validated as the program may have been
    // terminated via an external signal such as keyboard interrupt without
    // calling `Finalize`.
    if (signal_.valid() && signal_.get() != 0) {
      LOG(ERROR) << "Failed to send signal to the program";
    }
  }

  // ControlPlaneServiceImpl::Init()
  //
  // Initializes the training environment.
  grpc::Status Init(grpc::ServerContext *context,
                    const flatbuffers::grpc::Message<InitRequest> *request,
                    flatbuffers::grpc::Message<Empty> *response) override {
    CHECK_NE(context, nullptr);
    CHECK_NE(request, nullptr);
    CHECK_NE(response, nullptr);

    LOG(INFO) << absl::StrFormat("Init called from %s", context->peer());

    const auto args = request->GetRoot();
    CHECK_NE(args, nullptr);

    const auto sizes = args->sizes();
    CHECK_NE(sizes, nullptr);

    global_batch_size_ = args->global_batch_size();
    scheduler_ = Scheduler(data_parallel_world_size_, global_batch_size_,
                           args->micro_batch_size(), sizes->begin(),
                           sizes->end(), args->graph());

    _call_callbacks_on_train_begin();

    auto builder = flatbuffers::grpc::MessageBuilder();
    const auto empty = CreateEmpty(builder);
    builder.Finish(empty);
    *response = builder.ReleaseMessage<Empty>();

    return grpc::Status::OK;
  }

  // ControlPlaneServiceImpl::Scatter()
  //
  // Exchanges the given computation schedule with the reordered
  // computation schedule from the scheduler.
  grpc::Status Scatter(
      grpc::ServerContext *context,
      const flatbuffers::grpc::Message<ScatterRequest> *request,
      flatbuffers::grpc::Message<ScatterResponse> *response) override {
    CHECK_NE(context, nullptr);
    CHECK_NE(request, nullptr);
    CHECK_NE(response, nullptr);

    const auto args = request->GetRoot();
    CHECK_NE(args, nullptr);

    const auto rank = args->rank();

    // clang-format off
    LOG(INFO) << absl::StrFormat("Scatter called from %s (rank %u)", context->peer(), rank);
    // clang-format on

    if (rank == 0) {
      if (!indices_.empty()) {
        _call_callbacks_on_epoch_end();
      }

      epoch_ = args->epoch();
      _call_callbacks_on_epoch_begin();

      const auto indices = args->indices();
      CHECK_NE(indices, nullptr);

      indices_.resize(indices->size());
      scheduler_.Schedule(indices->begin(), indices->end(), indices_.begin());

      for (size_type rank = 0; rank < data_parallel_world_size_; ++rank) {
        producers_[rank].set_value();
      }
    }

    consumers_[rank].get();

    // The promise-future communication channel is disposable; each worker
    // should reset its own channel after receiving a fanout signal.
    producers_[rank] = std::promise<void>();
    consumers_[rank] = producers_[rank].get_future();

    auto indices =
        std::vector<size_type>(indices_.size() / data_parallel_world_size_);
    internal::Scatter(indices_.begin(), indices_.end(), indices.begin(),
                      data_parallel_world_size_, rank, global_batch_size_);

    auto builder = flatbuffers::grpc::MessageBuilder();
    const auto resp =
        CreateScatterResponse(builder, builder.CreateVector(indices));
    builder.Finish(resp);
    *response = builder.ReleaseMessage<ScatterResponse>();

    return grpc::Status::OK;
  }

  // ControlPlaneServiceImpl::Finalize()
  //
  // Terminates the training environment.
  grpc::Status Finalize(grpc::ServerContext *context,
                        const flatbuffers::grpc::Message<Empty> *request,
                        flatbuffers::grpc::Message<Empty> *response) override {
    std::ignore = request;
    CHECK_NE(context, nullptr);
    CHECK_NE(response, nullptr);

    LOG(INFO) << absl::StrFormat("Finalize called from %s", context->peer());

    // The launch policy should be `std::launch::async`; otherwise a deadlock
    // will occur.
    signal_ = std::async(std::launch::async, std::raise, SIGTERM);

    _call_callbacks_on_train_end();

    auto builder = flatbuffers::grpc::MessageBuilder();
    const auto empty = CreateEmpty(builder);
    builder.Finish(empty);
    *response = builder.ReleaseMessage<Empty>();

    return grpc::Status::OK;
  }

 private:
  // ControlPlaneServiceImpl::_call_callbacks_on_epoch_begin()
  //
  // Calls every callback's `on_epoch_begin` hook.
  void _call_callbacks_on_epoch_begin() const noexcept {
    scheduler_.on_epoch_begin(epoch_);
  }

  // ControlPlaneServiceImpl::_call_callbacks_on_epoch_end()
  //
  // Calls every callback's `on_epoch_end` hook.
  void _call_callbacks_on_epoch_end() const noexcept {
    scheduler_.on_epoch_end(epoch_);
  }

  // ControlPlaneServiceImpl::_call_callbacks_on_train_begin()
  //
  // Calls every callback's `on_train_begin` hook.
  void _call_callbacks_on_train_begin() const noexcept {
    scheduler_.on_train_begin();
  }

  // ControlPlaneServiceImpl::_call_callbacks_on_train_end()
  //
  // Calls every callback's `on_train_end` hook.
  void _call_callbacks_on_train_end() const noexcept {
    scheduler_.on_train_end();
  }

  size_type data_parallel_world_size_;
  size_type epoch_;
  size_type global_batch_size_;
  std::vector<size_type> indices_;
  std::vector<std::promise<void>> producers_;
  std::vector<std::future<void>> consumers_;
  std::future<int> signal_;
  Scheduler scheduler_;
};

// flatflow::run()
//
// Executes the control plane. This routine is invoked from the Python frontend
// via foreign function interface (FFI); that is, there is no direct entry point
// to the control plane and the actual initialization and termination are made
// through `Init` and `Finalize`, respectively.
void run(uint16_t port,
         typename ControlPlaneServiceImpl::size_type data_parallel_world_size) {
  if (!absl::log_internal::IsInitialized()) {
    absl::InitializeLog();
    absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  }

  auto builder = grpc::ServerBuilder();
  const auto addr = absl::StrFormat("[::]:%u", port);
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());

  static auto service = ControlPlaneServiceImpl(data_parallel_world_size);
  builder.RegisterService(&service);

  static auto server = builder.BuildAndStart();
  CHECK_NE(server, nullptr);

  auto handler = [](int signal) {
    std::ignore = signal;
    server->Shutdown();
  };
  if (std::signal(SIGTERM, handler) == SIG_ERR) {
    LOG(ERROR) << "Failed to change handling of SIGTERM";
  }

  LOG(INFO) << absl::StrFormat("Control plane started on %s", addr);
}

}  // namespace flatflow

#endif  // FLATFLOW_RPC_CONTROLPLANE_H_
