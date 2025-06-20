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

#ifndef FLATFLOW_RPC_DISTRIBUTED_H_
#define FLATFLOW_RPC_DISTRIBUTED_H_

#include <grpcpp/grpcpp.h>

#include <csignal>
#include <cstdint>
#include <future>
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

// flatflow::DistributedControlPlane
//
// A `flatflow::DistributedControlPlane` is an intermediary for communication
// between the scheduler and the data plane. It is responsible for exchanging
// the original computation schedule from the data plane with the reordered
// computation schedule and invoking callbacks exposed by the scheduler.
//
// Its primitives are based on the syntax of message passing interface (MPI);
// the control plane always starts with `Init` and ends with `Finalize`.
// At the beginning of each training epoch, `Scatter` is called to reorder
// the computation schedule of the data plane.
class DistributedControlPlane : public ControlPlane::Service {
 public:
  using size_type = typename Scheduler::size_type;

  // Constructors and assignment operators
  //
  // There are only basic constructors and assignment operators to allow copy
  // elision. The actual initialization is handled through `Init`.
  DistributedControlPlane() {}

  DistributedControlPlane(const DistributedControlPlane &other) = default;

  DistributedControlPlane &operator=(const DistributedControlPlane &other) =
      default;

  DistributedControlPlane(DistributedControlPlane &&other) = default;

  DistributedControlPlane &operator=(DistributedControlPlane &&other) = default;

  ~DistributedControlPlane() override {
    // The signal should be first validated as the program may have been
    // terminated via an external signal such as keyboard interrupt without
    // calling `Finalize`.
    if (signal_.valid() && signal_.get() != 0) {
      LOG(ERROR) << "Failed to send signal to the program";
    }
  }

  // DistributedControlPlane::Init()
  //
  // Initializes the training environment.
  grpc::Status Init(grpc::ServerContext *context,
                    const flatbuffers::grpc::Message<InitRequest> *request,
                    flatbuffers::grpc::Message<Empty> *response) override {
    CHECK_NE(context, nullptr);
    CHECK_NE(request, nullptr);
    CHECK_NE(response, nullptr);

    const auto args = request->GetRoot();
    CHECK_NE(args, nullptr);

    rank_ = args->rank();

    // clang-format off
    LOG(INFO) << absl::StrFormat("Init called from %s (rank %u)", context->peer(), rank_);
    // clang-format on

    const auto sizes = args->sizes();
    CHECK_NE(sizes, nullptr);

    data_parallel_world_size_ = args->data_parallel_world_size();
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

  // DistributedControlPlane::Scatter()
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

    // clang-format off
    LOG(INFO) << absl::StrFormat("Scatter called from %s (rank %u)", context->peer(), rank_);
    // clang-format on

    const auto args = request->GetRoot();
    CHECK_NE(args, nullptr);

    // If there is a previous computation schedule, the callback should not be
    // called as it is the beginning of the first epoch.
    if (!indices_.empty()) {
      _call_callbacks_on_epoch_end();
    }

    epoch_ = args->epoch();
    _call_callbacks_on_epoch_begin();

    const auto indices = args->indices();
    CHECK_NE(indices, nullptr);

    auto schedule = std::vector<size_type>(indices->size());
    scheduler_.Schedule(indices->begin(), indices->end(), schedule.begin());

    indices_.resize(schedule.size() / data_parallel_world_size_);
    internal::Scatter(schedule.begin(), schedule.end(), indices_.begin(),
                      data_parallel_world_size_, rank_, global_batch_size_);

    auto builder = flatbuffers::grpc::MessageBuilder();
    const auto resp =
        CreateScatterResponse(builder, builder.CreateVector(indices_));
    builder.Finish(resp);
    *response = builder.ReleaseMessage<ScatterResponse>();

    return grpc::Status::OK;
  }

  // DistributedControlPlane::Finalize()
  //
  // Terminates the training environment.
  grpc::Status Finalize(grpc::ServerContext *context,
                        const flatbuffers::grpc::Message<Empty> *request,
                        flatbuffers::grpc::Message<Empty> *response) override {
    std::ignore = request;
    CHECK_NE(context, nullptr);
    CHECK_NE(response, nullptr);

    // clang-format off
    LOG(INFO) << absl::StrFormat("Finalize called from %s (rank %u)", context->peer(), rank_);
    // clang-format on

    // The launch policy should be `std::launch::async`; otherwise a deadlock
    // will occur.
    signal_ = std::async(std::launch::async, std::raise, SIGTERM);

    _call_callbacks_on_epoch_end();
    _call_callbacks_on_train_end();

    auto builder = flatbuffers::grpc::MessageBuilder();
    const auto empty = CreateEmpty(builder);
    builder.Finish(empty);
    *response = builder.ReleaseMessage<Empty>();

    return grpc::Status::OK;
  }

 protected:
  // DistributedControlPlane::_call_callbacks_on_epoch_begin()
  //
  // Calls every callback's `on_epoch_begin` hook.
  void _call_callbacks_on_epoch_begin() const noexcept {
    scheduler_.on_epoch_begin(epoch_);
  }

  // DistributedControlPlane::_call_callbacks_on_epoch_end()
  //
  // Calls every callback's `on_epoch_end` hook.
  void _call_callbacks_on_epoch_end() const noexcept {
    scheduler_.on_epoch_end(epoch_);
  }

  // DistributedControlPlane::_call_callbacks_on_train_begin()
  //
  // Calls every callback's `on_train_begin` hook.
  void _call_callbacks_on_train_begin() const noexcept {
    scheduler_.on_train_begin();
  }

  // DistributedControlPlane::_call_callbacks_on_train_end()
  //
  // Calls every callback's `on_train_end` hook.
  void _call_callbacks_on_train_end() const noexcept {
    scheduler_.on_train_end();
  }

  size_type data_parallel_world_size_;
  size_type epoch_;
  size_type global_batch_size_;
  size_type rank_;
  std::vector<size_type> indices_;
  std::future<int> signal_;
  Scheduler scheduler_;
};

// flatflow::run()
//
// Executes the control plane. This routine is invoked from the Python frontend
// via foreign function interface (FFI); that is, there is no direct entry point
// to the control plane and the actual initialization and termination are made
// through `Init` and `Finalize`, respectively.
void run(std::uint16_t port) {
  if (!absl::log_internal::IsInitialized()) {
    absl::InitializeLog();
    absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  }

  auto builder = grpc::ServerBuilder();
  const auto addr = absl::StrFormat("[::1]:%u", port);
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());

  static auto service = DistributedControlPlane();
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

#endif  // FLATFLOW_RPC_DISTRIBUTED_H_
