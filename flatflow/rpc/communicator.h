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

#ifndef FLATFLOW_RPC_COMMUNICATOR_H_
#define FLATFLOW_RPC_COMMUNICATOR_H_

#include <atomic>
#include <csignal>
#include <cstdint>
#include <variant>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/internal/globals.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "grpcpp/grpcpp.h"

#include "flatflow/rpc/communicator.grpc.fb.h"
#include "flatflow/rpc/communicator_generated.h"
#include "flatflow/rpc/empty_generated.h"
#include "flatflow/scheduler/scheduler.h"

namespace flatflow {
namespace rpc {

// flatflow::rpc::CommunicatorServiceImpl
//
// A `flatflow::rpc::CommunicatorServiceImpl` is an intermediary to communicate
// between the scheduler and workers. It is responsible for invoking the
// callbacks exposed by the scheduler and exchanging schedules and costs between
// the scheduler and workers.
//
// Its primitives are based on the syntax of message passing interface (MPI);
// the communicator runtime always starts with `Init` and ends with `Finalize`.
// At the beginning of each scheduling interval, `Broadcast` is called to send
// the schedules to all workers.
class CommunicatorServiceImpl final : public Communicator::Service {
 public:
  // Constructors and assignment operators
  //
  // There are only default constructors and assignment operators to allow copy
  // elision, except for passing the data parallel size to synchronize workers
  // upon initialization. The actual initializations are handled through `Init`.
  explicit CommunicatorServiceImpl() {}

  explicit CommunicatorServiceImpl(uint64_t data_parallel_size)
      : data_parallel_size_(data_parallel_size) {}

  explicit CommunicatorServiceImpl(const CommunicatorServiceImpl &other) = default;

  CommunicatorServiceImpl &operator=(const CommunicatorServiceImpl &other) = default;

  explicit CommunicatorServiceImpl(CommunicatorServiceImpl &&other) = default;

  CommunicatorServiceImpl &operator=(CommunicatorServiceImpl &&other) = default;

  // CommunicatorServiceImpl::Init()
  //
  // Initializes the training environment.
  grpc::Status Init(grpc::ServerContext *context,
                    const flatbuffers::grpc::Message<InitRequest> *request,
                    flatbuffers::grpc::Message<Empty> *response) override {
    const auto args = request->GetRoot();
    const auto rank = args->rank();

    LOG(INFO) << absl::StrFormat("Init called from %s (rank %u)", context->peer(), rank);
    ++fanin_;

    if (rank == 0) {
      const auto order = args->order();

      if (order == 1) {
        if (args->heterogeneous()) {
          return grpc::Status(
              grpc::StatusCode::UNIMPLEMENTED,
              "Support for heterogeneous clusters is not yet implemented");
        }
        scheduler_ =
            flatflow::scheduler::Scheduler<uint64_t, uint16_t, 1, false>(
                args->sizes(), data_parallel_size_, args->global_batch_size(),
                args->micro_batch_size(), args->seed(),
                args->use_flat_shuffle());
      } else if (order == 2) {
        if (args->heterogeneous()) {
          return grpc::Status(
              grpc::StatusCode::UNIMPLEMENTED,
              "Support for heterogeneous clusters is not yet implemented");
        }
        scheduler_ =
            flatflow::scheduler::Scheduler<uint64_t, uint16_t, 2, false>(
                args->sizes(), data_parallel_size_, args->global_batch_size(),
                args->micro_batch_size(), args->hidden_size(), args->seed(),
                args->use_flat_shuffle());
      } else {
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            absl::StrFormat("Invalid order %u, order should be 1 or 2", order));
      }

      // `compare_exchange_weak` tolerates spurious failures, but may yield
      // better performance than `compare_exchange_strong` on some platforms
      // when compare-and-swap is inside a loop.
      // See https://en.cppreference.com/w/cpp/atomic/atomic/compare_exchange.
      auto expected = data_parallel_size_;
      while (!fanin_.compare_exchange_weak(expected, 0)) {
        expected = data_parallel_size_;
      }
    }

    auto offset = CreateEmpty(builder_);
    builder_.Finish(offset);
    *response = builder_.ReleaseMessage<Empty>();

    return grpc::Status::OK;
  }

  // CommunicatorServiceImpl::Broadcast()
  //
  // Broadcasts schedule to all workers. If the scheduler provides
  // profile-guided optimization, the given cost is used to estimate
  // the complexity.
  grpc::Status Broadcast(
      grpc::ServerContext *context,
      const flatbuffers::grpc::Message<BroadcastRequest> *request,
      flatbuffers::grpc::Message<BroadcastResponse> *response) override {
    const auto args = request->GetRoot();
    const auto rank = args->rank();

    LOG(INFO) << absl::StrFormat("Broadcast called from %s (rank %u)", context->peer(), rank);
    ++fanin_;

    return grpc::Status::OK;
  }

  // CommunicatorServiceImpl::Finalize()
  //
  // Terminates the training environment.
  grpc::Status Finalize(grpc::ServerContext *context,
                        const flatbuffers::grpc::Message<Empty> *request,
                        flatbuffers::grpc::Message<Empty> *response) override {
    LOG(INFO) << absl::StrFormat("Finalize called from %s", context->peer());

    std::raise(SIGTERM);

    auto offset = CreateEmpty(builder_);
    builder_.Finish(offset);
    *response = builder_.ReleaseMessage<Empty>();

    return grpc::Status::OK;
  }

 private:
  uint64_t data_parallel_size_;
  uint64_t batch_;
  uint64_t epoch_;
  std::atomic_uint64_t fanin_;
  flatbuffers::grpc::MessageBuilder builder_;
  std::variant<std::monostate,
               flatflow::scheduler::Scheduler<uint64_t, uint16_t, 1, false>,
               flatflow::scheduler::Scheduler<uint64_t, uint16_t, 2, false>>
      scheduler_;
};

// flatflow::rpc::run()
//
// Executes the communicator runtime. This routine is invoked from the Python
// frontend via foreign function interface (FFI); that is, there is no direct
// entry point to the communicator runtime and the actual initialization and
// termination are handled through `Init` and `Finalize`, respectively.
void run(uint16_t port, uint64_t data_parallel_size) {
  if (!absl::log_internal::IsInitialized()) {
    absl::InitializeLog();
    absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  }

  auto builder = grpc::ServerBuilder();
  auto addr = absl::StrFormat("[::]:%u", port);
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());

  auto service = CommunicatorServiceImpl(data_parallel_size);
  builder.RegisterService(&service);

  static auto server = builder.BuildAndStart();
  if (server == nullptr) {
    LOG(FATAL) << absl::StrFormat("Failed to start server on %s", addr);
  }

  auto shutdown = [](int signal) { server->Shutdown(); };
  std::signal(SIGTERM, shutdown);

  LOG(INFO) << absl::StrFormat("Server listening on %s", addr);
}

}  // namespace rpc
}  // namespace flatflow

#endif  // FLATFLOW_RPC_COMMUNICATOR_H_
