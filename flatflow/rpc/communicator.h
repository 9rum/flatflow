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

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <execution>
#include <future>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/internal/globals.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"

#include "flatflow/rpc/communicator.grpc.fb.h"
#include "flatflow/rpc/communicator_generated.h"
#include "flatflow/rpc/empty_generated.h"
#include "flatflow/scheduler/scheduler.h"

namespace flatflow {

// flatflow::CommunicatorServiceImpl
//
// A `flatflow::CommunicatorServiceImpl` is an intermediary to communicate
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
  using key_type = uint16_t;
  using mapped_type = uint64_t;
  using atomic_mapped_type = std::atomic_uint64_t;

  // Constructors and assignment operators
  //
  // There are only default constructors and assignment operators to allow copy
  // elision, except for opening communication channels to synchronize workers
  // upon initialization. The actual initializations are handled through `Init`.
  explicit CommunicatorServiceImpl() {}

  explicit CommunicatorServiceImpl(mapped_type world_size)
      : world_size_(world_size) {
    fanin_ = 0;

    fanout_.first.reserve(world_size);
    fanout_.second.reserve(world_size);
    for (mapped_type rank = 0; rank < world_size; ++rank) {
      fanout_.first.emplace_back();
      fanout_.second.emplace_back(std::move(fanout_.first[rank].get_future()));
    }
  }

  explicit CommunicatorServiceImpl(const CommunicatorServiceImpl &other) = default;

  CommunicatorServiceImpl &operator=(const CommunicatorServiceImpl &other) = default;

  explicit CommunicatorServiceImpl(CommunicatorServiceImpl &&other) = default;

  CommunicatorServiceImpl &operator=(CommunicatorServiceImpl &&other) = default;

  ~CommunicatorServiceImpl() override {
    if (signal_.get() != 0) {
      LOG(ERROR) << "Failed to send signal to program";
    }
  }

  // CommunicatorServiceImpl::Init()
  //
  // Initializes the training environment.
  grpc::Status Init(grpc::ServerContext *context,
                    const flatbuffers::grpc::Message<InitRequest> *request,
                    flatbuffers::grpc::Message<Empty> *response) override {
    const auto args = request->GetRoot();
    const auto rank = static_cast<std::size_t>(args->rank());

    LOG(INFO) << absl::StrFormat("Init called from %s (rank %u)", context->peer(), rank);
    ++fanin_;

    if (rank == 0) {
      batch_ = 0;
      num_scheduled_batches_ = 0;
      per_replica_batch_size_ = args->global_batch_size() / world_size_;
      num_batches_ =
          (args->sizes()->size() - 1) / args->global_batch_size() + 1;

      const auto order = args->order();
      if (order == 1) {
        if (args->heterogeneous()) {
          return grpc::Status(
              grpc::StatusCode::UNIMPLEMENTED,
              "Support for heterogeneous clusters is not yet implemented");
        }
        scheduler_ = Scheduler<mapped_type, key_type, 1, false>(
            args->sizes(), world_size_, args->global_batch_size(),
            args->micro_batch_size(), args->seed(), args->use_flat_shuffle());
      } else if (order == 2) {
        if (args->heterogeneous()) {
          return grpc::Status(
              grpc::StatusCode::UNIMPLEMENTED,
              "Support for heterogeneous clusters is not yet implemented");
        }
        scheduler_ = Scheduler<mapped_type, key_type, 2, false>(
            args->sizes(), world_size_, args->global_batch_size(),
            args->micro_batch_size(), args->hidden_size(), args->seed(),
            args->use_flat_shuffle());
      } else {
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            absl::StrFormat("Invalid order %u, order should be 1 or 2", order));
      }

      _call_callbacks_on_train_begin();

      // `compare_exchange_weak` tolerates spurious failures, but may yield
      // better performance than `compare_exchange_strong` on some platforms
      // when compare-and-swap is inside a loop.
      // See https://en.cppreference.com/w/cpp/atomic/atomic/compare_exchange.
      auto expected = world_size_;
      while (!fanin_.compare_exchange_weak(expected, 0)) {
        expected = world_size_;
      }

      std::for_each(std::execution::par, fanout_.first.begin(),
                    fanout_.first.end(), [](auto &producer) {
                      producer.set_value(std::move(std::vector<mapped_type>()));
                    });
    }

    auto builder = flatbuffers::grpc::MessageBuilder();
    auto offset = CreateEmpty(builder);
    builder.Finish(offset);
    *response = builder.ReleaseMessage<Empty>();

    fanout_.second[rank].get();

    // The promise-future communication channel is disposable; each worker
    // should reset its own channel after receiving a fanout signal.
    fanout_.first[rank] = std::promise<std::vector<mapped_type>>();
    fanout_.second[rank] = fanout_.first[rank].get_future();

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
    const auto rank = static_cast<std::size_t>(args->rank());

    LOG(INFO) << absl::StrFormat("Broadcast called from %s (rank %u)", context->peer(), rank);
    if (batch_ != 0 || num_scheduled_batches_ != 0) {
      _call_callbacks_on_batch_end(args->rank(), args->costs());
    }
    ++fanin_;

    if (rank == 0) {
      auto expected = world_size_;
      while (!fanin_.compare_exchange_weak(expected, 0)) {
        expected = world_size_;
      }
      if (batch_ != 0 || num_scheduled_batches_ != 0) {
        _call_callbacks_on_batch_end();
      }

      batch_ += num_scheduled_batches_;
      if (batch_ == num_batches_) {
        _call_callbacks_on_epoch_end();
        batch_ = 0;
      }
      if (batch_ == 0) {
        epoch_ = args->epoch();
        _call_callbacks_on_epoch_begin();
      }

      auto schedule = GetSchedule();
      num_scheduled_batches_ =
          (schedule.front().size() - 1) / per_replica_batch_size_ + 1;
      _call_callbacks_on_batch_begin();

      #pragma omp parallel for
      for (mapped_type _rank = 0; _rank < world_size_; ++_rank) {
        fanout_.first[_rank].set_value(std::move(schedule[_rank]));
      }
    }

    auto builder = flatbuffers::grpc::MessageBuilder();
    const auto indices = builder.CreateVector(fanout_.second[rank].get());
    const auto converged = CheckConvergence();
    const auto offset = CreateBroadcastResponse(builder, indices, converged);
    builder.Finish(offset);
    *response = builder.ReleaseMessage<BroadcastResponse>();

    fanout_.first[rank] = std::promise<std::vector<mapped_type>>();
    fanout_.second[rank] = fanout_.first[rank].get_future();

    return grpc::Status::OK;
  }

  // CommunicatorServiceImpl::Finalize()
  //
  // Terminates the training environment.
  grpc::Status Finalize(grpc::ServerContext *context,
                        const flatbuffers::grpc::Message<Empty> *request,
                        flatbuffers::grpc::Message<Empty> *response) override {
    LOG(INFO) << absl::StrFormat("Finalize called from %s", context->peer());

    _call_callbacks_on_train_end();

    auto builder = flatbuffers::grpc::MessageBuilder();
    auto offset = CreateEmpty(builder);
    builder.Finish(offset);
    *response = builder.ReleaseMessage<Empty>();

    signal_ = std::async(std::launch::async, std::raise, SIGTERM);

    return grpc::Status::OK;
  }

 private:
  // CommunicatorServiceImpl::GetSchedule()
  //
  // Gets computation schedule from the scheduler.
  std::vector<std::vector<mapped_type>> GetSchedule() {
    if (scheduler_.index() == 1) {
      return std::get<1>(scheduler_).Schedule();
    }
    return std::get<2>(scheduler_).Schedule();
  }

  // CommunicatorServiceImpl::CheckConvergence()
  //
  // Returns the convergence status of the underlying cost model of the
  // scheduler.
  bool CheckConvergence() const noexcept {
    if (scheduler_.index() == 1) {
      return std::get<1>(scheduler_).converged();
    }
    return std::get<2>(scheduler_).converged();
  }

  // CommunicatorServiceImpl::_call_callbacks_on_batch_begin()
  //
  // Calls every callback's `on_batch_begin` hook.
  void _call_callbacks_on_batch_begin() const noexcept {
    if (scheduler_.index() == 1) {
      std::get<1>(scheduler_).on_batch_begin(batch_);
    } else {
      std::get<2>(scheduler_).on_batch_begin(batch_);
    }
  }

  // CommunicatorServiceImpl::_call_callbacks_on_batch_end()
  //
  // Calls every callback's rank-wise `on_batch_end` hook.
  void _call_callbacks_on_batch_end(mapped_type rank,
                                    const flatbuffers::Vector<double> *costs) {
    if (scheduler_.index() == 1) {
      std::get<1>(scheduler_).on_batch_end(batch_, rank, costs);
    } else {
      std::get<2>(scheduler_).on_batch_end(batch_, rank, costs);
    }
  }

  // CommunicatorServiceImpl::_call_callbacks_on_batch_end()
  //
  // Calls every callback's `on_batch_end` hook.
  void _call_callbacks_on_batch_end() {
    if (scheduler_.index() == 1) {
      std::get<1>(scheduler_).on_batch_end(batch_);
    } else {
      std::get<2>(scheduler_).on_batch_end(batch_);
    }
  }

  // CommunicatorServiceImpl::_call_callbacks_on_epoch_begin()
  //
  // Calls every callback's `on_epoch_begin` hook.
  void _call_callbacks_on_epoch_begin() {
    if (scheduler_.index() == 1) {
      std::get<1>(scheduler_).on_epoch_begin(epoch_);
    } else {
      std::get<2>(scheduler_).on_epoch_begin(epoch_);
    }
  }

  // CommunicatorServiceImpl::_call_callbacks_on_epoch_end()
  //
  // Calls every callback's `on_epoch_end` hook.
  void _call_callbacks_on_epoch_end() {
    if (scheduler_.index() == 1) {
      std::get<1>(scheduler_).on_epoch_end(epoch_);
    } else {
      std::get<2>(scheduler_).on_epoch_end(epoch_);
    }
  }

  // CommunicatorServiceImpl::_call_callbacks_on_train_begin()
  //
  // Calls every callback's `on_train_begin` hook.
  void _call_callbacks_on_train_begin() const noexcept {
    if (scheduler_.index() == 1) {
      std::get<1>(scheduler_).on_train_begin();
    } else {
      std::get<2>(scheduler_).on_train_begin();
    }
  }

  // CommunicatorServiceImpl::_call_callbacks_on_train_end()
  //
  // Calls every callback's `on_train_end` hook.
  void _call_callbacks_on_train_end() const noexcept {
    if (scheduler_.index() == 1) {
      std::get<1>(scheduler_).on_train_end();
    } else {
      std::get<2>(scheduler_).on_train_end();
    }
  }

  mapped_type batch_;
  mapped_type epoch_;
  mapped_type num_batches_;
  mapped_type num_scheduled_batches_;
  mapped_type per_replica_batch_size_;
  mapped_type world_size_;
  atomic_mapped_type fanin_;
  std::future<int> signal_;
  std::pair<std::vector<std::promise<std::vector<mapped_type>>>,
            std::vector<std::future<std::vector<mapped_type>>>>
      fanout_;
  std::variant<std::monostate, Scheduler<mapped_type, key_type, 1, false>,
               Scheduler<mapped_type, key_type, 2, false>>
      scheduler_;
};

// flatflow::run()
//
// Executes the communicator runtime. This routine is invoked from the Python
// frontend via foreign function interface (FFI); that is, there is no direct
// entry point to the communicator runtime and the actual initialization and
// termination are handled through `Init` and `Finalize`, respectively.
void run(uint16_t port, CommunicatorServiceImpl::mapped_type world_size) {
  if (!absl::log_internal::IsInitialized()) {
    absl::InitializeLog();
    absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  }

  auto builder = grpc::ServerBuilder();
  auto addr = absl::StrFormat("[::]:%u", port);
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());

  static auto service = CommunicatorServiceImpl(world_size);
  builder.RegisterService(&service);

  static auto server = builder.BuildAndStart();
  if (server == nullptr) {
    LOG(FATAL) << absl::StrFormat("Failed to start communicator runtime on %s", addr);
  }

  auto handler = [](int signal) { server->Shutdown(); };
  if (std::signal(SIGTERM, handler) == SIG_ERR) {
    LOG(ERROR) << "Failed to change handling of SIGTERM";
  }

  LOG(INFO) << absl::StrFormat("Communicator runtime started on %s", addr);
}

}  // namespace flatflow

#endif  // FLATFLOW_RPC_COMMUNICATOR_H_
