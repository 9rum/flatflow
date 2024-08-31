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

#include <atomic>
#include <csignal>
#include <cstdint>
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
  using key_type = uint16_t;
  using mapped_type = uint64_t;
  using atomic_mapped_type = std::atomic_uint64_t;

  // Constructors and assignment operators
  //
  // There are only default constructors and assignment operators to allow copy
  // elision, except for opening communication channels to synchronize workers
  // upon initialization. The actual initializations are handled through `Init`.
  explicit CommunicatorServiceImpl() {}

  explicit CommunicatorServiceImpl(mapped_type data_parallel_size)
      : data_parallel_size_(data_parallel_size) {
    fanin_ = 0;

    const auto _data_parallel_size =
        static_cast<std::size_t>(data_parallel_size);
    fanout_.first.reserve(_data_parallel_size);
    fanout_.second.reserve(_data_parallel_size);
    for (std::size_t rank = 0; rank < _data_parallel_size; ++rank) {
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
      per_replica_batch_size_ = args->global_batch_size() / data_parallel_size_;
      num_batches_ =
          (args->sizes()->size() - 1) / args->global_batch_size() + 1;

      const auto order = args->order();
      if (order == 1) {
        if (args->heterogeneous()) {
          return grpc::Status(
              grpc::StatusCode::UNIMPLEMENTED,
              "Support for heterogeneous clusters is not yet implemented");
        }
        scheduler_ =
            flatflow::scheduler::Scheduler<mapped_type, key_type, 1, false>(
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
            flatflow::scheduler::Scheduler<mapped_type, key_type, 2, false>(
                args->sizes(), data_parallel_size_, args->global_batch_size(),
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
      auto expected = data_parallel_size_;
      while (!fanin_.compare_exchange_weak(expected, 0)) {
        expected = data_parallel_size_;
      }

      for (auto &producer : fanout_.first) {
        producer.set_value(std::move(std::vector<mapped_type>()));
      }
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
      auto expected = data_parallel_size_;
      while (!fanin_.compare_exchange_weak(expected, 0)) {
        expected = data_parallel_size_;
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

      for (std::size_t _rank = 0; _rank < data_parallel_size_; ++_rank) {
        fanout_.first[_rank].set_value(std::move(schedule[_rank]));
      }
    }

    auto builder = flatbuffers::grpc::MessageBuilder();
    auto indices = builder.CreateVector(fanout_.second[rank].get());
    auto offset = CreateBroadcastResponse(builder, indices, true);
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
  inline std::vector<std::vector<mapped_type>> GetSchedule() {
    if (scheduler_.index() == 1) {
      return std::get<1>(scheduler_).Schedule();
    }
    return std::get<2>(scheduler_).Schedule();
  }

  // CommunicatorServiceImpl::_call_callbacks_on_batch_begin()
  //
  // Calls every callback's `on_batch_begin` hook.
  inline void _call_callbacks_on_batch_begin() const noexcept {
    if (scheduler_.index() == 1) {
      std::get<1>(scheduler_).on_batch_begin(batch_);
    } else if (scheduler_.index() == 2) {
      std::get<2>(scheduler_).on_batch_begin(batch_);
    }
  }

  // CommunicatorServiceImpl::_call_callbacks_on_batch_end()
  //
  // Calls every callback's `on_batch_end` hook.
  inline void _call_callbacks_on_batch_end(
      mapped_type rank,
      const flatbuffers::Vector<double> *costs) const noexcept {
    if (scheduler_.index() == 1) {
      std::get<1>(scheduler_).on_batch_end(batch_, rank, costs);
    } else if (scheduler_.index() == 2) {
      std::get<2>(scheduler_).on_batch_end(batch_, rank, costs);
    }
  }

  // CommunicatorServiceImpl::_call_callbacks_on_epoch_begin()
  //
  // Calls every callback's `on_epoch_begin` hook.
  inline void _call_callbacks_on_epoch_begin() {
    if (scheduler_.index() == 1) {
      std::get<1>(scheduler_).on_epoch_begin(epoch_);
    } else if (scheduler_.index() == 2) {
      std::get<2>(scheduler_).on_epoch_begin(epoch_);
    }
  }

  // CommunicatorServiceImpl::_call_callbacks_on_epoch_end()
  //
  // Calls every callback's `on_epoch_end` hook.
  inline void _call_callbacks_on_epoch_end() {
    if (scheduler_.index() == 1) {
      std::get<1>(scheduler_).on_epoch_end(epoch_);
    } else if (scheduler_.index() == 2) {
      std::get<2>(scheduler_).on_epoch_end(epoch_);
    }
  }

  // CommunicatorServiceImpl::_call_callbacks_on_train_begin()
  //
  // Calls every callback's `on_train_begin` hook.
  inline void _call_callbacks_on_train_begin() const noexcept {
    if (scheduler_.index() == 1) {
      std::get<1>(scheduler_).on_train_begin();
    } else if (scheduler_.index() == 2) {
      std::get<2>(scheduler_).on_train_begin();
    }
  }

  // CommunicatorServiceImpl::_call_callbacks_on_train_end()
  //
  // Calls every callback's `on_train_end` hook.
  inline void _call_callbacks_on_train_end() const noexcept {
    if (scheduler_.index() == 1) {
      std::get<1>(scheduler_).on_train_end();
    } else if (scheduler_.index() == 2) {
      std::get<2>(scheduler_).on_train_end();
    }
  }

  mapped_type batch_;
  mapped_type data_parallel_size_;
  mapped_type epoch_;
  mapped_type num_batches_;
  mapped_type num_scheduled_batches_;
  mapped_type per_replica_batch_size_;
  atomic_mapped_type fanin_;
  std::future<int> signal_;
  std::pair<std::vector<std::promise<std::vector<mapped_type>>>,
            std::vector<std::future<std::vector<mapped_type>>>>
      fanout_;
  std::variant<std::monostate,
               flatflow::scheduler::Scheduler<mapped_type, key_type, 1, false>,
               flatflow::scheduler::Scheduler<mapped_type, key_type, 2, false>>
      scheduler_;
};

// flatflow::rpc::run()
//
// Executes the communicator runtime. This routine is invoked from the Python
// frontend via foreign function interface (FFI); that is, there is no direct
// entry point to the communicator runtime and the actual initialization and
// termination are handled through `Init` and `Finalize`, respectively.
void run(uint16_t port,
         CommunicatorServiceImpl::mapped_type data_parallel_size) {
  if (!absl::log_internal::IsInitialized()) {
    absl::InitializeLog();
    absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  }

  auto builder = grpc::ServerBuilder();
  auto addr = absl::StrFormat("[::]:%u", port);
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());

  static auto service = CommunicatorServiceImpl(data_parallel_size);
  builder.RegisterService(&service);

  static auto server = builder.BuildAndStart();
  if (server == nullptr) {
    LOG(FATAL) << absl::StrFormat("Failed to start server on %s", addr);
  }

  auto handler = [](int signal) { server->Shutdown(); };
  if (std::signal(SIGTERM, handler) == SIG_ERR) {
    LOG(ERROR) << "Failed to change handling of SIGTERM";
  }

  LOG(INFO) << absl::StrFormat("Server listening on %s", addr);
}

}  // namespace rpc
}  // namespace flatflow

#endif  // FLATFLOW_RPC_COMMUNICATOR_H_
