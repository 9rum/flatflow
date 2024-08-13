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

#include <cstdint>
#include <future>
#include <memory>
#include <thread>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/internal/globals.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "grpcpp/grpcpp.h"

#include "flatflow/rpc/communicator.h"

// snoop()
//
// Monitors the signal coming while blocking, then shuts down the server.
//
// Note that the service does nothing in this scope but is passed to manage
// the ownership to match the lifetime with the server.
void snoop(std::future<void> signal, std::unique_ptr<grpc::Server> server,
           std::unique_ptr<flatflow::rpc::CommunicatorServiceImpl> service) {
  signal.get();
  server->Shutdown();
}

// run()
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

  auto handler = std::promise<void>();
  auto signal = handler.get_future();

  auto service = std::make_unique<flatflow::rpc::CommunicatorServiceImpl>(
      data_parallel_size, std::move(handler));

  auto builder = grpc::ServerBuilder();
  auto addr = absl::StrFormat("[::]:%u", port);
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());

  auto server = builder.BuildAndStart();
  if (server == nullptr) {
    LOG(FATAL) << absl::StrFormat("Failed to start server on %s", addr);
  }

  std::thread(snoop, std::move(signal), std::move(server), std::move(service))
      .detach();

  LOG(INFO) << absl::StrFormat("Server listening on %s", addr);
}
