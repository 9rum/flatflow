# Copyright 2024 The FlatFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

pybind11_add_module(
  _C
  rpc/controlplane.grpc.fb.cc
  pybind.cc)
target_include_directories(
  _C
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  _C
  PRIVATE absl::check
  PRIVATE absl::flat_hash_map
  PRIVATE absl::log
  PRIVATE absl::log_initialize
  PRIVATE absl::str_format
  PRIVATE flatbuffers
  PRIVATE gRPC::grpc++
  PRIVATE OpenMP::OpenMP_CXX)
target_enable_warnings(_C)
target_enable_sanitizers(_C)
target_enable_lto(_C)
