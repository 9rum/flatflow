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

set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(abseil-cpp)

set(FLATBUFFERS_BUILD_TESTS OFF)
add_subdirectory(flatbuffers)

if(BUILD_TESTING)
  include(GoogleTest)
  add_subdirectory(googletest)
endif()

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(protobuf_INSTALL OFF)
set(utf8_range_ENABLE_INSTALL OFF)
set(RE2_BUILD_TESTING OFF)
set(HAVE_OFF64_T OFF)
set(gRPC_BUILD_TESTS OFF)
add_subdirectory(grpc)
