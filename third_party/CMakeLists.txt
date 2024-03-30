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
set(HAVE_OFF64_T OFF)
set(ABSL_ENABLE_INSTALL ON)
option(gRPC_BUILD_TESTS OFF)
set(gRPC_INSTALL ON)
option(RE2_BUILD_TESTING OFF)


add_subdirectory(abseil-cpp)
add_subdirectory(grpc)

set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
set(_REFLECTION grpc++_reflection)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
set(_GRPC_GRPCPP grpc++)
set(_PROTOBUF_LIBPROTOBUF libprotobuf)

include_directories(${gRPC_DIR}/include)
set(gRPC_DIR "./third_party/grpc/cmake/build")

set(FLATBUFFERS_BUILD_TESTS OFF)
add_subdirectory(flatbuffers)

if(BUILD_TESTING)
  include(GoogleTest)
  add_subdirectory(googletest)
endif()
