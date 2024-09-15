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

# CMake 3.18 is the minimum supported version to use imported target for BLAS.
cmake_minimum_required(VERSION 3.18)

project(flatflow LANGUAGES CXX)

include(CTest)

option(FLATFLOW_BUILD_TESTS "If ON, FlatFlow will build its own tests" OFF)

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  # Intel has contributed to PSTL headers with a dependency on tbb/task.h, then
  # deprecated it in a later TBB release. One has to either use an older version
  # of TBB or suppress the warning by defining TBB_SUPPRESS_DEPRECATED_MESSAGES.
  find_package(TBB REQUIRED)
  add_compile_definitions(TBB_SUPPRESS_DEPRECATED_MESSAGES)
endif()

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Torch imports CUDA, so there is no need to specify this with project or
# enable_language.
find_package(Torch REQUIRED)

add_subdirectory(third_party)
add_subdirectory(flatflow)

if(BUILD_TESTING AND FLATFLOW_BUILD_TESTS)
  add_subdirectory(tests)
endif()
