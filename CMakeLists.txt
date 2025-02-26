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

# This should be consistent with the CMake version in pyproject.toml.
cmake_minimum_required(VERSION 3.16)

project(flatflow LANGUAGES CXX)

include(CTest)

option(FLATFLOW_BUILD_TESTS "If ON, FlatFlow will build its own tests" OFF)
option(FLATFLOW_ENABLE_ASAN "If ON, enable AddressSanitizer" OFF)
option(FLATFLOW_ENABLE_UBSAN "If ON, enable UndefinedBehaviorSanitizer" OFF)

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

find_package(OpenMP REQUIRED)

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(third_party)
add_subdirectory(flatflow)

if(BUILD_TESTING AND FLATFLOW_BUILD_TESTS)
  add_subdirectory(tests)
endif()
