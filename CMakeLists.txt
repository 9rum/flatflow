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

# As of GoogleTest v1.14.0, CMake 3.13 is the minimum supported version.
cmake_minimum_required(VERSION 3.13)

project(flatflow LANGUAGES CXX)

# C++14 is the minimum standard to use Abseil and GoogleTest.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

find_package(OpenMP REQUIRED)

add_subdirectory(third_party)
add_subdirectory(flatflow)
