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

add_subdirectory(internal/algorithm)

add_executable(
  scheduler_test
  scheduler_test.cc)
target_include_directories(
  scheduler_test
  PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(
  scheduler_test
  PRIVATE absl::btree
  PRIVATE absl::inlined_vector
  PRIVATE absl::log
  PRIVATE absl::log_initialize
  PRIVATE absl::str_format
  PRIVATE BLAS::BLAS
  PRIVATE flatbuffers
  PRIVATE GTest::gtest_main
  PRIVATE OpenMP::OpenMP_CXX)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  target_link_libraries(
    scheduler_test
    PRIVATE TBB::tbb)
endif()
gtest_add_tests(
  TARGET scheduler_test)
