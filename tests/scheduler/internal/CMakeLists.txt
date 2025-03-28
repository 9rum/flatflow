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

add_executable(
  partition_test
  partition_test.cc)
target_include_directories(
  partition_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  partition_test
  PRIVATE absl::check
  PRIVATE absl::log
  PRIVATE absl::log_initialize
  PRIVATE absl::str_format
  PRIVATE GTest::gtest_main)
target_compile_options(
  partition_test
  PRIVATE -Wall -Wextra)
if(FLATFLOW_ENABLE_ASAN)
  target_compile_options(
    partition_test
    PRIVATE -fsanitize=address)
  target_link_options(
    partition_test
    PRIVATE -fsanitize=address)
endif()
if(FLATFLOW_ENABLE_UBSAN)
  target_compile_options(
    partition_test
    PRIVATE -fsanitize=undefined)
  target_link_options(
    partition_test
    PRIVATE -fsanitize=undefined)
endif()
gtest_discover_tests(partition_test)

add_executable(
  scatter_test
  scatter_test.cc)
target_include_directories(
  scatter_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  scatter_test
  PRIVATE absl::check
  PRIVATE GTest::gtest_main
  PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(
  scatter_test
  PRIVATE -Wall -Wextra)
if(FLATFLOW_ENABLE_ASAN)
  target_compile_options(
    scatter_test
    PRIVATE -fsanitize=address)
  target_link_options(
    scatter_test
    PRIVATE -fsanitize=address)
endif()
if(FLATFLOW_ENABLE_UBSAN)
  target_compile_options(
    scatter_test
    PRIVATE -fsanitize=undefined)
  target_link_options(
    scatter_test
    PRIVATE -fsanitize=undefined)
endif()
gtest_discover_tests(scatter_test)
