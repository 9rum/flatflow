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
  concat_test
  concat_test.cc)
target_include_directories(
  concat_test
  PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(
  concat_test
  PRIVATE GTest::gtest_main)
target_compile_options(
  concat_test
  PRIVATE -Wall -Wextra -Wpedantic)
if(FLATFLOW_ENABLE_ASAN)
  target_compile_options(
    concat_test
    PRIVATE -fsanitize=address)
  target_link_options(
    concat_test
    PRIVATE -fsanitize=address)
endif()
if(FLATFLOW_ENABLE_UBSAN)
  target_compile_options(
    concat_test
    PRIVATE -fsanitize=undefined)
  target_link_options(
    concat_test
    PRIVATE -fsanitize=undefined)
endif()
gtest_add_tests(
  TARGET concat_test)

add_executable(
  extract_test
  extract_test.cc)
target_include_directories(
  extract_test
  PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(
  extract_test
  PRIVATE GTest::gtest_main)
target_compile_options(
  extract_test
  PRIVATE -Wall -Wextra -Wpedantic)
if(FLATFLOW_ENABLE_ASAN)
  target_compile_options(
    extract_test
    PRIVATE -fsanitize=address)
  target_link_options(
    extract_test
    PRIVATE -fsanitize=address)
endif()
if(FLATFLOW_ENABLE_UBSAN)
  target_compile_options(
    extract_test
    PRIVATE -fsanitize=undefined)
  target_link_options(
    extract_test
    PRIVATE -fsanitize=undefined)
endif()
gtest_add_tests(
  TARGET extract_test)

add_executable(
  partition_test
  partition_test.cc)
target_include_directories(
  partition_test
  PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(
  partition_test
  PRIVATE absl::log
  PRIVATE absl::log_initialize
  PRIVATE absl::str_format
  PRIVATE GTest::gtest_main)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  target_link_libraries(
    partition_test
    PRIVATE TBB::tbb)
endif()
target_compile_options(
  partition_test
  PRIVATE -Wall -Wextra -Wpedantic)
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
gtest_add_tests(
  TARGET partition_test)

add_executable(
  reshape_test
  reshape_test.cc)
target_include_directories(
  reshape_test
  PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(
  reshape_test
  PRIVATE GTest::gtest_main)
target_compile_options(
  reshape_test
  PRIVATE -Wall -Wextra -Wpedantic)
if(FLATFLOW_ENABLE_ASAN)
  target_compile_options(
    reshape_test
    PRIVATE -fsanitize=address)
  target_link_options(
    reshape_test
    PRIVATE -fsanitize=address)
endif()
if(FLATFLOW_ENABLE_UBSAN)
  target_compile_options(
    reshape_test
    PRIVATE -fsanitize=undefined)
  target_link_options(
    reshape_test
    PRIVATE -fsanitize=undefined)
endif()
gtest_add_tests(
  TARGET reshape_test)

add_executable(
  shuffle_test
  shuffle_test.cc)
target_include_directories(
  shuffle_test
  PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(
  shuffle_test
  PRIVATE GTest::gtest_main)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  target_link_libraries(
    shuffle_test
    PRIVATE TBB::tbb)
endif()
target_compile_options(
  shuffle_test
  PRIVATE -Wall -Wextra -Wpedantic)
if(FLATFLOW_ENABLE_ASAN)
  target_compile_options(
    shuffle_test
    PRIVATE -fsanitize=address)
  target_link_options(
    shuffle_test
    PRIVATE -fsanitize=address)
endif()
if(FLATFLOW_ENABLE_UBSAN)
  target_compile_options(
    shuffle_test
    PRIVATE -fsanitize=undefined)
  target_link_options(
    shuffle_test
    PRIVATE -fsanitize=undefined)
endif()
gtest_add_tests(
  TARGET shuffle_test)
