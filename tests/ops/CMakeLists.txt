# Copyright 2025 The FlatFlow Authors
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

add_subdirectory(internal)

add_executable(
  gemma3_ops_test
  gemma3_ops_test.cc)
target_include_directories(
  gemma3_ops_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  gemma3_ops_test
  PRIVATE absl::check
  PRIVATE absl::flat_hash_map
  PRIVATE absl::log
  PRIVATE absl::log_initialize
  PRIVATE absl::str_format
  PRIVATE flatbuffers
  PRIVATE GTest::gtest_main
  PRIVATE OpenMP::OpenMP_CXX)
target_enable_warnings(gemma3_ops_test)
target_enable_sanitizers(gemma3_ops_test)
gtest_discover_tests(gemma3_ops_test)

add_executable(
  gpt3_ops_test
  gpt3_ops_test.cc)
target_include_directories(
  gpt3_ops_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  gpt3_ops_test
  PRIVATE absl::check
  PRIVATE absl::flat_hash_map
  PRIVATE absl::log
  PRIVATE absl::log_initialize
  PRIVATE absl::str_format
  PRIVATE flatbuffers
  PRIVATE GTest::gtest_main
  PRIVATE OpenMP::OpenMP_CXX)
target_enable_warnings(gpt3_ops_test)
target_enable_sanitizers(gpt3_ops_test)
gtest_discover_tests(gpt3_ops_test)

add_executable(
  llama3_ops_test
  llama3_ops_test.cc)
target_include_directories(
  llama3_ops_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  llama3_ops_test
  PRIVATE absl::check
  PRIVATE absl::flat_hash_map
  PRIVATE absl::log
  PRIVATE absl::log_initialize
  PRIVATE absl::str_format
  PRIVATE flatbuffers
  PRIVATE GTest::gtest_main
  PRIVATE OpenMP::OpenMP_CXX)
target_enable_warnings(llama3_ops_test)
target_enable_sanitizers(llama3_ops_test)
gtest_discover_tests(llama3_ops_test)

add_executable(
  mistral3_ops_test
  mistral3_ops_test.cc)
target_include_directories(
  mistral3_ops_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  mistral3_ops_test
  PRIVATE absl::check
  PRIVATE absl::flat_hash_map
  PRIVATE absl::log
  PRIVATE absl::log_initialize
  PRIVATE absl::str_format
  PRIVATE flatbuffers
  PRIVATE GTest::gtest_main
  PRIVATE OpenMP::OpenMP_CXX)
target_enable_warnings(mistral3_ops_test)
target_enable_sanitizers(mistral3_ops_test)
gtest_discover_tests(mistral3_ops_test)

add_executable(
  opt_ops_test
  opt_ops_test.cc)
target_include_directories(
  opt_ops_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  opt_ops_test
  PRIVATE absl::check
  PRIVATE absl::flat_hash_map
  PRIVATE absl::log
  PRIVATE absl::log_initialize
  PRIVATE absl::str_format
  PRIVATE flatbuffers
  PRIVATE GTest::gtest_main
  PRIVATE OpenMP::OpenMP_CXX)
target_enable_warnings(opt_ops_test)
target_enable_sanitizers(opt_ops_test)
gtest_discover_tests(opt_ops_test)

add_executable(
  phi4_ops_test
  phi4_ops_test.cc)
target_include_directories(
  phi4_ops_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  phi4_ops_test
  PRIVATE absl::check
  PRIVATE absl::flat_hash_map
  PRIVATE absl::log
  PRIVATE absl::log_initialize
  PRIVATE absl::str_format
  PRIVATE flatbuffers
  PRIVATE GTest::gtest_main
  PRIVATE OpenMP::OpenMP_CXX)
target_enable_warnings(phi4_ops_test)
target_enable_sanitizers(phi4_ops_test)
gtest_discover_tests(phi4_ops_test)

add_executable(
  qwen3_ops_test
  qwen3_ops_test.cc)
target_include_directories(
  qwen3_ops_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  qwen3_ops_test
  PRIVATE absl::check
  PRIVATE absl::flat_hash_map
  PRIVATE absl::log
  PRIVATE absl::log_initialize
  PRIVATE absl::str_format
  PRIVATE flatbuffers
  PRIVATE GTest::gtest_main
  PRIVATE OpenMP::OpenMP_CXX)
target_enable_warnings(qwen3_ops_test)
target_enable_sanitizers(qwen3_ops_test)
gtest_discover_tests(qwen3_ops_test)
