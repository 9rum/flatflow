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

add_executable(
  polynomial_test
  polynomial_test.cc)
target_include_directories(
  polynomial_test
  PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(
  polynomial_test
  PRIVATE GTest::gtest_main)
target_compile_options(
  polynomial_test
  PRIVATE -Wall -Wextra)
if(FLATFLOW_ENABLE_ASAN)
  target_compile_options(
    polynomial_test
    PRIVATE -fsanitize=address)
  target_link_options(
    polynomial_test
    PRIVATE -fsanitize=address)
endif()
if(FLATFLOW_ENABLE_UBSAN)
  target_compile_options(
    polynomial_test
    PRIVATE -fsanitize=undefined)
  target_link_options(
    polynomial_test
    PRIVATE -fsanitize=undefined)
endif()
gtest_discover_tests(polynomial_test)
