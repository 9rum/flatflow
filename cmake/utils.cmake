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

function(target_enable_warnings target)
  target_compile_options(
    ${target}
    PRIVATE -Wall -Wextra)
endfunction()

function(target_enable_asan target)
  if(FLATFLOW_ENABLE_ASAN)
    target_compile_options(
      ${target}
      PRIVATE -fsanitize=address)
    target_link_options(
      ${target}
      PRIVATE -fsanitize=address)
  endif()
endfunction()

function(target_enable_ubsan target)
  if(FLATFLOW_ENABLE_UBSAN)
    target_compile_options(
      ${target}
      PRIVATE -fsanitize=undefined)
    target_link_options(
      ${target}
      PRIVATE -fsanitize=undefined)
  endif()
endfunction()

function(target_enable_sanitizers target)
  target_enable_asan(${target})
  target_enable_ubsan(${target})
endfunction()

function(target_enable_lto target)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(
      ${target}
      PRIVATE -flto=auto)
    target_link_options(
      ${target}
      PRIVATE -flto=auto)
  endif()
endfunction()
