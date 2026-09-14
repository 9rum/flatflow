# This should be consistent with the CMake version in pyproject.toml.
cmake_minimum_required(VERSION 3.29)

project(flatflow LANGUAGES CXX)

option(FLATFLOW_BUILD_TESTS "If ON, FlatFlow will build its own tests" ON)
option(FLATFLOW_ENABLE_ASAN "If ON, enable AddressSanitizer (ASan)" OFF)
option(FLATFLOW_ENABLE_UBSAN "If ON, enable UndefinedBehaviorSanitizer (UBSan)" OFF)

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

find_package(OpenMP REQUIRED)

add_subdirectory(third_party)
add_subdirectory(flatflow)

if(BUILD_TESTING AND FLATFLOW_BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()
