add_executable(
  graph_test
  graph_test.cc)
target_include_directories(
  graph_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  graph_test
  PRIVATE absl::die_if_null
  PRIVATE flatbuffers
  PRIVATE GTest::gtest_main
  PRIVATE OpenMP::OpenMP_CXX)
gtest_discover_tests(graph_test)

add_library(
  numeric_test OBJECT
  numeric_test.cc)
target_include_directories(
  numeric_test
  PRIVATE ${PROJECT_SOURCE_DIR})

add_library(
  polynomial_test OBJECT
  polynomial_test.cc)
target_include_directories(
  polynomial_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  polynomial_test
  PRIVATE absl::check
  PRIVATE absl::core_headers
  PRIVATE absl::die_if_null
  PRIVATE flatbuffers)

add_library(
  scalar_type_test OBJECT
  scalar_type_test.cc)
target_include_directories(
  scalar_type_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  scalar_type_test
  PRIVATE absl::check
  PRIVATE absl::str_format
  PRIVATE flatbuffers)
