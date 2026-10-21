add_library(
  numeric_test OBJECT
  numeric_test.cc)
target_include_directories(
  numeric_test
  PRIVATE ${PROJECT_SOURCE_DIR})

add_executable(
  polynomial_test
  polynomial_test.cc)
target_include_directories(
  polynomial_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  polynomial_test
  PRIVATE absl::check
  PRIVATE absl::core_headers
  PRIVATE flatbuffers
  PRIVATE GTest::gtest_main)
gtest_discover_tests(polynomial_test)

add_library(
  scalar_type_test OBJECT
  scalar_type_test.cc)
target_include_directories(
  scalar_type_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  scalar_type_test
  PRIVATE absl::check
  PRIVATE flatbuffers)

add_executable(
  view_test
  view_test.cc)
target_include_directories(
  view_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  view_test
  PRIVATE flatbuffers
  PRIVATE GTest::gtest_main)
gtest_discover_tests(view_test)
