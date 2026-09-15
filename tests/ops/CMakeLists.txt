add_executable(
  scalar_type_test
  scalar_type_test.cc)
target_include_directories(
  scalar_type_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  scalar_type_test
  PRIVATE absl::check
  PRIVATE absl::str_format
  PRIVATE flatbuffers
  PRIVATE GTest::gtest_main)
gtest_discover_tests(scalar_type_test)
