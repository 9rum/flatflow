add_subdirectory(ops)

add_executable(
  ops_test
  ops_test.cc)
target_include_directories(
  ops_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  ops_test
  PRIVATE absl::check
  PRIVATE absl::core_headers
  PRIVATE absl::log
  PRIVATE absl::log_globals
  PRIVATE absl::log_initialize
  PRIVATE absl::log_severity
  PRIVATE absl::time
  PRIVATE flatbuffers
  PRIVATE GTest::gtest_main
  PRIVATE OpenMP::OpenMP_CXX)
gtest_discover_tests(ops_test)
