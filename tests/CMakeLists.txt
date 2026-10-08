add_library(
  types_test OBJECT
  types_test.cc)
target_include_directories(
  types_test
  PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(
  types_test
  PRIVATE flatbuffers)

add_subdirectory(ops)
