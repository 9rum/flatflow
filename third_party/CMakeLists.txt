set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(abseil-cpp)

set(FLATBUFFERS_BUILD_TESTS OFF)
set(FLATBUFFERS_INSTALL OFF)
add_subdirectory(flatbuffers)

if(BUILD_TESTING AND FLATFLOW_BUILD_TESTS)
  include(GoogleTest)
  set(BUILD_GMOCK OFF)
  set(INSTALL_GTEST OFF)
  add_subdirectory(googletest)
endif()
