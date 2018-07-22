cmake_minimum_required(VERSION 2.8.10)

add_library(parameter STATIC 
  "include/parameter/parameter.h"
  "src/parameter.cpp")

target_include_directories(parameter PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

target_link_libraries(parameter utils glog::glog)

set(TestFiles 
  "test/test_parameter.cpp")

add_executable(parameter_test ${TestFiles})
target_link_libraries(parameter_test parameter test_main)
add_test(NAME parameter_test
    COMMAND parameter_test)
