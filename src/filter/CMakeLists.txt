cmake_minimum_required(VERSION 2.8.10)

add_library(filter STATIC 
  "include/filter/filter.h"
  "src/filter.cpp")

target_include_directories(filter PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

target_link_libraries(filter utils glog::glog data)

set(TestFiles 
  "test/test_filter.cpp")

add_executable(filter_test ${TestFiles})
target_link_libraries(filter_test filter test_main)
add_test(NAME filter_test
    COMMAND filter_test)
