cmake_minimum_required(VERSION 2.8.10)

add_library(system STATIC 
  "include/system/message.h"
  "include/system/van.h"
  "src/message.cpp"
  "src/van.cpp")

target_include_directories(system PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

target_link_libraries(system utils glog::glog)

set(TestFiles 
  "test/test_message.cpp"
  "test/test_van.cpp")

add_executable(system_test ${TestFiles})
target_link_libraries(system_test system test_main)
add_test(NAME system_test
    COMMAND system_test)
