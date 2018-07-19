cmake_minimum_required(VERSION 2.8.10)

add_library(system STATIC 
  "include/system/message.h"
  "include/system/van.h"
  "include/system/postoffice.h"
  "include/system/manager.h"
  "include/system/assigner.h"
  "include/system/env.h"
  "include/system/heartbeat_info.h"
  "src/message.cpp"
  "src/van.cpp"
  "src/postoffice.cpp"
  "src/manager.cpp"
  "src/assigner.cpp"
  "src/env.cpp"
  "src/heartbeat_info.cpp")

target_include_directories(system PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

target_link_libraries(system utils glog::glog data filter)

set(TestFiles 
  "test/test_message.cpp"
  "test/test_van.cpp")

add_executable(system_test ${TestFiles})
target_link_libraries(system_test system test_main)
add_test(NAME system_test
    COMMAND system_test)
