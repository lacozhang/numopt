cmake_minimum_required(VERSION 2.8.10 FATAL_ERROR)

#start unit test block
add_executable(lccrf_test "src//lccrf_test.cpp")
target_link_libraries(lccrf_test model crflib dataop utils Boost::unit_test_framework)
add_test(NAME lccrf_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin 
  COMMAND ${CMAKE_BINARY_DIR}/bin/lccrf_test)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/template 
  ${CMAKE_BINARY_DIR}/bin/template 
  COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/lccrf_test.txt 
  ${CMAKE_BINARY_DIR}/bin/lccrf_test.txt
  COPYONLY)

add_library(test_main "src/test_main.cpp")
target_link_libraries(test_main gtest gflags)
