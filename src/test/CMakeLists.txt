cmake_minimum_required(VERSION 2.8.10 FATAL_ERROR)

find_package(Boost 1.54 REQUIRED
  COMPONENTS unit_test_framework)
#start unit test block
add_executable(lccrf_test "src//lccrf_test.cpp")
target_link_libraries(lccrf_test model crfeatlib dataop utils ${Boost_LIBRARIES})
set_target_properties(lccrf_test 
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY 
  ${CMAKE_CURRENT_BINARY_DIR}/testbin)
add_test(NAME lccrf_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testbin 
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testbin/lccrf_test)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/template 
  ${CMAKE_CURRENT_BINARY_DIR}/testbin/template 
  COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/lccrf_test.txt 
  ${CMAKE_CURRENT_BINARY_DIR}/testbin/lccrf_test
  COPYONLY)
