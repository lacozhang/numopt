cmake_minimum_required(VERSION 2.8.10)

project(newdataop C CXX)

add_library(dataop_new STATIC 
  "src/text_parser.cpp")
target_include_directories(dataop_new PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

target_link_libraries(dataop_new boost
  datastruct)
target_link_libraries(dataop_new
  utils)

# test case
find_package(Boost 1.54 REQUIRED
  COMPONENTS unit_test_framework)

add_executable(testTextParser "test/text_parser_test.cpp")
target_link_libraries(testTextParser dataop_new ${Boost_LIBRARIES})
add_test(NAME testTextParser
  COMMAND testTextParser)
