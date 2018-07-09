cmake_minimum_required(VERSION 2.8.10)

add_library(data STATIC 
  "include/data/common.h"
  "include/data/text_parser.h"
  "include/data/info_parser.h"
  "include/data/slot_reader.h"
  "include/data/stream_reader.h"
  "src/text_parser.cpp"
  "src/info_parser.cpp"
  "src/slot_reader.cpp"
  "src/stream_reader.cpp"
  "src/common.cpp")

target_include_directories(data PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

target_link_libraries(data proto utils glog::glog)

set(TestFiles 
  "test/text_parser_test.cpp"
  "test/stream_reader_test.cpp")

foreach(TestUnit ${TestFiles})
  get_filename_component(TestUnitName ${TestUnit} NAME_WE)
  add_executable(${TestUnitName} ${TestUnit})
  target_link_libraries(${TestUnitName} data test_main)
  add_test(NAME ${TestUnitName}
  COMMAND ${TestUnitName})
endforeach(TestUnit)
