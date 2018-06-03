cmake_minimum_required(VERSION 2.8.10)

add_library(dataop_new STATIC 
  "src/text_parser.cpp")

target_include_directories(dataop_new PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

target_link_libraries(dataop_new datastruct utils)

set(TestFiles 
	"test/text_parser_test.cpp")

foreach(TestUnit ${TestFiles})
	get_filename_component(TestUnitName ${TestUnit} NAME_WE)
	add_executable(${TestUnitName} ${TestUnit})
	target_link_libraries(${TestUnitName} dataop_new Boost::unit_test_framework)
	add_test(NAME ${TestUnitName}
	  COMMAND ${TestUnitName})
endforeach(TestUnit)