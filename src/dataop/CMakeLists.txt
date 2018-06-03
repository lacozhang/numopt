cmake_minimum_required(VERSION 2.8.10)

file(GLOB DATAOPSRC 
	"src/*.cpp")
add_library(dataop STATIC ${DATAOPSRC})
target_include_directories(dataop
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)
target_link_libraries(dataop nn model utils)

set(TestFiles 
	"test/test_dataop.cpp")

foreach(TestUnit ${TestFiles})
	get_filename_component(TestUnitName ${TestUnit} NAME_WE)
	add_executable(${TestUnitName} ${TestUnit})
	target_link_libraries(${TestUnitName} dataop nn Boost::unit_test_framework)
	add_test(NAME ${TestUnitName}
	  COMMAND ${TestUnitName})
endforeach(TestUnit)