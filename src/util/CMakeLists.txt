cmake_minimum_required(VERSION 2.8.10)

file(GLOB UTIL_SRC "src/*.cpp")

add_library(utils STATIC ${UTIL_SRC})

target_include_directories(utils 
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

set(TestFiles "test/test_stringop.cpp")

foreach(TestUnit ${TestFiles})
	get_filename_component(TestUnitName ${TestUnit} NAME_WE)
	add_executable(${TestUnitName} ${TestUnit})
	target_link_libraries(${TestUnitName} 
		utils 
		Boost::unit_test_framework)
	add_test(NAME ${TestUnitName}
	  COMMAND ${TestUnitName})
endforeach(TestUnit)
