cmake_minimum_required(VERSION 2.8.10)

file(GLOB LCCRFSRCS "src/*.cpp")
add_library(crflib STATIC ${LCCRFSRCS})
target_include_directories(crflib
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

target_link_libraries(crflib utils)

file(GLOB TestFiles "test/*.cpp")
# test case
find_package(Boost 1.54 REQUIRED
  COMPONENTS unit_test_framework
  filesystem
  log)

foreach(TestUnit ${TestFiles})
	get_filename_component(TestUnitName ${TestUnit} NAME_WE)
	add_executable(${TestUnitName} ${TestUnit})
	target_link_libraries(${TestUnitName} ${Boost_LIBRARIES} crflib)
	add_test(NAME ${TestUnitName}
	  COMMAND ${TestUnitName})
endforeach(TestUnit)
