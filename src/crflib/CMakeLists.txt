cmake_minimum_required(VERSION 2.8.10)

set(LCCRFSRCS "src/crftemplate.cpp" "src/lccrfeaturizer.cpp")

add_library(crflib STATIC ${LCCRFSRCS})
target_include_directories(crflib
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

target_link_libraries(crflib utils)

set(TestFiles "test/crf_test.cpp")

foreach(TestUnit ${TestFiles})
	get_filename_component(TestUnitName ${TestUnit} NAME_WE)
	add_executable(${TestUnitName} ${TestUnit})
	target_link_libraries(${TestUnitName} crflib Boost::unit_test_framework Boost::filesystem Boost::log)
	add_test(NAME ${TestUnitName}
	  COMMAND ${TestUnitName})
endforeach(TestUnit)
