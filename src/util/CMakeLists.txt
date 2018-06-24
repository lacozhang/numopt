cmake_minimum_required(VERSION 2.8.10)

file(GLOB UTIL_SRC "src/*.cpp")
file(GLOB UTIL_HDR "include/util/*.h")

add_library(utils STATIC ${UTIL_SRC} ${UTIL_HDR})

target_include_directories(utils 
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

target_link_libraries(utils proto)

set(TestFiles "test/test_stringop.cpp"
  "test/test_misc.cpp")

foreach(TestUnit ${TestFiles})
  get_filename_component(TestUnitName ${TestUnit} NAME_WE)
  add_executable(${TestUnitName} ${TestUnit})
  target_link_libraries(${TestUnitName} utils test_main)
  add_test(NAME ${TestUnitName} 
    COMMAND ${TestUnitName})
endforeach(TestUnit)
