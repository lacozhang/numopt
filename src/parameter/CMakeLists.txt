cmake_minimum_required(VERSION 2.8.10)

add_library(parameter STATIC 
  "include/parameter/parameter.h"
  "include/parameter/kv_vector.h"
  "include/parameter/kv_map.h"
  "include/parameter/kv_layer.h"
  "src/parameter.cpp")

target_include_directories(parameter PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

target_link_libraries(parameter utils system filter glog::glog)

set(TestFiles 
  "test/kv_layer_perf_ps.cpp"
  "test/kv_map_perf_ps.cpp"
  "test/kv_vector_buffer_ps.cpp"
  "test/kv_vector_ps.cpp"
  "test/kv_layer_ps.cpp"
  "test/kv_map_ps.cpp")

foreach(SRC_PATH ${TestFiles})
  get_filename_component(SRC_NAME ${SRC_PATH} NAME_WE)
  add_executable(${SRC_NAME} ${SRC_PATH})
  target_link_libraries(${SRC_NAME} parameter utils)
endforeach()
