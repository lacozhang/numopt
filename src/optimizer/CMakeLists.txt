cmake_minimum_required(VERSION 2.8.10)

file(GLOB OPTSRC "src/*.cpp")
add_library(opt ${OPTSRC})
target_include_directories(opt
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)
target_link_libraries(opt utils)
