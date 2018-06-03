cmake_minimum_required(VERSION 2.8.10)

file(GLOB UTIL_SRC "src/*.cpp")

add_library(utils ${UTIL_SRC})

target_include_directories(utils 
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)
