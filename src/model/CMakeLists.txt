cmake_minimum_required(VERSION 2.8.10)

file(GLOB MODEL_SRCS "src/*.cpp")
file(GLOB MODEL_HDRS "include/model/*.h")
add_library(model ${MODEL_HDRS} ${MODEL_SRCS})
target_include_directories(model
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)
target_link_libraries(model dataop utils)
