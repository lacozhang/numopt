cmake_minimum_required(VERSION 2.8.10)

file(GLOB MODELSRC "src/*.cpp")
add_library(model ${MODELSRC})
target_include_directories(model
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)
target_link_libraries(model dataop utils)
