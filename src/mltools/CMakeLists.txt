cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

file(GLOB MLSRCS "src/*.cpp")
add_executable(mltools ${MLSRCS})
target_include_directories(dataop
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)
  
target_link_libraries(mltools 
	utils dataop model nn opt ${Boost_LIBRARIES})