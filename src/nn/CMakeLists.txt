cmake_minimum_required(VERSION 2.8.10)

# support Neural network models
file(GLOB NNSOURCE "src/*.cpp" "include/nn/*.h")
add_library(nn STATIC ${NNSOURCE})
target_include_directories(nn
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)
target_link_libraries(nn ${Boost_LIBRARIES} dataop utils)
