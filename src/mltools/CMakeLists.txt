cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

file(GLOB MLSRCS "src/mltools.cpp" "src/RunModel.h")
add_executable(mltools ${MLSRCS})
target_link_libraries(mltools optimizer dataop model nn glog::glog)
