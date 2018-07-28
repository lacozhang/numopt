cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

add_executable(datacache "src/txtfeat.cpp")
target_link_libraries(datacache dataop utils glog::glog)
