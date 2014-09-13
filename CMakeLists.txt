# program to CMakeLists.txt

cmake_minimum_required(VERSION 2.6)

set(target_name logisRegression)

PROJECT(${target_name})

include(/Users/zhangyu/Src/Project/tools/include/dlib/cmake)
include_directories(/Users/zhangyu/Src/Project/tools/include/eigen)

ADD_EXECUTABLE(${target_name} 
  parameter.h
  parameter.cpp
  cmdline.h
  cmdline.cpp
  dataop.h
  dataop.cpp
  util.h
  util.cpp
  linearmodelopt.h
  linearmodelopt.cpp
  logistic_regression.cpp
)

TARGET_LINK_LIBRARIES(${target_name} dlib boost_regex boost_program_options)
