
# program to CMakeLists.txt

cmake_minimum_required(VERSION 2.6)

set(target_name logisRegression)

PROJECT(${target_name})

include(/Users/zhangyu/Src/Project/tools/include/dlib/cmake)

ADD_EXECUTABLE(${target_name} 
  parameter.h
  parameter.cpp
  cmdline.h
  cmdline.cpp
  logistic_regression.cpp
)
TARGET_LINK_LIBRARIES(${target_name} dlib)
