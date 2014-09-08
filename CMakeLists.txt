
# program to CMakeLists.txt

cmake_minimum_required(VERSION 2.6)

set(target_name logisRegression)

PROJECT(${target_name})

include(/home/zhangyu/src/dlib/dlib/cmake)

ADD_EXECUTABLE(${target_name} logistic_regression.cpp)
TARGET_LINK_LIBRARIES(${target_name} dlib)