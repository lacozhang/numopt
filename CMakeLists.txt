# program to CMakeLists.txt

cmake_minimum_required(VERSION 2.8.7)

PROJECT(NumOpt C CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.56 COMPONENTS log
                                   program_options
                                   system
                                   filesystem
                                   date_time
                                   thread
                        REQUIRED)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIR})
else(Boost_FOUND)
    message(FATAL_ERROR "Can not find Boost")
endif()

find_package(Eigen3 3.0.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
else(EIGEN3_FOUND)
    message(FATAL_ERROR "Can not find Eigen3")
endif()

set(UTILSRC cmdline.cpp parameter.cpp util.cpp)
set(UTILHEADER cmdline.h parameter.h typedef.h util.h)
add_library(utils ${UTILSRC} ${UTILHEADER})

set(DATAOPSRC DataIterator.cpp dataop.cpp IndexData.cpp ModelData.cpp)
set(DATAOPHEADER DataIterator.h dataop.h IndexData.h ModelData.h)
add_library(dataop ${DATAOPSRC} ${DATAOPHEADER})

set(OPTMETHODSRC opt.cpp sgd.cpp)
set(OPTMETHODHEADER opt.h sgd.h)
add_library(opt ${OPTMETHODSRC} ${OPTMETHODHEADER})

set(MODELSRC lossfunc.cpp linearmodel.cpp)
set(MODELHEADER AbstractModel.h linearmodel.h lossfunc.h)
add_library(model ${MODELSRC} ${MODELHEADER})

add_executable(datacache datacache.cpp)
target_link_libraries(datacache ${Boost_LIBRARIES} utils dataop)

set(GENERALHEADER RunModel.h)
add_executable(mltools logistic_regression.cpp ${GENERALHEADER})
target_link_libraries(mltools ${Boost_LIBRARIES} utils dataop model opt)
