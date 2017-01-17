# program to CMakeLists.txt

cmake_minimum_required(VERSION 2.8.7)

PROJECT(NumOpt C CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

if(MSVC)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else(MSVC)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-std=c++11"    SUPPORT_CXX11)
    set(CMAKE_CXX_FLAGS "-Wall -Wno-unknown-pragmas")
    if(SUPPORT_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif()
endif(MSVC)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.54 COMPONENTS log
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
target_link_libraries(datacache ${Boost_LIBRARIES} dataop utils)

set(GENERALHEADER RunModel.h)
set(MLTOOLSRC mltools.cpp)
add_executable(mltools ${MLTOOLSRC} ${GENERALHEADER})
target_link_libraries(mltools ${Boost_LIBRARIES} dataop model opt utils)
