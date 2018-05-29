# program to CMakeLists.txt

cmake_minimum_required(VERSION 2.8.7)

PROJECT(NumOpt C CXX)
enable_testing()

option(ENABLE_PREFIX_TREE "enable doubld array trie with prefix" OFF)

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

if(MSVC)
	list(APPEND CMAKE_EXE_LINKER_FLAGS " /STACK:10000000")
endif(MSVC)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.54 COMPONENTS log
                                   program_options
                                   system
                                   filesystem
                                   date_time
                                   thread
				   unit_test_framework
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

set(UTILSRC cmdline.cpp parameter.cpp util.cpp "util/stringop.cpp")
set(UTILHEADER cmdline.h parameter.h typedef.h util.h "util/stringop.h" "util/producer_consumer.h" "util/threadsafe_limited_queue.h" "util/threadsafe_queue.h" "util/barrier.h")
add_library(utils ${UTILSRC} ${UTILHEADER})

set(DATAOPSRC DataIterator.cpp dataop.cpp IndexData.cpp ModelData.cpp "dataop/vocabulary.cpp")
set(DATAOPHEADER DataIterator.h dataop.h IndexData.h ModelData.h "dataop/vocabulary.h")
add_library(dataop ${DATAOPSRC} ${DATAOPHEADER})

file(GLOB OPTMETHODSRC "optimizer/*.cpp")
file(GLOB OPTMETHODHEADER "optimizer/*.h")
add_library(opt ${OPTMETHODSRC} ${OPTMETHODHEADER})

set(MODELSRC lossfunc.cpp linearmodel.cpp LccrfModel.cpp)
set(MODELHEADER AbstractModel.h linearmodel.h lossfunc.h LccrfModel.h LccrfDataType.h)
add_library(model ${MODELSRC} ${MODELHEADER})

add_executable(datacache datacache.cpp)
target_link_libraries(datacache dataop ${Boost_LIBRARIES} utils)

# support CRF featurizer
file(GLOB CRFEATHEADER "lccrf//*.h")
file(GLOB CRFEATSRC "lccrf//crftemplate.cpp" "lccrf//lccrfeaturizer.cpp")
add_library(crfeatlib ${CRFEATHEADER} ${CRFEATSRC})

list(APPEND CRFEATMAIN "lccrf/crfeaturize.cpp")
add_executable(lcrfeat ${CRFEATMAIN})
target_link_libraries(lcrfeat crfeatlib ${Boost_LIBRARIES})
if(ENABLE_PREFIX_TREE)
	add_definitions(-DUSE_PREFIX_TREE)
endif(ENABLE_PREFIX_TREE)

find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

# support Neural network models
file(GLOB NNHEADER "nn//*.h")
file(GLOB NNSOURCE "nn//*.cpp")
add_library(nnlib ${NNHEADER} ${NNSOURCE})
target_link_libraries(nnlib ${Boost_LIBRARIES})


set(GENERALHEADER RunModel.h)
set(MLTOOLSRC mltools.cpp)
add_executable(mltools ${MLTOOLSRC} ${GENERALHEADER})
target_link_libraries(mltools dataop model nnlib opt utils ${Boost_LIBRARIES})

# new generation data pipeline based on protocol buffer
find_package(Protobuf REQUIRED)


file(GLOB TextParser_SRCS "test/text_parser.*")
set(DATA_EXAMPLE_PROTOS "proto/example.proto")
set(DATA_CONFIG_PROTOS "proto/dataconfig.proto")
protobuf_generate_cpp(DATA_EXAMPLE_PROTO_SRCS DATA_EXAMPLE_PROTO_HDRS ${DATA_EXAMPLE_PROTOS})
list(APPEND TextParser_SRCS ${DATA_EXAMPLE_PROTO_HDRS} ${DATA_EXAMPLE_PROTOS_SRCS})
protobuf_generate_cpp(DATA_CONFIG_PROTO_SRCS DATA_CONFIG_PROTO_HDRS ${DATA_CONFIG_PROTOS})
list(APPEND TextParser_SRCS ${DATA_CONFIG_PROTO_HDRS} ${DATA_CONFIG_PROTO_SRCS})
add_executable(TextParser ${TextParser_SRCS})

target_include_directories(TextParser PUBLIC ${PROTOBUF_INCLUDE_DIRS})
target_link_libraries(TextParser ${PROTOBUF_LIBRARIES})

#start unit test block
add_executable(lccrf_test "test//lccrf_test.cpp")
target_link_libraries(lccrf_test model crfeatlib dataop utils ${Boost_LIBRARIES})
set_target_properties(lccrf_test 
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY 
  ${CMAKE_CURRENT_BINARY_DIR}/testbin)
add_test(NAME lccrf_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testbin 
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testbin/lccrf_test)
configure_file(${PROJECT_SOURCE_DIR}/test/template 
  ${CMAKE_CURRENT_BINARY_DIR}/testbin/template
  COPYONLY)
configure_file(${PROJECT_SOURCE_DIR}/test/lccrf_test.txt 
  ${CMAKE_CURRENT_BINARY_DIR}/testbin/lccrf_test
  COPYONLY)
