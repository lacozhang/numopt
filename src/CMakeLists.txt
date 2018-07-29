# program to CMakeLists.txt

cmake_minimum_required(VERSION 2.8.7)

PROJECT(NumOpt C CXX)
enable_testing()

option(ENABLE_PREFIX_TREE "enable doubld array trie with prefix" OFF)
option(ENABLE_DEBUG "enable debug macro XDEBUG" ON)

if(ENABLE_DEBUG)
  set(CMAKE_CXX_FLAGS_DEBUG "-DXDEBUG")
endif()

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

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(glog 0.3.5 REQUIRED)
find_package(gflags REQUIRED)
find_package(ZeroMQ REQUIRED)
find_package(zlib REQUIRED)

if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
else(ZLIB_FOUND)
  message(FATAL_ERROR "Zlib not found")
endif()

find_package(snappy REQUIRED)

# setup boost for numopt

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.54 REQUIRED COMPONENTS 
  log 
  program_options 
  system
  filesystem
  date_time
  thread 
  unit_test_framework)

if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
else(Boost_FOUND)
  message(FATAL_ERROR "Can not find Boost")
endif()

find_package(Eigen3 3.0.0 REQUIRED)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
else(EIGEN3_FOUND)
    message(FATAL_ERROR "Can not find Eigen3")
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#gtest part
configure_file("thirdparty/CMakeLists.txt.in" googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                 ${CMAKE_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
 include_directories("${gtest_SOURCE_DIR}/include")
endif()

target_include_directories(gtest_main
  PUBLIC $<BUILD_INTERFACE:${gtest_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

add_subdirectory(proto)
add_subdirectory(util)
add_subdirectory(dataop)
add_subdirectory(model)
add_subdirectory(optimizer)

add_subdirectory(data)
add_subdirectory(filter)
add_subdirectory(system)
add_subdirectory(parameter)
add_subdirectory(learner)
add_subdirectory(app/linear_method)

# other library do not depend on ps structure
add_subdirectory(nn)
add_subdirectory(crflib)
add_subdirectory(mltools)
add_subdirectory(textfeaturize)
add_subdirectory(crfeaturize)
add_subdirectory(test)
