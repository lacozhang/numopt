cmake_minimum_required(VERSION 2.8.10)

file(GLOB DATAOPSRC
	"src/*.cpp")
add_library(dataop STATIC ${DATAOPSRC})
target_include_directories(dataop
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)
target_link_libraries(dataop nn model utils ${GLOG_LIBRARIES})

set(TestFiles
	"test/test_dataop.cpp")

add_executable(dataop_test ${TestFiles})
target_link_libraries(dataop_test dataop nn ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
add_test(NAME dataop_test
    COMMAND dataop_test)
