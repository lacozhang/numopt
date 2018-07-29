cmake_minimum_required(VERSION 2.8.10)

add_library(learner STATIC 
  "include/learner/workload_pool.h"
  "include/learner/sgd.h"
  "include/learner/bcd.h"
  "src/workload_pool.cpp"
  "src/sgd.cpp"
  "src/bcd.cpp")

target_include_directories(learner PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

target_link_libraries(learner system parameter filter utils glog::glog)

set(TestFiles 
  "test/test_learner.cpp")

add_executable(learner_test ${TestFiles})
target_link_libraries(learner_test learner test_main)
add_test(NAME learner_test
    COMMAND learner_test)
