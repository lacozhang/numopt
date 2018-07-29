cmake_minimum_required(VERSION 2.8.10)

add_executable(linear_method
  "include/linear_method/async_sgd.h"
  "include/linear_method/darlin.h"
  "include/linear_method/learning_rate.h"
  "include/linear_method/loss.h"
  "include/linear_method/model_evaluation.h"
  "include/linear_method/penalty.h"
  "src/main.cpp")

target_include_directories(linear_method PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

target_link_libraries(linear_method system parameter filter learner util)
