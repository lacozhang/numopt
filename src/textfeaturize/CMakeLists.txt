cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

find_package(Boost 1.54 REQUIRED COMPONENTS 
  log 
  program_options 
  system
  filesystem
  date_time
  thread 
  unit_test_framework)

add_executable(datacache "src/txtfeat.cpp")
target_link_libraries(datacache dataop ${Boost_LIBRARIES} utils)