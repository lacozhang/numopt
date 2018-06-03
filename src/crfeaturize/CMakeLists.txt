cmake_minimum_required(VERSION 2.8.10)

if(ENABLE_PREFIX_TREE)
	add_definitions(-DUSE_PREFIX_TREE)
endif(ENABLE_PREFIX_TREE)

find_package(Boost 1.54 REQUIRED COMPONENTS 
  log 
  program_options 
  system
  filesystem
  date_time
  thread 
  unit_test_framework)

add_executable(crfeat "src/crfeaturize.cpp")
target_link_libraries(crfeat crflib ${Boost_LIBRARIES})
