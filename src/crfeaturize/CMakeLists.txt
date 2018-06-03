cmake_minimum_required(VERSION 2.8.10)

if(ENABLE_PREFIX_TREE)
	add_definitions(-DUSE_PREFIX_TREE)
endif(ENABLE_PREFIX_TREE)

add_executable(crfeat "src/*.cpp")
target_link_libraries(crfeat crflib)
