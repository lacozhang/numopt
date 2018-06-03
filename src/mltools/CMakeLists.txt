cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

file(GLOB MLSRCS "src/*.cpp")
add_executable(mltools ${MLSRCS})
target_include_directories(mltools PRIVATE ./)
target_link_libraries(mltools 
	optimizer
	utils 
	dataop
	model 
	nn 
	${Boost_LIBRARIES})