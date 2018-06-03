cmake_minimum_required(VERSION 2.8.10)

set(OPTSRC "src/cg.cpp" "src/lbfgs.cpp" 
	"src/linesearch.cpp" "src/opt.cpp" 
	"src/optimfactory.cpp" "src/proximalgd.cpp"
	"src/sdca.cpp" "src/sgd.cpp"
	"src/svrg.cpp")
add_library(optimizer ${OPTSRC})
target_include_directories(optimizer
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE src)
target_link_libraries(optimizer utils model)
