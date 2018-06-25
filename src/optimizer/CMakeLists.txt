cmake_minimum_required(VERSION 2.8.10)

set(OPT_SRCS "src/cg.cpp"
    "src/lbfgs.cpp"
	"src/linesearch.cpp"
    "src/opt.cpp"
	"src/optimfactory.cpp"
    "src/proximalgd.cpp"
	"src/sdca.cpp"
    "src/sgd.cpp"
	"src/svrg.cpp")

set(OPT_HDRS "include/optimizer/cg.h"
    "include/optimizer/lbfgs.h"
    "include/optimizer/linesearch.h"
    "include/optimizer/opt.h"
    "include/optimizer/optimfactory.h"
    "include/optimizer/proximalgd.h"
    "include/optimizer/sdca.h"
    "include/optimizer/sgd.h"
    "include/optimizer/svrg.h")

add_library(optimizer ${OPT_HDRS} ${OPT_SRCS})
target_include_directories(optimizer
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE src)

target_link_libraries(optimizer utils model)
