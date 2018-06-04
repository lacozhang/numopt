cmake_minimum_required(VERSION 3.0.0)

find_package(Protobuf REQUIRED)
file(GLOB ALL_PROTOS "*.proto")

set(PROTO_HDRS "")
set(PROTO_SRCS "")
foreach(PROTO_FILE ${ALL_PROTOS})
  get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
  protobuf_generate_cpp(${PROTO_NAME}_SRCS ${PROTO_NAME}_HDRS ${PROTO_FILE})
  list(APPEND PROTO_HDRS ${${PROTO_NAME}_HDRS})
  list(APPEND PROTO_SRCS ${${PROTO_NAME}_SRCS})
endforeach(PROTO_FILE)

add_library(proto STATIC ${PROTO_HDRS} ${PROTO_SRCS})

add_custom_command(TARGET proto PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "include/proto")
add_custom_command(TARGET proto PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "src")

add_custom_command(TARGET proto POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${PROTO_HDRS} "include/proto")

add_custom_command(TARGET proto POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${PROTO_SRCS} "src")

target_include_directories(proto PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include/proto>)

target_link_libraries(proto ${Protobuf_LIBRARIES})

