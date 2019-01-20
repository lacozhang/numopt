cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

add_executable(text_featurize "src/txtfeat.cpp")
target_link_libraries(text_featurize dataop utils ${GLOG_LIBRARIES})
