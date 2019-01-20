cmake_minimum_required(VERSION 2.8.10)

add_library(system STATIC
  "include/system/message.h"
  "include/system/van.h"
  "include/system/postoffice.h"
  "include/system/manager.h"
  "include/system/assigner.h"
  "include/system/env.h"
  "include/system/heartbeat_info.h"
  "include/system/customer.h"
  "include/system/remote_node.h"
  "include/system/executor.h"
  "include/system/monitor.h"
  "include/system/sysutil.h"
  "src/message.cpp"
  "src/van.cpp"
  "src/postoffice.cpp"
  "src/manager.cpp"
  "src/assigner.cpp"
  "src/env.cpp"
  "src/heartbeat_info.cpp"
  "src/remote_node.cpp"
  "src/executor.cpp")

target_include_directories(system PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

target_link_libraries(system utils ${GLOG_LIBRARIES} data filter libzmq
${Snappy_LIBRARIES} ${GFLAGS_LIBRARIES})

set(TestFiles
  "test/run_van_util.h"
  "test/run_van_util.cpp"
  "test/test_message.cpp"
  "test/test_van.cpp")

add_executable(system_test ${TestFiles})
target_link_libraries(system_test system test_main)
add_test(NAME system_test
    COMMAND system_test)

set(TestProgram
  "test/run_van.cpp"
  "test/run_van_util.h"
  "test/run_van_util.cpp")
add_executable(run_van ${TestProgram})
target_link_libraries(run_van system)
add_test(NAME run_van
    COMMAND run_van)

set(VanWorker
  "test/run_van_worker.cpp"
  "test/run_van_util.h"
  "test/run_van_util.cpp")
add_executable(run_van_worker ${VanWorker})
target_link_libraries(run_van_worker system)
add_test(NAME run_van_worker
    COMMAND run_van_worker)

set(HelloPS
  "test/hello_ps.cpp")
add_executable(hello_ps ${HelloPS})
target_link_libraries(hello_ps system)
