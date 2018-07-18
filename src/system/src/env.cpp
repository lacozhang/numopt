/*
 * =====================================================================================
 *
 *       Filename:  env.cpp
 *
 *    Description:  implementation of env.h
 *
 *        Version:  1.0
 *        Created:  07/15/2018 20:23:48
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "system/env.h"
#include "data/common.h"
#include "proto/node.pb.h"
#include "system/van.h"
#include "util/common.h"
#include "util/file.h"

namespace mltools {
DEFINE_int32(num_servers, 0, "number of servers");
DEFINE_int32(num_workers, 0, "number of workers");
DEFINE_int32(num_threads, 2, "number of computational threads");
DEFINE_int32(num_replicas, 0, "number of replicas per server node");

DEFINE_string(my_node, "role:SCHEDULER,hostname:'127.0.0.1',port:8000,id:'H'",
              "my node");
DEFINE_string(scheduler, "role:SCHEDULER,hostname:'127.0.0.1',port:8000,id:'H'",
              "scheduler node definition");
DEFINE_string(interface, "", "network interface");

void Env::init(char *argv0) { initGlog(argv0); }

void Env::initGlog(char *argv0) {
  if (FLAGS_log_dir.empty()) {
    FLAGS_log_dir = "/tmp";
  }
  if (!dirExists(FLAGS_log_dir)) {
    dirCreate(FLAGS_log_dir);
  }
  string logFilePrefix = FLAGS_log_dir + "/" + std::string(basename(argv0)) +
                         "." + Van::parseNode(FLAGS_my_node).id() + ".log.";
  google::SetLogDestination(google::INFO, (logFilePrefix + "INFO.").c_str());
  google::SetLogDestination(google::WARNING,
                            (logFilePrefix + "WARNING.").c_str());
  google::SetLogDestination(google::ERROR, (logFilePrefix + "ERROR.").c_str());
  google::SetLogDestination(google::FATAL, (logFilePrefix + "FATAL.").c_str());
  google::SetLogSymlink(google::INFO, "");
  google::SetLogSymlink(google::WARNING, "");
  google::SetLogSymlink(google::ERROR, "");
  google::SetLogSymlink(google::FATAL, "");
  FLAGS_logbuflevel = -1;
}
} // namespace mltools
