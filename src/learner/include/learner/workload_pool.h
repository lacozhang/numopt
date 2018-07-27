/*
 * =====================================================================================
 *
 *       Filename:  workload_pool.h
 *
 *    Description:  work distributed to different workers
 *
 *        Version:  1.0
 *        Created:  07/27/2018 13:23:19
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once
#include "proto/workload.pb.h"
#include "system/message.h"
#include "util/common.h"

namespace mltools {

/// @brief workload abstraction
class WorkloadPool {
public:
  WorkloadPool() {}
  WorkloadPool(const Workload &load) { set(load); }

  void set(const Workload &load);
  bool assign(const NodeID &node, Workload *load);
  void restore(const NodeID &node);
  void finish(int id);
  void waitUntilDone();

protected:
  struct WorkloadInfo {
    NodeID node_;
    Workload load_;
    bool assigned_ = false;
    bool finished_ = false;
  };
  std::vector<WorkloadInfo> alldatas_;
  int numFinished_ = 0;
  std::mutex mu_;
  std::condition_variable workDone_;
};
} // namespace mltools
