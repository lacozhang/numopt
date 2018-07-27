/*
 * =====================================================================================
 *
 *       Filename:  workload_pool.cpp
 *
 *    Description:  implementation of workload
 *
 *        Version:  1.0
 *        Created:  07/27/2018 13:23:37
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "learner/workload_pool.h"
#include "data/common.h"

namespace mltools {

void WorkloadPool::set(const Workload &load) {
  VLOG(1) << "initial training data " << load.ShortDebugString();
  Lock l(mu_);
  CHECK_GT(load.replica(), 0);
  DataConfig files = searchFiles(load.data());
  VLOG(1) << "files found " << files.ShortDebugString() << " of size "
          << files.file_size();

  alldatas_.resize(files.file_size() * load.replica());
  int k = 0;
  for (int r = 0; r < load.replica(); ++r) {
    if (load.shuffle()) {
      files = shuffleFiles(files);
    }
    for (int i = 0; i < files.file_size(); ++i) {
      *alldatas_.at(k).load_.mutable_data() = ithFile(files, i);
      alldatas_[k].load_.set_id(k);
      ++k;
    }
  }
  CHECK_EQ(k, alldatas_.size());
}

bool WorkloadPool::assign(const NodeID &node, Workload *load) {
  Lock l(mu_);
  for (auto &info : alldatas_) {
    if (!info.assigned_) {
      load->CopyFrom(info.load_);
      info.node_ = node;
      info.assigned_ = true;
      VLOG(1) << "assign " << load->ShortDebugString() << " to " << node;
      return true;
    }
  }

  return false;
}

void WorkloadPool::restore(const mltools::NodeID &node) {
  Lock l(mu_);
  for (auto &info : alldatas_) {
    if (info.assigned_ && !info.finished_ && info.node_ == node) {
      info.assigned_ = false;
      LOG(INFO) << "restore workload id " << info.load_.id() << " from node "
                << node;
    }
  }
}

void WorkloadPool::finish(int id) {
  Lock l(mu_);
  CHECK_GE(id, 0);
  CHECK_LT(id, alldatas_.size());
  alldatas_[id].finished_ = true;
  ++numFinished_;
  workDone_.notify_all();
  VLOG(1) << "workload " << id << " is finished";
}

void WorkloadPool::waitUntilDone() {
  std::unique_lock<std::mutex> lock(mu_);
  size_t alltasks = alldatas_.size();
  workDone_.wait(lock, [this, alltasks]() { return numFinished_ >= alltasks; });
}
} // namespace mltools
