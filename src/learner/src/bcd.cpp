/*
 * =====================================================================================
 *
 *       Filename:  bcd.cpp
 *
 *    Description:  impl
 *
 *        Version:  1.0
 *        Created:  07/27/2018 13:50:05
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "learner/bcd.h"
#include "util/resource_usage.h"
#include "util/stringop.h"

namespace mltools {

void BCDScheduler::processRequest(mltools::Message *request) {
  CHECK(request->task_.has_bcd());
  auto &bcd = request->task_.bcd();
  if (bcd.cmd() == BCDCall::REQUEST_WORKLOAD) {
    Task req;
    req.mutable_bcd()->set_cmd(BCDCall::LOAD_DATA);
    CHECK(dataAssigner_.next(req.mutable_bcd()->mutable_data()));
    submit(req, request->sender_);
  }
}

void BCDScheduler::processResponse(mltools::Message *response) {
  const auto &task = response->task_;
  if (!task.has_bcd()) {
    return;
  }

  if (task.bcd().cmd() == BCDCall::LOAD_DATA) {
    LoadDataResponse info;
    CHECK(info.ParseFromString(task.msg()));
    globalTrainInfo_ = mergeExampleInfo(globalTrainInfo_, info.example_info());
    hitCache_ += info.hit_cache();
    ++loadData_;
  } else if (task.bcd().cmd() == BCDCall::EVALUATE_PROGRESS) {
    BCDProgress prog;
    CHECK(prog.ParseFromString(task.msg()));
    mergeProgress(task.bcd().iter(), prog);
  }
}

void BCDScheduler::loadData() {
  sys_.manager().waitWorkersReady();
  sys_.manager().waitServersReady();
  auto loadTime = tic();
  int n = sys_.manager().numWorkers();
  while (loadData_ < n) {
    usleep(500);
  }
  if (hitCache_ > 0) {
    CHECK_EQ(hitCache_, n) << "data is old, watch out";
    NOTICE("Hit local caches for the training data");
  }
  NOTICE("Loaded %llu examples in %g sec", globalTrainInfo_.num_ex(),
         toc(loadTime));
}

void BCDScheduler::preprocessData() {
  for (int i = 0; i < globalTrainInfo_.slot_size(); ++i) {
    auto info = globalTrainInfo_.slot(i);
    CHECK(info.has_id());
    if (info.id() == 0) {
      continue;
    }
    featGroup_.push_back(info.id());
  }

  auto prepTime = tic();
  Task req;
  auto bcd = req.mutable_bcd();
  bcd->set_cmd(BCDCall::PREPROCESS_DATA);
  bcd->set_time(0);
  for (auto grp : featGroup_) {
    bcd->add_fea_grp(grp);
  }
  bcd->set_hit_cache(hitCache_ > 0);
  wait(submit(req, kCompGroup));
  NOTICE("Preprocessing is finished in %lf sec", toc(prepTime));
  if (bcdConf_.has_tail_feature_freq()) {
    NOTICE("Features with frequency <= %d are filtered",
           bcdConf_.tail_feature_freq());
  }
}

void BCDScheduler::divideFeatureBlocks() {

  // iterate over all the feature groups.
  for (int i = 0; i < globalTrainInfo_.slot_size(); ++i) {
    auto info = globalTrainInfo_.slot(i);
    if (!info.has_id()) {
      continue;
    }
    if (info.id() == 0) {
      continue;
    }
    CHECK(info.has_nnz_ele());
    CHECK(info.has_nnz_ex());
    double nnzPerRow = (double)info.nnz_ex() / (double)info.nnz_ex();
    int n = 1;
    if (nnzPerRow > 1 + 1e-6) {
      // if some feature group are relatively dense
      n = std::max((int)std::ceil(nnzPerRow * bcdConf_.feature_block_ratio()),
                   1);
    }
    for (int j = 0; j < n; ++j) {
      auto block = Range<Key>(info.min_key(), info.max_key()).evenDivide(n, j);
      if (block.empty()) {
        continue;
      }
      featBlk_.push_back(std::pair<int, Range<Key>>(info.id(), block));
    }
  }

  NOTICE("Features are partitioned into %ld blocks", featBlk_.size());

  for (int i = 0; i < featBlk_.size(); ++i) {
    blkOrder_.push_back(i);
  }

  std::vector<std::string> hitBlk;
  for (int i = 0; i < bcdConf_.prior_fea_group_size(); ++i) {
    int grpId = bcdConf_.prior_fea_group(i);
    std::vector<int> tmp;
    for (int k = 0; k < featBlk_.size(); ++k) {
      if (featBlk_[k].first == grpId) {
        tmp.push_back(k);
      }
    }
    if (tmp.empty()) {
      continue;
    }
    hitBlk.push_back(std::to_string(grpId));

    for (int j = 0; j < bcdConf_.num_iter_for_prior_fea_group(); ++j) {
      if (bcdConf_.random_feature_block_order()) {
        std::random_shuffle(tmp.begin(), tmp.end());
      }
      priorBlkOrder_.insert(priorBlkOrder_.end(), tmp.begin(), tmp.end());
    }
  }

  if (!hitBlk.empty()) {
    NOTICE("Prior feature groups: %s", Util::join(hitBlk, ", ").c_str());
  }
  totalTimer_.restart();
}

void BCDScheduler::mergeProgress(int iter, const mltools::BCDProgress &recv) {
  auto &p = globalProgress_[iter];
  p.set_objective(p.objective() + recv.objective());
  p.set_nnz_w(p.nnz_w() + recv.nnz_w());
  if (recv.busy_time_size() > 0) {
    p.add_busy_time(recv.busy_time(0));
  }
  p.set_total_time(totalTimer_.stop());
  totalTimer_.start();
  p.set_relative_obj(
      iter == 0 ? 1
                : globalProgress_[iter - 1].objective() / p.objective() - 1);
  p.set_violation(std::max(p.violation(), recv.violation()));
  p.set_nnz_active_set(p.nnz_active_set() + recv.nnz_active_set());
}

int BCDScheduler::saveModel(const mltools::DataConfig &data) {
  Task task;
  task.mutable_bcd()->set_cmd(BCDCall::SAVE_MODEL);
  *task.mutable_bcd()->mutable_data() = data;
  return submit(task, kCompGroup);
}

std::string BCDScheduler::showTime(int iter) {
  char buf[512] = {0};
  if (iter == -3) {
    snprintf(buf, 512, "|    time (sec.)");
  } else if (iter == -2) {
    snprintf(buf, 512, "|(app:min max) total");
  } else if (iter == -1) {
    snprintf(buf, 512, "+-----------------");
  } else {
    auto &prog = globalProgress_[iter];
    double ttlT = prog.total_time() -
                  (iter > 0 ? globalProgress_[iter - 1].total_time() : 0);
    int n = prog.busy_time_size();
    Eigen::ArrayXd busyT(n);
    for (int i = 0; i < n; ++i) {
      busyT[i] = prog.busy_time(i);
    }
    snprintf(buf, 512, "|%6.1f%6.1f%6.1f", busyT.minCoeff(), busyT.maxCoeff(),
             ttlT);
  }

  return std::string(buf);
}

std::string BCDScheduler::showObjective(int iter) {
  char buf[512] = {0};
  if (iter == -3) {
    snprintf(buf, 512, "     |        training        |  sparsity ");
  } else if (iter == -2) {
    snprintf(buf, 512, "iter |  objective    relative |     |w|_0 ");
  } else if (iter == -1) {
    snprintf(buf, 512, " ----+------------------------+-----------");
  } else {
    auto prog = globalProgress_[iter];
    snprintf(buf, 512, "%4d | %.5e  %.3e |%10lu ", iter, prog.objective(),
             prog.relative_obj(), (size_t)prog.nnz_w());
  }
  return std::string(buf);
}
} // namespace mltools
