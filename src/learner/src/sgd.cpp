/*
 * =====================================================================================
 *
 *       Filename:  sgd.cpp
 *
 *    Description:  impl
 *
 *        Version:  1.0
 *        Created:  07/27/2018 13:48:28
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "learner/sgd.h"
namespace mltools {

ISGDScheduler::~ISGDScheduler() {}

void ISGDScheduler::run() {
  // init monitor
  using namespace std::placeholders;
  monitor_.setMerger(std::bind(&ISGDScheduler::mergeProgress, this, _1, _2));
  monitor_.setPrinter(1, std::bind(&ISGDScheduler::showProgress, this, _1, _2));

  // wait all job done.
  sys_.manager().addNodeFailureHandler(
      [this](const NodeID &id) { CHECK_NOTNULL(workpoll_)->restore(id); });
  CHECK_NOTNULL(workpoll_)->waitUntilDone();
  // save model
  Task task;
  task.mutable_sgd()->set_cmd(SGDCall::SAVE_MODEL);
  int ts = submit(task, kServerGroup);
  wait(ts);
}

void ISGDScheduler::processResponse(mltools::Message *response) {
  const auto &sgd = response->task_.sgd();
  if (sgd.cmd() == SGDCall::UPDATE_MODEL) {
    for (int i = 0; i < sgd.load().finished_size(); ++i) {
      workpoll_->finish(sgd.load().finished(i));
    }
    sendWorkload(response->sender_);
  }
}

void ISGDScheduler::processRequest(mltools::Message *request) {
  if (request->task_.sgd().cmd() == SGDCall::REQUEST_WORKLOAD) {
    sendWorkload(request->sender_);
  }
}

void ISGDScheduler::sendWorkload(const mltools::NodeID &recver) {
  Task task;
  task.mutable_sgd()->set_cmd(SGDCall::UPDATE_MODEL);
  if (workpoll_->assign(recver, task.mutable_sgd()->mutable_load())) {
    submit(task, recver);
  }
}

void ISGDScheduler::mergeProgress(const SGDProgress &src, SGDProgress *dst) {
  auto old = *dst;
  *dst = src;
  dst->set_num_examples_processed(dst->num_examples_processed() +
                                  old.num_examples_processed());
}

void ISGDScheduler::showProgress(
    double time, std::unordered_map<NodeID, SGDProgress> *progress) {
  uint64 numEx = 0, nnzW = 0;
  DArray<double> objv, auc, acc;
  double weightSum = 0, deltaSum = 0;
  for (const auto &it : *progress) {
    auto &prog = it.second;
    numEx += prog.num_examples_processed();
    nnzW += prog.nnz();
    for (int i = 0; i < prog.objective_size(); ++i) {
      objv.push_back(prog.objective(i));
    }
    for (int i = 0; i < prog.auc_size(); ++i) {
      auc.push_back(prog.auc(i));
    }
    for (int i = 0; i < prog.accuracy_size(); ++i) {
      acc.push_back(prog.accuracy(i));
    }
    weightSum += prog.weight_sum();
    deltaSum += prog.delta_sum();
  }
  progress->clear();
  numExpProcessed_ += numEx;
  if (showProgHead_) {
    NOTICE(" sec  examples    loss      auc   accuracy   |w|_0  updt ratio");
    showProgHead_ = false;
  }
  NOTICE("%4d  %.2e  %.3e  %.4f  %.4f  %.2e  %.2e", (int)time,
         (double)numExpProcessed_, objv.sum() / (double)numEx, auc.mean(),
         acc.mean(), (double)nnzW, sqrt(deltaSum) / sqrt(weightSum));
}
} // namespace mltools
