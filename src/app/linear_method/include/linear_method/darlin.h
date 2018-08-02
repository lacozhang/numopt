/*
 * =====================================================================================
 *
 *       Filename:  darlin.h
 *
 *    Description:  block coordinate descent algorithm
 *
 *        Version:  1.0
 *        Created:  07/29/2018 10:01:49
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once

#include "filter/sparse_filter.h"
#include "learner/bcd.h"
#include "proto/linear.pb.h"
#include "system/sysutil.h"
#include "util/bitmap.h"
#include "util/stringop.h"

namespace mltools {
DECLARE_int32(num_workers);
namespace linear {
typedef double Real;
class DarlinScheduler : public BCDScheduler {
public:
  DarlinScheduler(const Config &conf)
      : BCDScheduler(conf.darlin()), conf_(conf) {
    dataAssigner_.set(conf_.training_data(), FLAGS_num_workers,
                      bcdConf_.load_local_data());
  }

  virtual ~DarlinScheduler() {}

  virtual void run() override {
    CHECK_EQ(conf_.loss().type(), LossConfig::LOGIT);
    CHECK_EQ(conf_.penalty().type(), PenaltyConfig::L1);
    LOG(INFO) << "Train l1 logistic regression";

    // load data
    BCDScheduler::run();

    auto darlin = conf_.darlin();
    int tau = darlin.max_block_delay();
    LOG(WARNING) << "max delay " << tau;
    bool randomBlkOrder = darlin.random_feature_block_order();
    if (!randomBlkOrder) {
      LOG(WARNING) << "randomized will accelerate the BCD convergence";
    }

    kktFilterThreshold_ = 1e20;
    bool resetKKTFilter = false;
    int maxIter = darlin.max_pass_of_data();

    // timestamp of application message
    int time = exec_.time();
    int firstTime = time;

    // timestamp of model parameter <still a customer>
    int modelTime = featGroup_.size() * 6;
    for (int iter = 0; iter < maxIter; ++iter) {
      // select the block update order
      auto order = blkOrder_;
      if (randomBlkOrder) {
        std::random_shuffle(order.begin(), order.end());
      }
      if (iter == 0) {
        order.insert(order.begin(), priorBlkOrder_.begin(),
                     priorBlkOrder_.end());
      }

      // go over all the feature blocks <not feature groups, some feature group
      // will be breaked into multiple feature blocks>
      for (int i = 0; i < order.size(); ++i) {
        Task update;
        update.set_more(true);
        auto cmd = update.mutable_bcd();
        cmd->set_cmd(BCDCall::UPDATE_MODEL);
        // set bcd parameter time
        cmd->set_time(modelTime);
        modelTime += 3;

        // KKT filter
        if (iter == 0) {
          cmd->set_kkt_filter_threshold(kktFilterThreshold_);
          if (resetKKTFilter) {
            cmd->set_reset_kkt_filter(true);
          }
        }

        // block info
        auto blk = featBlk_[order[i]];
        blk.second.to(cmd->mutable_key());
        cmd->add_fea_grp(blk.first);

        // set command time
        update.set_time(time + 1);
        if (iter == 0 && i < priorBlkOrder_.size()) {
          addWaitTime(0, firstTime, &update);
          firstTime = time;
        } else {
          addWaitTime(tau, firstTime, &update);
        }

        time = submit(update, kCompGroup);
      }

      Task eval;
      eval.set_more(false);
      eval.mutable_bcd()->set_cmd(BCDCall::EVALUATE_PROGRESS);
      eval.mutable_bcd()->set_iter(iter);
      addWaitTime(tau, firstTime, &eval);
      time = submit(eval, kCompGroup);
      wait(time);
      showProgress(iter);

      int k = bcdConf_.save_model_every_n_iter();
      if (k > 0 && ((iter + 1) % k == 0) && conf_.has_model_output()) {
        time = saveModel(
            ithFile(conf_.model_output(), 0, "_it_" + std::to_string(iter)));
      }

      Real vio = globalProgress_[iter].violation();
      Real ratio = bcdConf_.GetExtension(kkt_filter_threshold_ratio);

      kktFilterThreshold_ = vio / (Real)globalTrainInfo_.num_ex() * ratio;

      Real rel = globalProgress_[iter].relative_obj();
      if (rel > 0 && rel <= darlin.epsilon()) {
        if (resetKKTFilter) {
          break;
        } else {
          resetKKTFilter = true;
        }
      } else {
        resetKKTFilter = false;
      }
    }

    for (int t = firstTime; t < time; ++t) {
      wait(t);
    }

    if (conf_.has_model_output()) {
      wait(saveModel(conf_.model_output()));
    }
  }

protected:
  std::string showKKTFilter(int iter) {
    char buf[500];
    if (iter == -3) {
      snprintf(buf, 500, "|      KKT filter     ");
    } else if (iter == -2) {
      snprintf(buf, 500, "| threshold  #activet ");
    } else if (iter == -1) {
      snprintf(buf, 500, "+---------------------");
    } else {
      snprintf(buf, 500, "| %.1e %11llu ", kktFilterThreshold_,
               (uint64)globalProgress_[iter].nnz_active_set());
    }
    return string(buf);
  }

  void showProgress(int iter) {
    int s = iter == 0 ? -3 : iter;
    for (int i = s; i <= iter; ++i) {
      string str = showObjective(i) + showKKTFilter(i) + showTime(i);
      NOTICE("%s", str.c_str());
    }
  }

  void addWaitTime(int tau, int first, Task *task) {
    int cur = task->time();
    for (int t = std::max(first, cur - 2 * tau - 1);
         t < std::max(first + 1, cur - tau); ++t) {
      task->add_wait_time(t);
    }
  }

  Real kktFilterThreshold_;
  Config conf_;
};

class DarlinCompNode {
public:
  DarlinCompNode() {}
  virtual ~DarlinCompNode() {}
  Real delta(Real deltaMax, Real deltaW) {
    return std::min(deltaMax, 2 * std::fabs(deltaW) + 0.1);
  }

protected:
  std::unordered_map<int, Bitmap> activeSet_;
  std::unordered_map<int, DArray<Real>> delta_;
  SparseFilter kktFilter_;
  Real kktFilterThreshold_;
};

class DarlinServer : public BCDServer<Real>, public DarlinCompNode {
public:
  DarlinServer(const Config &conf)
      : BCDServer<Real>(conf.darlin()), conf_(conf) {}
  virtual ~DarlinServer() {}

protected:
  virtual void preprocessData(int time, Message *request) override {
    BCDServer<Real>::preprocessData(time, request);
    for (int grp : featGroup_) {
      size_t featSize = model_[grp].key_.size();
      activeSet_[grp].resize(featSize, false);
      delta_[grp].resize(featSize, bcdConf_.GetExtension(delta_init_value));
    }
  }

  virtual void update(int time, Message *request) override {
    const BCDCall &call = request->task_.bcd();
    if (call.has_kkt_filter_threshold()) {
      kktFilterThreshold_ = call.kkt_filter_threshold();
      violation_ = 0;
    }
    if (call.reset_kkt_filter()) {
      for (int grp : featGroup_) {
        activeSet_[grp].fill(true);
      }
    }

    CHECK_EQ(call.fea_grp_size(), 1);
    int grp = call.fea_grp(0);
    Range<Key> globalKeyRange(call.key());
    if (MyKeyRange().setIntersection(globalKeyRange).empty()) {
      VLOG(1) << "requested key group not in node's range";
      return;
    }
    auto colRange = model_[grp].key_.findRange(globalKeyRange);

    VLOG(1) << "updating group " << grp << ", global key " << globalKeyRange
            << ", local index " << colRange;

    // aggregate work's local gradients
    model_.waitReceivedRequest(time, kWorkerGroup);

    // update weights
    if (!colRange.empty()) {
      auto data = model_.buffer(time);
      CHECK_EQ(data.channel_, grp);
      CHECK_EQ(data.idxRange_, colRange);
      CHECK_EQ(data.values_.size(), 2);
      updateWeight(grp, colRange, data.values_[0], data.values_[1]);
    }

    model_.finishReceivedRequest(time + 1, kWorkerGroup);
    VLOG(1) << "updating group " << grp;
  }

  virtual void evaluate(BCDProgress *prog) override {
    size_t nnzW = 0, nnzAS = 0;
    Real objv = 0;
    for (int grp : featGroup_) {
      const auto &val = model_[grp].val_;
      for (Real w : val) {
        if (kktFilter_.marked(w) || w == 0) {
          continue;
        }
        ++nnzW;
        objv += std::fabs(w);
      }
      nnzAS += activeSet_[grp].nnz();
    }

    prog->set_objective(objv * conf_.penalty().lambda(0));
    prog->set_nnz_w(nnzW);
    prog->set_violation(violation_);
    prog->set_nnz_active_set(nnzAS);
  }

  Config conf_;
  Real violation_ = 0;

private:
  void updateWeight(int groupId, Range<Key> range, DArray<Real> G,
                    DArray<Real> U) {
    CHECK_EQ(G.size(), range.size());
    CHECK_EQ(U.size(), range.size());

    Real eta = conf_.learning_rate().alpha();
    Real lambda = conf_.penalty().lambda(0);
    Real deltaMax = bcdConf_.GetExtension(delta_max_value);
    auto &value = model_[groupId].val_;
    auto &activeSet = activeSet_[groupId];
    auto &delta = delta_[groupId];

    for (size_t i = 0; i < range.size(); ++i) {
      size_t k = i + range.begin();
      if (!activeSet.test(k)) {
        continue;
      }
      Real g = G[i], u = U[i] / eta + 1e-10;
      Real gPos = g + lambda, gNeg = g - lambda;
      Real &w = value[k];
      Real d = -w, vio = 0;

      if (w == 0) {
        if (gPos < 0) {
          vio = -gPos;
        } else if (gNeg > 0) {
          vio = gNeg;
        } else if (gPos > kktFilterThreshold_ && gNeg < -kktFilterThreshold_) {
          activeSet.clear(k);
          kktFilter_.mark(&w);
          continue;
        }
      }

      violation_ = std::max(violation_, vio);

      if (gPos <= u * w) {
        d = -gPos / u;
      } else if (gNeg >= u * w) {
        d = -gNeg / u;
      }
      d = std::min(delta[k], std::max(-delta[k], d));
      delta[k] = DarlinCompNode::delta(deltaMax, d);
      w += d;
    }
  }
};

class DarlinWorker : public BCDWorker<Real>, public DarlinCompNode {
public:
  DarlinWorker(const Config &conf)
      : BCDWorker<Real>(conf.darlin()), conf_(conf) {}
  virtual ~DarlinWorker() {}

protected:
  virtual void preprocessData(int time, Message *request) override {
    BCDWorker<Real>::preprocessData(time, request);

    if (bcdConf_.init_w().type() == ParamInitConfig::ZERO) {
      dual_.setValue(1);
    } else {
      dual_.eigenArray() =
          exp(label_->value().eigenArray() * dual_.eigenArray());
    }

    for (int grp : featGroup_) {
      size_t n = model_[grp].key_.size();
      activeSet_[grp].resize(n, true);
      delta_[grp].resize(n, bcdConf_.GetExtension(delta_init_value));
    }
  }

  virtual void evaluate(BCDProgress *prog) override {
    busyTimer_.start();
    mu_.lock(); // lock the dual_
    prog->set_objective(log(1 + 1 / dual_.eigenArray()).sum());
    mu_.unlock();
    prog->add_busy_time(busyTimer_.stop());
    busyTimer_.restart();
  }

  virtual void update(int time, Message *msg) override {
    auto &call = msg->task_.bcd();
    if (call.reset_kkt_filter()) {
      for (int grp : featGroup_) {
        activeSet_[grp].fill(true);
      }
    }

    CHECK_EQ(call.fea_grp_size(), 1);
    int grp = call.fea_grp(0);
    Range<Key> globalKeyRange(call.key());
    auto colRange = model_[grp].key_.findRange(globalKeyRange);

    computeAndPushGradient(time, globalKeyRange, grp, colRange);
    pullAndUpdateDual(time + 2, globalKeyRange, grp, colRange, msg);
    msg->finished_ = false;
  }

  void computeAndPushGradient(int time, Range<Key> glbKr, int grp,
                              SizeR colRange) {
    VLOG(1) << "compute gradient group " << grp << ", global key " << glbKr
            << ", local index " << colRange;

    // first order gradient
    DArray<Real> G(colRange.size(), 0);

    // the upper bound of the diag hession
    DArray<Real> U(colRange.size(), 0);

    mu_.lock();
    busyTimer_.start();
    if (!colRange.empty()) {
      CHECK_GT(FLAGS_num_threads, 0);
      int numThreads = colRange.size() < 64 ? 1 : FLAGS_num_threads;
      int nparts = numThreads;
      ThreadPool pool(numThreads);
      for (int i = 0; i < nparts; ++i) {
        auto thrRange = colRange.evenDivide(nparts, i);
        if (thrRange.empty()) {
          continue;
        }

        // range of selected gradient relative to column range.
        auto gr = thrRange - colRange.begin();
        pool.add([this, grp, thrRange, gr, &G, &U]() {
          computeGradient(grp, thrRange, G.segment(gr), U.segment(gr));
        });
      }
      pool.startWorkers();
    }
    busyTimer_.stop();
    mu_.unlock();

    Task req = Parameter::request(grp, time, {}, bcdConf_.comm_filter(), glbKr);
    model_.push(req, model_[grp].key_.segment(colRange), {G, U});
  }

  void computeGradient(int grp, SizeR colRange, DArray<Real> G,
                       DArray<Real> U) {
    CHECK_EQ(G.size(), colRange.size());
    CHECK_EQ(U.size(), colRange.size());
    CHECK(data_[grp]->colMajor());

    const auto &activeSet = activeSet_[grp];
    const auto &delta = delta_[grp];
    const Real *y = label_->value().data();
    auto X = std::static_pointer_cast<SparseMatrix<uint32, Real>>(
        data_[grp]->colBlock(colRange));
    const size_t *offset = X->offset().data();
    uint32 *index = X->index().data() + offset[0];
    Real *value = X->value().data() + offset[0];
    bool binary = X->binary();

    for (size_t j = 0; j < X->cols(); ++j) {
      size_t k = j + colRange.begin();
      size_t n = offset[j + 1] - offset[j];
      if (!activeSet.test(k)) {
        index += n;
        if (!binary) {
          value += n;
        }
        kktFilter_.mark(&G[j]);
        kktFilter_.mark(&U[j]);
        continue;
      }

      Real g = 0, u = 0;
      Real d = binary ? exp(delta[k]) : delta[k];
      for (size_t o = 0; o < n; ++o) {
        auto i = *(index++);
        Real tau = 1 / (1 + dual_[i]);
        if (binary) {
          g -= y[i] * tau;
          u += std::min(tau * (1 - tau) * d, .25);
        } else {
          Real v = *(value++);
          g -= y[i] * tau * v;
          u += std::min(tau * (1 - tau) * exp(fabs(v) * d), .25) * v * v;
        }
      }

      G[j] = g;
      U[j] = u;
    }
  }

  void pullAndUpdateDual(int time, Range<Key> glbKr, int grp, SizeR colRange,
                         Message *msg) {
    Task req = Parameter::request(grp, time, {time - 1}, bcdConf_.comm_filter(),
                                  glbKr);
    Message origReq = *msg;
    model_.pull(req, model_[grp].key_.segment(colRange),
                [this, time, grp, colRange, origReq]() mutable {
                  VLOG(1) << "pulled the weight from group " << grp
                          << ", local index " << colRange;
                  if (!colRange.empty()) {
                    auto data = model_.buffer(time);
                    CHECK_EQ(data.channel_, grp);
                    CHECK_EQ(data.idxRange_, colRange);
                    CHECK_EQ(data.values_.size(), 1);
                    updateDual(grp, colRange, data.values_[0]);
                  }

                  finishReceivedRequest(origReq.task_.time(), origReq.sender_);
                  reply(&origReq);
                });
  }

  void updateDual(int grp, SizeR colRange, DArray<Real> newW) {
    auto &currW = model_[grp].val_;
    auto &activeSet = activeSet_[grp];
    auto &delta = delta_[grp];
    Real deltaMax = bcdConf_.GetExtension(delta_max_value);
    DArray<Real> deltaW(newW.size());
    for (int i = 0; i < newW.size(); ++i) {
      size_t j = colRange.begin() + i;
      Real &cw = currW[i];
      Real &nw = newW[i];
      if (kktFilter_.marked(nw)) {
        activeSet.clear(j);
        cw = 0;
        deltaW[i] = 0;
        continue;
      }

      deltaW[i] = nw - cw;
      delta[j] = DarlinCompNode::delta(deltaMax, deltaW[i]);
    }

    CHECK(data_[grp]);
    mu_.lock();
    busyTimer_.start();
    {
      SizeR rowRange(0, data_[grp]->rows());
      ThreadPool pool(FLAGS_num_threads);
      for (int i = 0; i < FLAGS_num_threads; ++i) {
        auto thrRange = rowRange.evenDivide(FLAGS_num_threads, i);
        if (thrRange.empty()) {
          continue;
        }
        pool.add([this, grp, thrRange, colRange, deltaW]() {
          updateDual(grp, thrRange, colRange, deltaW);
        });
      }
      pool.startWorkers();
    }
    busyTimer_.stop();
    mu_.unlock();
  }

  void updateDual(int grp, SizeR rowRange, SizeR colRange,
                  DArray<Real> deltaW) {
    CHECK_EQ(colRange.size(), deltaW.size());
    CHECK(data_[grp]->colMajor());

    const auto &activeSet = activeSet_[grp];
    Real *y = label_->value().data();
    auto X = std::static_pointer_cast<SparseMatrix<uint32, Real>>(
        data_[grp]->colBlock(colRange));
    size_t *offset = X->offset().data();
    uint32 *index = X->index().data() + offset[0];
    Real *value = X->value().data();
    bool binary = X->binary();

    for (size_t j = 0; j < X->cols(); ++j) {
      size_t k = j + colRange.begin();
      size_t n = offset[j + 1] - offset[j];
      Real wd = deltaW[j];
      if (wd == 0 || !activeSet.test(k)) {
        index += n;
        continue;
      }

      for (size_t o = offset[j]; o < offset[j + 1]; ++o) {
        auto i = *(index++);
        if (!rowRange.contains(i)) {
          continue;
        }
        dual_[i] *= binary ? exp(y[i] * wd) : exp(y[i] * wd * value[o]);
      }
    }
  }

private:
  Config conf_;
  Timer busyTimer_;
};
} // namespace linear
} // namespace mltools
