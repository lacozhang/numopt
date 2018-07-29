/*
 * =====================================================================================
 *
 *       Filename:  async_sgd.h
 *
 *    Description:  main interface for asynchronous SGD
 *
 *        Version:  1.0
 *        Created:  07/29/2018 10:01:22
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "learner/sgd.h"
#include "linear_method/learning_rate.h"
#include "linear_method/loss.h"
#include "linear_method/penalty.h"
#include "parameter/kv_map.h"
#include "parameter/kv_vector.h"
#include "proto/linear.pb.h"
#include "system/sysutil.h"
#include "util/evaluation.h"
#include <random>

namespace mltools {
namespace linear {

/// @brief the scheduler node
class ASyncSGDScheduler : public ISGDScheduler {
public:
  ASyncSGDScheduler(const Config &conf) : ISGDScheduler(), conf_(conf) {
    Workload load;
    *load.mutable_data() = conf_.training_data();
    load.mutable_data()->set_ignore_feature_group(true);
    load.set_replica(conf_.async_sgd().num_data_pass());
    load.set_shuffle(true);
    workpoll_ = new WorkloadPool(load);
  }

private:
  Config conf_;
};

template <typename V> class ASyncSGDServer : public ISGDComputeNode {
public:
  ASyncSGDServer(const Config &conf) : ISGDComputeNode(), conf_(conf) {
    SGDState state(conf_.penalty(), conf_.learning_rate());
    state.reporter_ = &reporter_;
    if (conf_.async_sgd().algo() == SGDConfig::FTRL) {
      auto model = new KVMap<Key, V, FTRLEntry, SGDState>();
      model->setState(state);
      model_ = model;
    } else {
      if (conf_.async_sgd().ada_grad()) {
        model_ = new KVMap<Key, V, AdaGradEntry, SGDState>();
      } else {
        CHECK(false);
      }
    }
  }

  virtual ~ASyncSGDServer() { delete model_; }

  void saveModel() {
    auto output = conf_.model_output();
    if (output.format() == DataConfig::TEXT) {
      CHECK(output.file_size());
      std::string file = output.file(0) + "_" + MyNodeID();
      CHECK_NOTNULL(model_)->writeToFile(file);
      LOG(INFO) << MyNodeID() << " written the model to " << file;
    }
  }

  virtual void processRequest(Message *request) {
    if (request->task_.sgd().cmd() == SGDCall::SAVE_MODEL) {
      saveModel();
    }
  }

protected:
  Parameter *model_ = nullptr;
  Config conf_;

  struct SGDState {
    SGDState() {}
    SGDState(const PenaltyConfig &regConf, const LearningRateConfig &lrConf) {
      lr_ = std::shared_ptr<LearningRate<V>>(new LearningRate<V>(lrConf));
      reg_ = std::shared_ptr<Penalty<V>>(new Penalty<V>(regConf));
    }
    virtual ~SGDState() {}

    void update() {
      if (!reporter_) {
        return;
      }
      SGDProgress prog;
      prog.set_nnz(nnz_);
      prog.set_weight_sum(weightSum_);
      weightSum_ = 0;
      prog.set_delta_sum(deltaSum_);
      deltaSum_ = 0;
      reporter_->report(prog);
    }

    void updateWeight(V newWeight, V oldWeight) {
      if (newWeight == 0 && oldWeight != 0) {
        --nnz_;
      } else if (newWeight != 0 && oldWeight == 0) {
        ++nnz_;
      }

      weightSum_ += newWeight * newWeight;
      V delta = newWeight - oldWeight;
      deltaSum_ += delta * delta;
    }

    std::shared_ptr<Penalty<V>> reg_;
    std::shared_ptr<LearningRate<V>> lr_;

    int iter_ = 0;
    size_t nnz_ = 0;
    V weightSum_ = 0;
    V deltaSum_ = 0;
    V maxDelta_ = 1.0;
    MonitorSlaver<SGDProgress> *reporter_ = nullptr;
  };

  struct FTRLEntry {
    V w_ = 0; // projected model parameter
    V z_ = 0; // value before projection
    V sqrtN_ = 0;

    void set(const V *data, void *state) {
      SGDState *st = reinterpret_cast<SGDState *>(state);

      // update model parameters
      V wOld = w_;
      V grad = *data;
      V sqrtNNew = sqrt(sqrtN_ * sqrtN_ + grad * grad);
      V sigma = (sqrtNNew - sqrtN_) / st->lr_->alpha();
      z_ += grad - sigma * w_;
      sqrtN_ = sqrtNNew;
      V eta = st->lr->eval(sqrtN_);
      w_ = st->reg_->proximal(-z_ * eta, eta);

      // update state.
      st->updateWeight(w_, wOld);
    }

    void get(V *data, void *state) { *data = w_; }
  };

  struct AdaGradEntry {
    void set(const V *data, void *state) {
      SGDState *st = reinterpret_cast<SGDState *>(state);

      V grad = *data;
      sumSqGrad_ += grad * grad;
      V eta = st->lr_->eval(sqrt(sumSqGrad_));
      V wOld = w_;
      w_ = st->reg_->proximal(w_ - eta * grad, eta);

      st->updateWeight(w_, wOld);
    }

    V w_ = 0;
    V sumSqGrad_ = 0;
  };
};

template <typename V> class ASyncSGDWorker : public ISGDComputeNode {
public:
  ASyncSGDWorker(const Config &conf) : ISGDComputeNode(), conf_(conf) {
    loss_ = createLoss<V>(conf_.loss());
  }

  ~ASyncSGDWorker() {}

  virtual void processRequest(Message *request) override {
    const auto &sgd = request->task_.sgd();
    if (sgd.cmd() == SGDCall::UPDATE_MODEL) {
      updateModel(sgd.load());
      Task done;
      done.mutable_sgd()->set_cmd(SGDCall::UPDATE_MODEL);
      done.mutable_sgd()->mutable_load()->add_finished(sgd.load().id());
      reply(request, done);
    }
  }

  virtual void run() override {
    Task task;
    task.mutable_sgd()->set_cmd(SGDCall::REQUEST_WORKLOAD);
    submit(task, SchedulerID());
  }

private:
  void updateModel(const Workload &load) {
    LOG(INFO) << MyNodeID() << ": get work " << load.id();
    VLOG(1) << "workload data : " << load.data().ShortDebugString();
    const auto &sgd = conf_.async_sgd();
    MinibatchReader<V> reader;
    reader.initReader(load.data(), sgd.mini_batch(), sgd.data_buf());
    reader.initFilter(sgd.countmin_n(), sgd.countmin_k(),
                      sgd.tail_feature_freq());
    reader.start();

    processedBatch_ = 0;
    int id = 0;
    DArray<Key> key;
    for (;; ++id) {
      mu_.lock();
      auto &data = data_[id];
      mu_.unlock();
      if (!reader.read(data.first, data.second, key)) {
        break;
      }
      VLOG(1) << "load minibatch " << id << ", X: " << data.second->rows()
              << "-by-" << data.second->cols();

      auto req = Parameter::request(id, -1, {}, sgd.pull_filter());
      model_[id].key = key;
      model_.pull(req, key, [this, id]() { computeGradient(id); });
    }

    while (processedBatch_ < id) {
      usleep(500);
    }
    LOG(INFO) << MyNodeID() << ": finished workload " << load.id();
  }

  void computeGradient(int id) {
    mu_.lock();
    auto Y = data_[id].first;
    auto X = data_[id].second;
    data_.erase(id);
    mu_.unlock();
    CHECK_EQ(X->rows(), Y->rows());
    VLOG(1) << "compute gradient ";

    DArray<V> Xw(Y->rows());
    auto w = model_[id].value;
    Xw.eigenArray() = *X * w.eigenArray();
    SGDProgress prog;
    prog.add_objective(loss_->evaluate({Y, Xw.SMatrix()}));
    prog.add_auc(Evaluation<V>::auc(Y->value(), Xw));
    prog.add_accuracy(Evaluation<V>::accuracy(Y->value(), Xw));
    prog.set_num_examples_processed(prog.num_examples_processed() + Xw.size());
    this->reporter_.report(prog);
    DArray<V> grad(X->cols());
    loss_->compute({Y, X, Xw.SMatrix()}, {grad.SMatrix()});
    auto req = Parameter::request(id, -1, {}, conf_.async_sgd().push_filter());
    model_.push(req, model_[id].key, {grad}, [this]() { ++processedBatch_; });
    model_.clear(id);
  }

  KVVector<Key, V> model_;
  LossPtr<V> loss_;
  std::unordered_map<int, std::pair<MatrixPtr<V>, MatrixPtr<V>>> data_;
  std::mutex mu_;
  std::atomic_int processedBatch_;
  int workLoadId_ = -1;
  Config conf_;
};
} // namespace linear
} // namespace mltools
