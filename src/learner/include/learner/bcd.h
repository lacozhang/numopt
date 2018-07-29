/*
 * =====================================================================================
 *
 *       Filename:  bcd.h
 *
 *    Description:  coordinate descent method
 *
 *        Version:  1.0
 *        Created:  07/27/2018 13:48:47
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "data/common.h"
#include "data/slot_reader.h"
#include "parameter/kv_vector.h"
#include "proto/bcd.pb.h"
#include "system/assigner.h"
#include "util/localizer.h"

namespace mltools {

/// @brief The scheduler
class BCDScheduler : public App {
public:
  BCDScheduler(const BCDConfig &conf) : App(), bcdConf_(conf) {}
  virtual ~BCDScheduler() {}

  virtual void processRequest(Message *request) override;
  virtual void processResponse(Message *response) override;
  virtual void run() override {
    loadData();
    preprocessData();
    divideFeatureBlocks();
  }

protected:
  /// @brief issues model saving tasks to workers.
  int saveModel(const DataConfig &data);

  /// @brief returns the time string
  std::string showTime(int iter);

  /// @brief return the objective string
  std::string showObjective(int iter);

  std::map<int, BCDProgress> globalProgress_;

  /// @brief <feature group id, key range>
  typedef std::vector<std::pair<int, Range<Key>>> FeatureBlocks;
  FeatureBlocks featBlk_;
  std::vector<int> blkOrder_;
  std::vector<int> priorBlkOrder_;
  ExampleInfo globalTrainInfo_;
  BCDConfig bcdConf_;
  std::vector<int> featGroup_;
  DataAssigner dataAssigner_;

private:
  /// @brief waits until all workers finished data loading.
  void loadData();

  /// @brief issues data preprocessing tasks to workers.
  void preprocessData();

  /// @brief divide feature into blocks.
  void divideFeatureBlocks();

  /// @brief merge the progress of all nodes at iteration iter
  void mergeProgress(int iter, const BCDProgress &recv);

  int hitCache_ = 0;
  int loadData_ = 0;
  Timer totalTimer_;
};

template <typename V> class BCDCompNode : public App {
public:
  BCDCompNode(const BCDConfig &conf) : bcdConf_(conf), model_(true) {}
  virtual ~BCDCompNode() {}
  virtual void processRequest(Message *request) override {
    const auto &task = request->task_;
    CHECK(task.has_bcd());
    int time = task.bcd().time();
    switch (task.bcd().cmd()) {
    case BCDCall::PREPROCESS_DATA:
      preprocessData(time, request);
      break;
    case BCDCall::UPDATE_MODEL:
      update(time, request);
      break;
    case BCDCall::EVALUATE_PROGRESS: {
      BCDProgress prog;
      evaluate(&prog);
      std::string str;
      CHECK(prog.SerializeToString(&str));
      Task res;
      res.set_msg(str);
      res.mutable_bcd()->set_iter(task.bcd().iter());
      res.mutable_bcd()->set_cmd(BCDCall::EVALUATE_PROGRESS);
      reply(request, res);
    } break;
    default:
      break;
    }
  }

protected:
  /// @brief update the model as requested.
  virtual void update(int time, Message *msg) = 0;

  /// @brief evaluate the current model
  virtual void evaluate(BCDProgress *prog) = 0;

  /**
   * @brief preprocess the training data
   *
   * transpose the data into column-major format, also build the key mapping
   * from global key to local key.
   */
  virtual void preprocessData(int time, Message *msg) = 0;

private:
  const int timeRatio_ = 3;
  // feature group
  std::vector<int> featGroup_;
  BCDConfig bcdConf_;
  KVVector<Key, V> model_;
};

#define USING_BCD_COMP_NODE                                                    \
  using BCDCompNode<V>::featGroup_;                                            \
  using BCDCompNode<V>::model_;                                                \
  using BCDCompNode<V>::bcdConf_;

/// @brief A server mode.
template <typename V> class BCDServer : public BCDCompNode<V> {
public:
  BCDServer(const BCDConfig &conf) : BCDCompNode<V>(conf) {}
  virtual ~BCDServer() {}

  virtual void processRequest(Message *request) override {
    BCDCompNode<V>::processRequest(request);
    auto bcd = request->task_.bcd();
    if (bcd.cmd() == BCDCall::SAVE_MODEL) {
      CHECK(bcd.has_data());
      saveModel(bcd.data());
    }
  }

protected:
  virtual void preprocessData(int time, Message *request) override {
    auto &call = request->task_.bcd();
    int grpSize = call.fea_grp_size();
    featGroup_.clear();
    for (int i = 0; i < grpSize; ++i) {
      featGroup_.push_back(call.fea_grp(i));
    }
    bool hitCache = call.hit_cache();
    for (int i = 0; i < grpSize; ++i, time += 3) {
      if (hitCache) {
        continue;
      }
      model_.waitReceivedRequest(time, kWorkerGroup);
      model_.finishReceivedRequest(time + 1, kWorkerGroup);
    }
    for (int i = 0; i < grpSize; ++i, time += 3) {
      model_.waitReceivedRequest(time, kWorkerGroup);
      auto &grp = model_[featGroup_[i]];
      grp.value.resize(grp.key_.size());
      grp.value.setValue(bcdConf_.init_w());
      model_.finishReceivedRequest(time + 1, kWorkerGroup);
    }
    model_.clearFilter();
  }

  void saveModel(const DataConfig &conf) {
    if (conf.format() == DataConfig::TEXT) {
      CHECK(conf.file_size());
      std::string filepath = conf.file(0) + "_" + MyNodeID();
      if (!dirExists(getPath(filepath))) {
        dirCreate(getPath(filepath));
      }
      std::ofstream sink(filepath);
      CHECK(sink.good());
      for (int grp : featGroup_) {
        auto &key = model_[grp].key_;
        auto &val = model_[grp].val_;
        CHECK_EQ(key.size(), val.size());

        for (size_t i = 0; i < key.size(); ++i) {
          if (val[i] == 0.0) {
            continue;
          }
          sink << key[i] << "\t" << val[i];
        }
      }
      LOG(INFO) << "write model parameter to file " << filepath;
    }
  }
  USING_BCD_COMP_NODE;
};

template <typename V> class BCDWorker : public BCDCompNode<V> {
public:
  BCDWorker(const BCDConfig &conf) : BCDCompNode<V>(conf) {
    if (!bcdConf_.has_local_cache()) {
      bcdConf_.mutable_local_cache()->add_file("/tmp/bcd_");
    }
  }

  virtual ~BCDWorker() {}

  virtual void run() override {
    Task task;
    task.mutable_bcd()->set_cmd(BCDCall::REQUEST_WORKLOAD);
    this->submit(task, SchedulerID());
  }

  virtual void processRequest(Message *msg) override {
    BCDCompNode<V>::processRequest(msg);
    auto &bcd = msg->task_.bcd();
    if (bcd.cmd() == BCDCall::LOAD_DATA) {
      CHECK(bcd.has_data());
      LoadDataResponse ret;
      int hitCache = 0;
      loadData(bcd.data(), ret.mutable_example_info(), &hitCache);
      ret.set_hit_cache(hitCache);
      std::string str;
      CHECK(ret.SerializeToString(&str));
      Task res;
      res.set_msg(str);
      res.mutable_bcd()->set_cmd(BCDCall::LOAD_DATA);
      this->reply(msg, res);
    }
  }

protected:
  void loadData(const DataConfig &conf, ExampleInfo *info, int *hitCache) {
    *hitCache = dataCache("train", true);
    if (!*hitCache) {
      slotReader_.init(searchFiles(conf), bcdConf_.local_cache());
      slotReader_.read(info);
    }
  }

  virtual void preprocessData(int time, Message *msg) override {
    const auto &call = msg->task_.bcd();
    int grpSize = call.fea_grp_size();
    featGroup_.clear();
    for (int i = 0; i < grpSize; ++i) {
      featGroup_.push_back(call.fea_grp(i));
    }
    bool hitCache = call.hit_cache();
    int maxParallel =
        std::max(1, bcdConf_.max_num_parallel_groups_in_preprocessing());

    std::vector<int> pullTime(grpSize);
    for (int i = 0; i < grpSize; ++i, time += 3) {
      if (hitCache) {
        continue;
      }
      pullTime[i] = filterTailFeatures(time, i);
      if (i >= maxParallel) {
        model_.wait(pullTime[i - maxParallel]);
      }
    }

    std::vector<std::promise<void>> waitDual(grpSize);
    for (int i = 0; i < grpSize; ++i, time += 3) {
      if (!hitCache && i >= (grpSize - maxParallel)) {
        model_.wait(pullTime[i]);
      }
      initModel(time, featGroup_[i], waitDual[i]);
    }

    if (!hitCache) {
      label_ = MatrixPtr<double>(new DenseMatrix<double>(
          slotReader_.info<double>(0), slotReader_.value<double>(0)));
    }

    for (int i = 0; i < grpSize; ++i) {
      waitDual[i].get_future().wait();
    }

    dataCache("train", false);
  }

private:
  int filterTailFeatures(int time, int i) {
    int grpSize = featGroup_.size();
    int grp = featGroup_[i];

    // find all unique feature ids within group i
    DArray<Key> uniqKey;
    DArray<uint8> keyCnt;

    Localizer<Key, double> *localizer = new Localizer<Key, double>();
    VLOG(1) << "count features ids in [" << i << "/" << grpSize << "]";
    localizer->countUniqIndex(slotReader_.index(grp), &uniqKey, &keyCnt);
    VLOG(1) << "finish count " << i << "/" << grpSize;

    Task push = Parameter::request(grp, time, {}, bcdConf_.comm_filter());
    auto tail = push.mutable_param()->mutable_tail_filter();
    tail->set_insert_count(true);
    tail->set_countmin_k(bcdConf_.countmin_k());
    tail->set_countmin_n((int)uniqKey.size() * bcdConf_.countmin_n_ratio());
    Message pushMsg(push, kServerGroup);
    pushMsg.set_key(uniqKey);
    pushMsg.add_value(keyCnt);
    model_.push(&pushMsg);

    // pull filtered keys after the server aggregates all the keys
    Task pull =
        Parameter::request(grp, time + 2, {time + 1}, bcdConf_.comm_filter());
    tail = pull.mutable_param()->mutable_tail_filter();
    tail->set_freq_threshold(bcdConf_.tail_feature_freq());
    Message pullMsg(pull, kServerGroup);
    pullMsg.set_key(uniqKey);
    pullMsg.callback = [this, grp, localizer, i, grpSize]() mutable {
      // localize the training data.
      VLOG(1) << "remap index [" << i << "/" << grpSize << "]";
      DArray<V> val;
      auto info = slotReader_.info<V>(grp);
      if (info.type() == MatrixInfo::SPARSE) {
        val = slotReader_.value<V>(grp);
      }
      auto x =
          localizer->remapIndex(info, slotReader_.offset(grp),
                                slotReader_.index(grp), val, model_[grp].key_);
      delete localizer;
      slotReader_.clear(grp);
      if (!x) {
        return;
      }
      VLOG(1) << "convert feat group " << grp << " into column major";
      x = x->toColMajor();
      {
        Lock lk(mu_);
        data_[grp] = x;
      }
    };
    return model_.pull(&pullMsg);
  }

  void initModel(int time, int grp, std::promise<void> &wait) {
    // push the filtered keys to let the server build the key maps. when the
    // key-caching filter is used, the communication cost is little.
    model_.push(Parameter::request(grp, time, {}, bcdConf_.comm_filter()),
                model_[grp].key_);

    // fetch the initial value of the model
    model_.pull(
        Parameter::request(grp, time + 2, {time + 1}, bcdConf_.comm_filter()),
        model_[grp].key_, [this, grp, time, &wait]() {
          size_t n = model_[grp].key_.size();
          if (n > 0) {
            auto initW = model_.buffer(time + 2);
            CHECK_EQ(initW.values_.size(), 1);
            CHECK_EQ(initW.idxRange_.size(), n);
            CHECK_EQ(initW.channel_, grp);

            model_[grp].val_ = initW.values_[0];
            auto x = data_[grp];
            CHECK(x);
            if (dual_.empty()) {
              dual_.resize(x->rows(), 0);
            } else {
              CHECK_EQ(dual_.size(), x->rows());
            }
            if (bcdConf_.init_w().type() != ParamInitConfig::ZERO) {
              dual_.eigenVector() = *x * model_[grp].val_.eigenVector();
            }
          }
          wait.set_value();
        });
  }

  bool dataCache(const std::string &name, bool load) {}

  SlotReader slotReader_;

protected:
  /// @brief <feature group id, data>
  std::unordered_map<int, MatrixPtr<V>> data_;

  /// @brief label data
  MatrixPtr<V> label_;

  /// @brief dual = X * w
  DArray<V> dual_;
  std::mutex mu_;
  USING_BCD_COMP_NODE;
};

} // namespace mltools
