/*
 * =====================================================================================
 *
 *       Filename:  sgd.h
 *
 *    Description:  first order method base
 *
 *        Version:  1.0
 *        Created:  07/27/2018 13:48:05
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once

#include "data/stream_reader.h"
#include "filter/frequency_filter.h"
#include "learner/workload_pool.h"
#include "proto/sgd.pb.h"
#include "system/assigner.h"
#include "system/monitor.h"
#include "system/sysutil.h"
#include "util/localizer.h"
#include "util/producer_consumer.h"

namespace mltools {

/// @brief base class of a scheduler node
class ISGDScheduler : public App {
public:
  ISGDScheduler() : App() {}
  virtual ~ISGDScheduler();

  /// @brief App class, main entry method.
  virtual void run() override;

  /// @brief Customer, get response.
  virtual void processResponse(Message *response) override;
  virtual void processRequest(Message *request) override;

protected:
  /// @brief function used for MonitorMaster
  virtual void showProgress(double time,
                            std::unordered_map<NodeID, SGDProgress> *progress);

  /// @brief merge the progress for MonitorMaster
  virtual void mergeProgress(const SGDProgress &src, SGDProgress *dst);

  void sendWorkload(const NodeID &recver);
  MonitorMaster<SGDProgress> monitor_;
  WorkloadPool *workpoll_ = nullptr;

  size_t numExpProcessed_ = 0;
  bool showProgHead_ = true;
};

class ISGDComputeNode : public App {
public:
  ISGDComputeNode() : App(), reporter_(SchedulerID()) {}
  virtual ~ISGDComputeNode() {}

protected:
  MonitorSlaver<SGDProgress> reporter_;
};

template <typename V> class MinibatchReader {
public:
  MinibatchReader() {}
  ~MinibatchReader() {}

  void initReader(const DataConfig &file, int batchSize, int dataBuf = 1000) {
    reader_.init(file);
    batchSize_ = batchSize;
    dataPrefetcher_.setCapacity(dataBuf);
  }

  void initFilter(size_t n, int k, int freq) {
    filter_.resize(n, k);
    keyThreshold_ = freq;
  }

  void start() {
    std::function<bool(MatrixPtrList<V>&, size_t&)> func = [this](MatrixPtrList<V> &data, size_t &size) -> bool {
      bool ret = reader_.readMatrices(batchSize_, &data);
      for (const auto &mat : data) {
        size += mat->memSize();
      }
      return ret;
    };
    dataPrefetcher_.StartProducer(func);
  }

  bool read(MatrixPtr<V> &Y, MatrixPtr<V> &X, DArray<Key> &key) {
    MatrixPtrList<V> data;
    if (!dataPrefetcher_.pop(data)) {
      return false;
    }
    CHECK_GE(data.size(), 2);
    Y = data[0];
    DArray<Key> uniqKey;
    DArray<uint8> keyCnt;
    Localizer<Key, V> localizer;
    // get key & corresponding count
    localizer.countUniqIndex(data[1], &uniqKey, &keyCnt);
    // insert the statistics
    filter_.insertKeys(uniqKey, keyCnt);
    // fileter the key by statistics
    key = filter_.queryKeys(uniqKey, keyThreshold_);
    // remap the feature index
    X = localizer.remapIndex(key);
    return true;
  }

private:
  int batchSize_ = 1000;
  StreamReader<V> reader_;
  FrequencyFilter<Key, uint8> filter_;
  int keyThreshold_ = 0;
  ProducerConsumer<MatrixPtrList<V>> dataPrefetcher_;
};
} // namespace mltools
