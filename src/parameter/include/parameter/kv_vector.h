/*
 * =====================================================================================
 *
 *       Filename:  kv_vector.h
 *
 *    Description:  value is vector; distributed representation
 *
 *        Version:  1.0
 *        Created:  07/23/2018 21:09:50
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "filter/frequency_filter.h"
#include "parameter/parameter.h"
#include "system/sysutil.h"
#include "util/parallel_ordered_match.h"

namespace mltools {

/**
 * @brief key-value vectors
 *  keys of type K, value is a fixed length array of type V. Physical stroage
 * format: key_0,  ... key_n val_00, ... val_n0 val_01, ... val_n1
 *   ...    ... ...
 *  val_0k, ... val_nk
 *  keys are ordered and unique. values stored in a column-major format. support
 * multiple channels.
 */
template <typename K, typename V> class KVVector : public Parameter {
public:
  /**
   * @brief constructor
   *
   * @param bufferValue if true, then the received data in push request or pull
   * response is merged into data_ directly.
   * @param k value entry size
   * @param id id of customer
   */
  KVVector(bool bufferValue = false, int k = 1, int id = NextCustomerID())
      : Parameter(id), bufferValue_(bufferValue), k_(k) {
    CHECK_GT(k, 0);
  }
  virtual ~KVVector() {}
  struct KVPairs {
    DArray<K> key_;
    DArray<V> val_;
  };

  struct Buffer {
    int channel_;
    SizeR idxRange_;
    std::vector<DArray<V>> values_;
  };

  /// @brief return the parameter in challel (i.e, feature group)
  KVPairs &operator[](int chl) {
    Lock l(mu_);
    return data_[chl];
  }

  /// @brief clear both the key and value at channel
  void clear(int chl) {
    Lock l(mu_);
    data_[chl].key_.clear();
    data_[chl].val_.clear();
  }

  /// @brief return cached data at timestamp
  Buffer buffer(int ts) {
    Lock l(mu_);
    return buffer_[ts];
  }

  void clearBuffer(int ts) {
    for (auto &v : buffer_[ts].values_) {
      v.clear();
    }
  }

  void clearFilter() { freqFilters_.clear(); }

  /**
   * @brief push data into servers
   *
   * @return time of this request.
   */
  int push(const Task &request, const DArray<K> &keys,
           const std::initializer_list<DArray<V>> &values = {},
           const Message::Callback &callback = Message::Callback());

  /**
   * @brief pull the data from servers.
   */
  int pull(const Task &request, const DArray<K> &keys,
           const Message::Callback &callback = Message::Callback());

  virtual void slice(const Message &request, const std::vector<Range<Key>> &krs,
                     std::vector<Message *> *msgs) override {
    sliceKOfVMessage<K>(request, krs, msgs);
  }

  virtual void getValue(Message *msg) override;
  virtual void setValue(const Message *msg) override;
  using Parameter::pull;
  using Parameter::push;

protected:
  int k_;                                 // size of value entry
  std::unordered_map<int, KVPairs> data_; // mapping <channel, data>
  bool bufferValue_;
  std::unordered_map<int, Buffer> buffer_; // <timestamp, buffer>
  std::mutex mu_;
  std::unordered_map<int, FrequencyFilter<K, uint8>>
      freqFilters_; // <channel, count of features>
};

template <typename K, typename V>
void KVVector<K, V>::setValue(const Message *msg) {
  DArray<K> recvKeys(msg->key_);
  VLOG(1) << "setValue : from " << msg->sender_ << " to " << msg->recver_;
  if (recvKeys.empty()) {
    return;
  }
  int chl = msg->task_.key_channel();

  // upload count statistics of featues to this node
  if (msg->task_.param().has_tail_filter() && msg->task_.request()) {
    const auto &tailFilter = msg->task_.param().tail_filter();
    CHECK(tailFilter.insert_count());
    CHECK_EQ(msg->value_.size(), 1);
    DArray<uint8> count(msg->value_[0]);
    CHECK_EQ(count.size(), recvKeys.size());
    auto &filter = freqFilters_[chl];
    if (filter.empty()) {
      double w = (double)std::max(sys_.manager().numWorkers(), 1);
      int n = (w / log(w + 1)) * tailFilter.countmin_n();
      int k = tailFilter.countmin_k();
      filter.resize(n, k);
      VLOG(1) << "Resize channel " << chl << " into n : " << n << " K: " << k;
    }
    filter.insertKeys(recvKeys, count);
    return;
  }

  mu_.lock();
  auto &kv = data_[chl];
  mu_.unlock();

  if (msg->value_.empty()) {
    // merge keys without value update
    kv.key_ = kv.key_.setUnion(recvKeys);
    // clear the values, because value is not alighed with key anymore.
    kv.val_.clear();
    VLOG(1) << "merge new keys, now size " << kv.key_.size();
    return;
  } else if (kv.key_.empty()) {
    LOG(ERROR) << "No keys at channel " << chl;
    return;
  }

  for (int i = 0; i < msg->value_.size(); ++i) {
    DArray<V> recvVal(msg->value_[i]);
    if (!bufferValue_) {
      CHECK_EQ(i, 0) << "can only support one value";
      CHECK_EQ(recvVal.size(), recvKeys.size() * k_);
      if (kv.val_.empty()) {
        kv.val_ = DArray<V>(kv.key_.size() * k_, 0);
      }
      CHECK_EQ(kv.key_.size() * k_, kv.val_.size());
      size_t n = ParallelOrderedMatch(recvKeys, recvVal, kv.key_, &kv.val_,
                                      k_, AssignOpType::PLUS);
      CHECK_EQ(n, recvKeys.size() * k_);
      VLOG(1) << recvKeys.size() << " matched";
    } else {
      mu_.lock();
      auto &buf = buffer_[msg->task_.time()];
      mu_.unlock();

      if (i == 0) {
        SizeR idxRange = kv.key_.findRange(Range<K>(msg->task_.key_range()));
        if (buf.values_.size() == 0) {
          buf.values_.resize(msg->value_.size());
          buf.idxRange_ = idxRange;
          buf.channel_ = chl;
        } else {
          CHECK_EQ(buf.idxRange_.begin(), idxRange.begin());
          CHECK_EQ(buf.idxRange_.end(), idxRange.end());
          CHECK_EQ(buf.channel_, chl);
        }
      }

      size_t k = recvVal.size() / recvKeys.size();
      size_t n = ParallelOrderedMatch(recvKeys, recvVal,
                                      kv.key_.segment(buf.idxRange_),
                                      &buf.values_[i], k, AssignOpType::PLUS);
      CHECK_EQ(n, recvKeys.size() * k);
      VLOG(1) << "matched " << n << " keys";
    }
  }
}

template <typename K, typename V> void KVVector<K, V>::getValue(Message *msg) {
  CHECK_NOTNULL(msg);
  DArray<K> recvKeys(msg->key_);
  VLOG(1) << "Get request to retrieve " << recvKeys.size() << " keys";
  if (recvKeys.empty()) {
    return;
  }
  int chl = msg->task_.key_channel();

  if (msg->task_.param().has_tail_filter()) {
    const auto &tail = msg->task_.param().tail_filter();
    CHECK(tail.has_freq_threshold());
    auto &filter = freqFilters_[chl];
    msg->key_ = filter.queryKeys(recvKeys, tail.freq_threshold());
    return;
  }

  Lock l(mu_);
  auto &kv = data_[chl];
  CHECK_EQ(kv.key_.size() * k_, kv.val_.size());
  DArray<V> val;
  size_t n = ParallelOrderedMatch(kv.key_, kv.val_, recvKeys, &val, k_);
  CHECK_LE(n, recvKeys.size() * k_);
  VLOG(1) << "matched " << n << " keys";
  msg->clear_value();
  msg->add_value(val);
}

template <typename K, typename V>
int KVVector<K, V>::push(const mltools::Task &request, const DArray<K> &keys,
                         const std::initializer_list<DArray<V>> &values,
                         const Message::Callback &callback) {
  Message push(request, kServerGroup);
  push.set_key(keys);
  for (auto &v : values) {
    if (!v.empty()) {
      push.add_value(v);
    }
  }
  if (callback) {
    push.callback = callback;
  }
  return Parameter::push(&push);
}

template <typename K, typename V>
int KVVector<K, V>::pull(const mltools::Task &request, const DArray<K> &keys,
                         const Message::Callback &callback) {
  Message pull(request, kServerGroup);
  pull.set_key(keys);
  if (callback) {
    pull.callback = callback;
  }
  return Parameter::pull(&pull);
}
} // namespace mltools
