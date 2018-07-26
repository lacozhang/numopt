/*
 * =====================================================================================
 *
 *       Filename:  kv_layer.h
 *
 *    Description:  value is layer
 *
 *        Version:  1.0
 *        Created:  07/23/2018 21:10:29
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once
#include "parameter/parameter.h"
#include "system/sysutil.h"

namespace mltools {

template <typename V> class KVLayerUpdater {
public:
  void init(int id, size_t size, V *data) {}

  void update(int id, size_t size, const V *recvData, V *data) {}
};

template <typename V, class Updater = KVLayerUpdater<V>>
class KVLayer : public Parameter {
public:
  KVLayer(size_t partitionThr = 1000, int id = NextCustomerID())
      : Parameter(id), partitionThr_(partitionThr) {}

  void setUpdater(Updater *updater) { updater_ = updater; }

  DArray<V> operator[](int key) {
    Lock l(mu_);
    return layer_[key];
  }

  DArray<V> layer(int key) {
    Lock l(mu_);
    return layer_[key];
  }

  int push(const Task &task, V *data, size_t size, bool zerocopy = false);
  int pull(const Task &task, V *data, size_t size,
           Message::Callback callback = Message::Callback());

  virtual void slice(const Message &request, const std::vector<Range<Key>> &krs,
                     std::vector<Message *> *msgs) override;
  virtual void getValue(Message *msg) override;
  virtual void setValue(const Message *msg) override;

protected:
  std::mutex mu_;
  std::unordered_map<int, DArray<V>> layer_;
  size_t partitionThr_;
  Updater *updater_ = nullptr;
  int call_ = 0;
};

template <typename V, class Updater>
int KVLayer<V, Updater>::push(const Task &task, V *data, size_t size,
                              bool zerocopy) {
  DArray<V> val;
  if (zerocopy) {
    val = DArray<V>(data, size, false);
  } else {
    val.copyFrom(data, size);
  }

  Message push(task, kServerGroup);
  Range<Key>(0, size).to(push.task_.mutable_key_range());
  push.add_value(val);
  return Parameter::push(&push);
}

template <typename V, class Updater>
int KVLayer<V, Updater>::pull(const mltools::Task &task, V *data, size_t size,
                              Message::Callback callback) {
  int id = task.key_channel();
  if (data == nullptr) {
    if (layer_[id].size() != size) {
      layer_[id].resize(size, 0);
    }
  } else {
    layer_[id] = DArray<V>(data, size, false);
  }
  Message pull(task, kServerGroup);
  Range<Key>(0, size).to(pull.task_.mutable_key_range());
  if (callback) {
    pull.callback = callback;
  }
  return Parameter::pull(&pull);
}

template <typename V, class Updater>
void KVLayer<V, Updater>::slice(const mltools::Message &request,
                                const std::vector<Range<Key>> &krs,
                                std::vector<Message *> *msgs) {
  size_t n = krs.size();
  int key = request.task_.key_channel();
  Range<Key> kr(request.task_.key_range());
  for (int i = 0; i < n; ++i) {
    Message *msg = (*msgs)[i];
    auto mutKr = msg->task_.mutable_key_range();
    if (kr.size() < partitionThr_) {
      int k = (key * 991) % n;
      if (i == k) {
        kr.to(mutKr);
      } else {
        Range<Key>(0, 0).to(mutKr);
        msg->valid_ = false;
      }
    } else {
      kr.evenDivide(n, i).to(mutKr);
    }
  }

  for (size_t i = 0; i < request.value_.size(); ++i) {
    DArray<V> data(request.value_[i]);
    CHECK_EQ(data.size(), kr.size());
    for (size_t j = 0; j < n; ++j) {
      Message *msg = (*msgs)[i];
      if (msg->valid_) {
        Range<Key> kr(msg->task_.key_range());
        msg->add_value(data.segment(kr));
      }
    }
  }
}

template <typename V, class Updater>
void KVLayer<V, Updater>::getValue(mltools::Message *msg) {
  mu_.lock();
  auto &myVal = layer_[msg->task_.key_channel()];
  mu_.unlock();

  Range<Key> kr(msg->task_.key_range());
  if (myVal.empty()) {
    myVal.resize(kr.size(), 0);
    CHECK_NOTNULL(updater_)->init(msg->task_.key_channel(), myVal.size(),
                                  myVal.data());
  }

  CHECK_EQ(myVal.size(), kr.size());
  DArray<V> sendData(kr.size());
  sendData.copyFrom(myVal);
  msg->add_value(sendData);
}

template <typename V, class Updater>
void KVLayer<V, Updater>::setValue(const mltools::Message *msg) {
  CHECK_EQ(msg->value_.size(), 1);
  DArray<V> recvData(msg->value_[0]);
  Range<Key> kr(msg->task_.key_range());
  CHECK_EQ(kr.size(), recvData.size());
  int key = msg->task_.key_channel();
  mu_.lock();
  auto &myVal = layer_[key];
  mu_.unlock();

  if (IsWorker()) {
    if (myVal.empty()) {
      myVal.resize(kr.size(), 0);
    }
    CHECK_GE(myVal.size(), kr.end());
    myVal.segment(kr).copyFrom(recvData);
  } else if (IsServer()) {
    if (myVal.empty()) {
      myVal.resize(kr.size(), 0);
      CHECK_NOTNULL(updater_)->init(key, kr.size(), myVal.data());
    }

    CHECK_GE(myVal.size(), kr.size());
    CHECK_NOTNULL(updater_)->update(key, kr.size(), recvData.data(),
                                    myVal.data());
  }
}
} // namespace mltools
