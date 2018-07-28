/*
 * =====================================================================================
 *
 *       Filename:  kv_map.h
 *
 *    Description:  value is map
 *
 *        Version:  1.0
 *        Created:  07/23/2018 21:10:54
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
#include "util/file.h"

namespace mltools {

/// @brief default kmap entry type
template <typename V> struct KVMapEntry {
  void get(V *data, void *state) { *data = value_; }
  void set(const V *data, void *state) { value_ = *data; }
  V value_;
};

/// @brief default state type
struct KVMapState {
  void update() {}
};

template <typename K, typename V, typename E = KVMapEntry<V>,
          typename S = KVMapState>
class KVMap : public Parameter {
public:
    KVMap(int k = 1, int id = NextCustomerID()) : Parameter(id), k_(k) {
      CHECK_GT(k, 0);
    }

    virtual ~KVMap() {}

    void setState(const S &s) { state_ = s; }

    virtual void slice(const Message &request,
                       const std::vector<Range<Key>> &krs,
                       std::vector<Message *> *msgs) override {
      sliceKOfVMessage<V>(request, krs, msgs);
    }

    virtual void getValue(Message *msg) override;
    virtual void setValue(const Message *msg) override;
    virtual void writeToFile(std::string filepath);

  protected:
    int k_;
    S state_;
    std::unordered_map<K, E> data_;
};

template <typename K, typename V, typename E, typename S>
void KVMap<K, V, E, S>::getValue(mltools::Message *msg) {
  CHECK_NOTNULL(msg);
  DArray<K> recvKeys(msg->key_);
  size_t n = recvKeys.size();
  DArray<V> val(n * k_);
  for (int i = 0; i < n; ++i) {
    data_[recvKeys[i]].get(val.data() + i * k_, &state_);
  }
  msg->add_value(val);
}

template <typename K, typename V, typename E, typename S>
void KVMap<K, V, E, S>::setValue(const mltools::Message *msg) {
  DArray<K> recvKeys(msg->key_);
  size_t n = recvKeys.size();
  CHECK_EQ(msg->value_.size(), 1);
  DArray<V> val(msg->value_[0]);
  CHECK_EQ(n * k_, val.size());
  for (int i = 0; i < n; ++i) {
    data_[recvKeys[i]].set(val.data() + i * k_, &state_);
  }
  state_.update();
}

template <typename K, typename V, typename E, typename S>
void KVMap<K, V, E, S>::writeToFile(std::string filepath) {
  if (!dirExists(getPath(filepath))) {
    dirCreate(getPath(filepath));
  }
  std::ofstream sink(filepath);
  CHECK(sink.good());
  V v;
  for (auto &e : data_) {
    e.second.get(&v, &state_);
    if (v != 0) {
      sink << e.first << "\t" << v << std::endl;
    }
  }
}
} // namespace mltools
