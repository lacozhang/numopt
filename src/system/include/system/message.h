/*
 * =====================================================================================
 *
 *       Filename:  message.h
 *
 *    Description:  message is the main representation of info between nodes
 *
 *        Version:  1.0
 *        Created:  07/15/2018 10:26:35
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "proto/filter.pb.h"
#include "proto/task.pb.h"
#include "util/common.h"
#include "util/dynamic_array.h"

namespace mltools {

// A message is the entity of information shared between nodes
// Generally, it contains everything essential info to sent a request or
// response For a sender, a message stores the information for request; For a
// receiver, a message stores the request from another node and response to the
// request.
struct Message {
public:
  static const int kInvalidTime = -1;
  Message() {}
  Message(const Task &tk, const NodeID &dst) : task_(tk), recver_(dst) {}
  explicit Message(const Task &tk) : task_(tk) {}

  Task task_;

  // keys
  bool has_key() const { return !key_.empty(); }
  template <typename T> void set_key(const DArray<T> &key);
  void clear_key() {
    task_.clear_has_key();
    key_.clear();
  }
  DArray<char> key_;

  // values
  template <typename T> void add_value(const DArray<T> &value);
  void clear_value() {
    task_.clear_value_type();
    value_.clear();
  }
  std::vector<DArray<char>> value_;

  // clear keys and values
  void clear_data() {
    clear_key();
    clear_value();
  }

  bool has_data() const { return key_.size() > 0 || value_.size() > 0; }

  // memory size in bytes
  size_t mem_size();

  FilterConfig *add_filter(FilterConfig::Type type);

  // local state management information
  NodeID sender_; // sender node id
  NodeID recver_; // receiver node id

  bool replied_ = false; // true if this message has been replied.
  bool finished_ = true; // true if the request associated with this message has
                         // been processed.
  bool valid_ = true;    // an invalid message will not be sent, but marked with
                         // finished directly.
  bool terminate_ = false; // used to stop the sending thread in PostOffice.

  typedef std::function<void()> Callback;
  Callback callback; // the callback when the associated request is finished.

  // debug
  std::string ShortDebugString() const;
  std::string DebugString() const;

private:
  // helper
  template <typename V> static DataType EncodeType();
};

template <typename V> DataType Message::EncodeType() {
  if (std::is_same<V, uint32>::value) {
    return DataType::UINT32;
  } else if (std::is_same<V, uint64>::value) {
    return DataType::UINT64;
  } else if (std::is_same<V, int32>::value) {
    return DataType::INT32;
  } else if (std::is_same<V, int64>::value) {
    return DataType::INT64;
  } else if (std::is_same<typename std::remove_cv<V>::type, float>::value) {
    return DataType::FLOAT;
  } else if (std::is_same<V, double>::value) {
    return DataType::DOUBLE;
  } else if (std::is_same<V, uint8>::value) {
    return DataType::UINT8;
  } else if (std::is_same<V, int8>::value) {
    return DataType::INT8;
  } else if (std::is_same<V, char>::value) {
    return DataType::CHAR;
  }
  return DataType::OTHER;
}

template <typename T> void Message::set_key(const DArray<T> &key) {
  task_.set_key_type(EncodeType<T>());
  if (has_key()) {
    clear_key();
  }
  task_.set_has_key(true);
  key_ = DArray<char>(key);
  if (!task_.has_key_range()) {
    Range<Key>::all().to(task_.mutable_key_range());
  }
}

template <typename T> void Message::add_value(const DArray<T> &value) {
  task_.add_value_type(EncodeType<T>());
  value_.push_back(DArray<char>(value));
}

// slice a message "msg" according to key ranges "krs". "msg.key" must be
// ordered, and each value entry must have the same length.
template <typename K>
void sliceKOfVMessage(const Message &msg, const std::vector<Range<Key>> &krs,
                      std::vector<Message *> *rets) {
  if (rets == nullptr) {
    LOG(ERROR) << "Empty array of message";
    return;
  }
  CHECK_EQ(krs.size(), rets->size());

  // find the positions in msg.key
  size_t n = krs.size();
  std::vector<size_t> pos(n + 1);
  DArray<K> keys(msg.key_);
  Range<Key> msgKeyRange(msg.task_.key_range());

  // get the index range of each key range in *msg.key*
  for (int i = 0; i < n; ++i) {
    if (i == 0) {
      K k = static_cast<K>(msgKeyRange.project(krs[0].begin()));
      pos[0] = std::lower_bound(keys.begin(), keys.end(), k) - keys.begin();
    } else {
      CHECK_EQ(krs[i - 1].end(), krs[i].begin());
    }
    K k = static_cast<K>(msgKeyRange.end());
    pos[i + 1] = std::lower_bound(keys.begin(), keys.end()) - keys.begin();
  }

  // set the keys and values for each message independently.
  for (int i = 0; i < n; ++i) {
    auto ret = CHECK_NOTNULL((*rets)[i]);
    if (krs[i].setIntersection(msgKeyRange).empty()) {
      // remote node i doesn't maintain key range
      ret->valid_ = false;
    } else {
      ret->valid_ = true;
      if (keys.empty()) {
        continue;
      }
      SizeR lr(pos[i], pos[i + 1]);
      ret->set_key(keys.segment(lr));
      for (auto &v : msg.value_) {
        size_t k = v.size() / keys.size();
        CHECK_EQ(k * keys.size(), v.size());
        ret->value_.push_back(v.segment(lr * k));
      }
    }
  }
}

} // namespace mltools
