/*
 * =====================================================================================
 *
 *       Filename:  message.cpp
 *
 *    Description:  implementation of message.h
 *
 *        Version:  1.0
 *        Created:  07/15/2018 10:27:14
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "system/message.h"

namespace mltools {
FilterConfig *Message::add_filter(FilterConfig::Type type) {
  auto ptr = task_.add_filter();
  ptr->set_type(type);
  return ptr;
}

size_t Message::mem_size() {
  size_t nbytes = task_.SpaceUsed() + key_.memSize();
  for (auto &v : value_) {
    nbytes += v.memSize();
  }
  return nbytes;
}

std::string Message::ShortDebugString() const {
  std::stringstream ss;
  if (key_.size())
    ss << "key [" << key_.size() << "] ";
  if (value_.size()) {
    ss << "value [";
    for (int i = 0; i < value_.size(); ++i) {
      ss << value_[i].size();
      if (i < value_.size() - 1)
        ss << ",";
    }
    ss << "] ";
  }
  auto t = task_;
  t.clear_msg();
  ss << t.ShortDebugString();
  return ss.str();
}

std::string Message::DebugString() const {
  std::stringstream ss;
  ss << "[message]: " << sender_ << "=>" << recver_
     << "[task]:" << task_.ShortDebugString() << "\n[key]:" << key_.size()
     << "\n[" << value_.size() << " value]: ";
  for (const auto &x : value_) {
    ss << x.size() << " ";
  }
  return ss.str();
}
} // namespace mltools
