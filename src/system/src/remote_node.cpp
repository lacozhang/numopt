/*
 * =====================================================================================
 *
 *       Filename:  remote_node.cpp
 *
 *    Description:  implementation of remote_node.h
 *
 *        Version:  1.0
 *        Created:  07/18/2018 21:20:17
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "system/remote_node.h"
#include "system/customer.h"
#include "util/dynamic_array_impl.h"

namespace mltools {
Filter *RemoteNode::findFilterOrCreate(const mltools::FilterConfig &conf) {
  auto id = conf.type();
  auto it = filters_.find(id);
  if (it == filters_.end()) {
    filters_[id] = Filter::create(conf);
    it = filters_.find(id);
  }
  return it->second;
}

void RemoteNode::encodeMessage(mltools::Message *msg) {
  const auto &tk = msg->task_;
  for (int i = 0; i < tk.filter_size(); ++i) {
    findFilterOrCreate(tk.filter(i))->encode(msg);
  }
}

void RemoteNode::decodeMessage(mltools::Message *msg) {
  const auto &tk = msg->task_;
  for (int i = tk.filter_size() - 1; i >= 0; --i) {
    findFilterOrCreate(tk.filter(i))->decode(msg);
  }
}

void RemoteNode::addGroupNode(mltools::RemoteNode *rnode) {
  CHECK_NOTNULL(rnode);
  int pos = 0;
  Range<Key> kr(rnode->node_.key());
  while (pos < group_.size()) {
    if (kr.inLeft(Range<Key>(group_[pos]->node_.key()))) {
      break;
    }
    ++pos;
  }
  group_.insert(group_.begin() + pos, rnode);
  keys_.insert(keys_.begin() + pos, kr);
}

void RemoteNode::removeGroupNode(mltools::RemoteNode *rnode) {
  CHECK(keys_.size() == group_.size());
  for (int i = 0; i < group_.size(); ++i) {
    if (group_[i] == rnode) {
      group_.erase(group_.begin() + i);
      keys_.erase(keys_.begin() + i);
    }
  }
}
} // namespace mltools
