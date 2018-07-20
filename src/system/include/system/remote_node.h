/*
 * =====================================================================================
 *
 *       Filename:  remote_node.h
 *
 *    Description:  entity represent other worker & servers
 *
 *        Version:  1.0
 *        Created:  07/18/2018 21:17:42
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "filter/filter.h"
#include "postoffice.h"
#include "proto/task.pb.h"
#include "system/van.h"
#include "util/common.h"

namespace mltools {
// The representation of remote node, used by executor to track the interaction
// of customers: request or response.

/**
 * @brief Track the request using timestamp by Sender.
 */
class RequestTracker {
public:
  RequestTracker() {}
  ~RequestTracker() {}

  /// @brief return true if message from "ts" is marked as finished.
  bool isFinished(int ts) {
    return (ts < 0) || (((int)data_.size() > ts) && data_[ts]);
  }

  /// @brief mark "ts" as finished.
  void finish(int ts) {
    CHECK_GE(ts, 0);
    CHECK_LT(ts, 1000000);
    if ((int)data_.size() <= ts) {
      data_.resize(2 * ts + 5);
    }
    data_[ts] = true;
  }

private:
  std::vector<bool> data_;
};

/**
 * @brief A remote node represent other node.
 */
struct RemoteNode {
public:
  enum class RemoteNodeType : unsigned char { LEAF_NODE = 0, COMPOSITE_NODE };

  RemoteNode() { type_ = RemoteNodeType::LEAF_NODE; }
  explicit RemoteNode(RemoteNodeType type) : type_(type) {}
  ~RemoteNode() {
    for (auto &f : filters_) {
      delete f.second;
    }
  }

  void encodeMessage(Message *msg);
  void decodeMessage(Message *msg);

  /// @brief the node info.
  Node node_;
  bool alive_ = true;
  RemoteNodeType type_;

  /// @brief request to others/request from others.
  RequestTracker sentReqTracker_;
  RequestTracker recvReqTracker_;

  std::vector<RemoteNode *> group_;
  /// @brief used by server group node.
  std::vector<Range<Key>> keys_;

  // Some remote node used to represent composite node (i.e., node group).
  void addGroupNode(RemoteNode *rnode);
  void removeGroupNode(RemoteNode *rnode);

private:
  Filter *findFilterOrCreate(const FilterConfig &conf);
  std::unordered_map<int, Filter *> filters_;
};
} // namespace mltools
