/*
 * =====================================================================================
 *
 *       Filename:  van.h
 *
 *    Description:  main interface to network
 *
 *        Version:  1.0
 *        Created:  07/15/2018 10:28:08
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "proto/node.pb.h"
#include "system/message.h"
#include "util/common.h"
#include "util/macro.h"

namespace mltools {

/**
 * @brief Van used as the interface to network, all the messages from one node
 * to another node must go through Van. Van also maintain the connection to
 * scheduler and all other nodes.
 *
 */
class Van {
public:
  Van() {}
  ~Van();

  void init();

  void disconnect(const Node &node);
  bool connect(const Node &node);

  bool send(Message *msg, size_t *sendBytes);
  bool recv(Message *msg, size_t *recvBytes);

  static Node parseNode(const std::string &nodeStr);
  Node &myNode() { return myNode_; }
  Node &scheduler() { return scheduler_; }

private:
  void bind();

  static void freeData(void *data, void *hint) {
    if (hint == nullptr) {
      delete[](char *) data;
    } else {
      delete (DArray<char> *)hint;
    }
  }

  bool isScheduler() const { return myNode_.role() == Node::SCHEDULER; }

  // for scheduler: monitor the liveness of other nodes
  // for worker node: monitor the liveness of scheduler
  void monitor();

  void *context_ = nullptr;
  void *receiver_ = nullptr;
  Node myNode_;
  Node scheduler_;
  std::unordered_map<NodeID, void *> senders_;
  std::unordered_map<NodeID, std::string> hostnames_;

  DISALLOW_COPY_AND_ASSIGN(Van);

  void statistics();
  size_t sentToLocal_ = 0;
  size_t sentToOthers_ = 0;
  size_t receivedFromLocal_ = 0;
  size_t receivedFromOthers_ = 0;

  // for debug purpose only
  std::unordered_map<int, NodeID> fdToNodeId_;
  std::mutex fdToNodeIdMu_;
  std::thread *monitorThread_;
};
} // namespace mltools
