/*
 * =====================================================================================
 *
 *       Filename:  executor.h
 *
 *    Description:  maintain the task request & response for customer
 *
 *        Version:  1.0
 *        Created:  07/18/2018 21:14:49
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "system/message.h"
#include "system/remote_node.h"
namespace mltools {

const static NodeID kGroupPrefix = "all_";
/// @brief all server nodes
const static NodeID kServerGroup = kGroupPrefix + "servers";
/// @brief all worker nodes
const static NodeID kWorkerGroup = kGroupPrefix + "workers";
/// @brief all server + worker nodes
const static NodeID kCompGroup = kGroupPrefix + "comp_nodes";
/// @brief backup of current node which maintaining the replica of key segment
const static NodeID kReplicaGroup = kGroupPrefix + "replicas";
/// @brief the owner nodes of the key segments this node backuped for.
const static NodeID kOwnerGroup = kGroupPrefix + "owners";
/// @brief all live nodes, including scheduler/server/worker/unused nodes.
const static NodeID kLiveGroup = kGroupPrefix + "lives";

/// @brief Executor maintains the connections of customers. Using separate
/// thread to processing request/response from other customers.
class Executor {
public:
  Executor(Customer &obj);
  ~Executor();

  // communication & synchronization
  /// @brief send request to other customers.
  int submit(Message *msg);

  /// @brief send response of request from other customers.
  void reply(Message *request, Message *response);

  void accept(Message *msg);

  /// @brief blocked until request to other cusomer has been processed by others
  /// & get response successfully.
  void waitSentReq(int timestamp);

  /// @brief wait until request from others has been processed by self.
  void waitRecvReq(int timestamp, const NodeID &sender);
  void finishRecvReq(int timestamp, const NodeID &sender);
  int queryRecvReq(int timestamp, const NodeID &sender);

  /// @brief the last received request from other customers.
  inline std::shared_ptr<Message> lastRequest() { return lastRequest_; }

  /// @brief the last received response from other customers.
  inline std::shared_ptr<Message> lastResponse() { return lastResponse_; }

  int time() {
    Lock l(nodeMu_);
    return time_;
  }

  // node managment
  void addNode(const Node &node);
  void removeNode(const Node &node);
  void replaceNode(const Node &oldNode, const Node &newNode);

private:
  void run() {
    while (!done_) {
      if (pickActiveMsg()) {
        processActiveMsg();
      }
    }
  }

  /// @brief return true if there are message available for processing.
  bool pickActiveMsg();
  void processActiveMsg();

  /// @brief received message
  std::list<Message *> recvMsgs_;
  std::mutex msgMu_;

  std::shared_ptr<Message> activeMsg_, lastRequest_, lastResponse_;
  std::condition_variable dagCond_;

  // @brief connections to other customers.
  std::mutex nodeMu_;
  std::condition_variable recvReqCond_;
  std::condition_variable sentReqCond_;

  // @brief the interaction of this customer with other node.
  std::unordered_map<NodeID, RemoteNode> remoteNode_;

  inline RemoteNode *getRNode(const NodeID &nodeID) {
    auto it = remoteNode_.find(nodeID);
    CHECK(it != remoteNode_.end()) << "node " << nodeID << " not connected";
    return &(it->second);
  }

  inline bool checkFinished(RemoteNode *rnode, int timestamp, bool sent);
  inline int numFinished(RemoteNode *rnode, int timestamp, bool send);

  std::vector<NodeID> groupIDs() {
    static std::vector<NodeID> ids = {kServerGroup,  kWorkerGroup, kCompGroup,
                                      kReplicaGroup, kOwnerGroup,  kLiveGroup};
    return ids;
  }

  Customer &obj_;
  PostOffice &sys_;
  Node myNode_;
  int numReplicas_ = 0;
  int time_ = Message::kInvalidTime;
  struct ReqInfo {
    NodeID recver_;
    Message::Callback callback_;
  };

  /// @brief request sent to others.
  /// <timestamp, (receiver, callback)>
  std::unordered_map<int, ReqInfo> sentReqs_;

  bool done_ = false;
  std::thread *thread_ = nullptr;
};
} // namespace mltools
