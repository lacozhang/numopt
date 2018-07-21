/*
 * =====================================================================================
 *
 *       Filename:  executor.cpp
 *
 *    Description:  implementation of executor.h
 *
 *        Version:  1.0
 *        Created:  07/18/2018 21:16:49
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#include "system/executor.h"
#include "system/customer.h"
#include <thread>

namespace mltools {
Executor::Executor(Customer &obj) : obj_(obj), sys_(PostOffice::getInstance()) {
  myNode_ = PostOffice::getInstance().manager().van().myNode();
  for (auto id : groupIDs()) {
    Node node;
    node.set_role(Node::GROUP);
    node.set_id(id);
    addNode(node);
  }
  thread_ = new std::thread(&Executor::run, this);
}

void Executor::addNode(const mltools::Node &node) {
  Lock l(nodeMu_);
  VLOG(1) << "customer " << obj_.id()
          << " add node : " << node.ShortDebugString();
  // usually this information is updated by scheduler.
  if (node.id() == myNode_.id()) {
    myNode_ = node;
  }
  auto id = node.id();
  if (remoteNode_.find(id) != remoteNode_.end()) {
    // update remote node information.
    auto r = getRNode(id);
    CHECK(r->alive_);
    r->node_ = node;
    for (const NodeID &gid : groupIDs()) {
      remoteNode_[gid].removeGroupNode(r);
    }
  } else {
    remoteNode_[id].node_ = node;
  }

  // add "node" into group.
  auto role = node.role();
  auto w = getRNode(id);
  if (role != Node::GROUP) {
    remoteNode_[id].type_ = RemoteNode::RemoteNodeType::LEAF_NODE;
    remoteNode_[id].addGroupNode(w);
    remoteNode_[kLiveGroup].addGroupNode(w);
  } else {
    remoteNode_[id].type_ = RemoteNode::RemoteNodeType::COMPOSITE_NODE;
  }

  if (role == Node::SERVER) {
    remoteNode_[kServerGroup].addGroupNode(w);
    remoteNode_[kCompGroup].addGroupNode(w);
  }

  if (role == Node::WORKER) {
    remoteNode_[kWorkerGroup].addGroupNode(w);
    remoteNode_[kCompGroup].addGroupNode(w);
  }

  // update replica group and owner group if necessary
  // replica group/owner group works for server node only.
  if (node.role() != Node::SERVER || myNode_.role() != Node::SERVER) {
    return;
  }
  if (numReplicas_ <= 0) {
    return;
  }

  CHECK(numReplicas_ < remoteNode_[kServerGroup].group_.size());

  const auto &servers = remoteNode_[kServerGroup];
  for (int i = 0; i < servers.group_.size(); ++i) {
    auto s = servers.group_[i];
    if (s->node_.id() != myNode_.id()) {
      continue;
    }

    // remake the replica group; replica group is just before me.
    auto &replicas = remoteNode_[kReplicaGroup];
    replicas.group_.clear();
    replicas.keys_.clear();
    for (int j = std::max(i - numReplicas_, 0); j < i; ++j) {
      replicas.group_.push_back(servers.group_[j]);
      replicas.keys_.push_back(servers.keys_[j]);
    }

    // remake the owner group; owner group is just after me.
    auto &owners = remoteNode_[kOwnerGroup];
    owners.group_.clear();
    owners.keys_.clear();
    for (int j = i; j < std::max((int)servers.group_.size(), i + numReplicas_);
         ++j) {
      owners.group_.push_back(servers.group_[j]);
      owners.keys_.push_back(servers.keys_[j]);
    }
    break;
  }
}

Executor::~Executor() {
  if (done_) {
    return;
  }
  done_ = true;
  // Wake thread
  { Lock l(msgMu_); }
  dagCond_.notify_one();

  CHECK_NOTNULL(thread_)->join();
  delete thread_;
}

bool Executor::pickActiveMsg() {
  std::unique_lock<std::mutex> lk(msgMu_);
  auto it = recvMsgs_.begin();
  while (it != recvMsgs_.end()) {
    bool process = true;
    Message *msg = *it;
    CHECK(msg);
    CHECK(!msg->task_.control());

    Lock l(nodeMu_);
    auto rnode = getRNode(msg->sender_);

    // check remote node is still alive.
    if (!rnode->alive_) {
      LOG(WARNING) << myNode_.id() << " : remote node "
                   << rnode->node_.ShortDebugString()
                   << " is not alive, ignore received message "
                   << msg->ShortDebugString();
      it = recvMsgs_.erase(it);
      delete msg;
      continue;
    }

    // check if double receiving
    bool req = msg->task_.request();
    int ts = msg->task_.time();
    if ((req && rnode->recvReqTracker_.isFinished(ts)) ||
        (!req && rnode->sentReqTracker_.isFinished(ts))) {
      LOG(WARNING) << myNode_.id() << " : rnode " << msg->sender_
                   << " receive message twice, ignore "
                   << msg->ShortDebugString();
      it = recvMsgs_.erase(it);
      delete msg;
      continue;
    }

    // check for dependency constraint
    if (req) {
      for (int i = 0; i < msg->task_.wait_time_size(); ++i) {
        int waitTime = msg->task_.wait_time(i);
        if (waitTime <= Message::kInvalidTime) {
          continue;
        }
        if (!rnode->recvReqTracker_.isFinished(waitTime)) {
          process = false;
          ++it;
          break;
        }
      }
    }

    if (process) {
      VLOG(1) << obj_.id() << ": pick the "
              << std::distance(recvMsgs_.begin(), it) << "-th message in ["
              << recvMsgs_.size() << "] from " << msg->sender_ << ": "
              << msg->ShortDebugString();

      activeMsg_ = std::shared_ptr<Message>(msg);
      recvMsgs_.erase(it);
      rnode->decodeMessage(activeMsg_.get());
      return true;
    }
  }

  // sleep until received a new message or another message marked as finished.
  VLOG(1) << obj_.id() << ": pick nothing buffer size " << recvMsgs_.size();
  dagCond_.wait(lk);
  return false;
}

void Executor::processActiveMsg() {
  // customer will need to handle the message & executor will do
  // post-processing.
  bool req = activeMsg_->task_.request();
  int ts = activeMsg_->task_.time();
  if (req) {
    lastRequest_ = activeMsg_;
    obj_.processRequest(activeMsg_.get());
    if (activeMsg_->finished_) {
      // if the message is marked as finished, the set the mark in corresponding
      // request tracker. otherwise, customer need to call `finishRecvReq`
      // manually.
      finishRecvReq(ts, activeMsg_->sender_);
      // reply an empty ack response message.
      if (!activeMsg_->replied_) {
        obj_.reply(activeMsg_.get());
      }
    }
  } else {
    lastResponse_ = activeMsg_;
    obj_.processResponse(activeMsg_.get());
    std::unique_lock<std::mutex> lk(nodeMu_);
    // mark finished message.
    auto rnode = getRNode(activeMsg_->sender_);
    rnode->sentReqTracker_.finish(ts);

    // check if the callback is ready to run.
    auto it = sentReqs_.find(ts);
    CHECK(it != sentReqs_.end());
    const NodeID &origRecver = it->second.recver_;
    if (origRecver != activeMsg_->sender_) {
      auto onode = getRNode(origRecver);
      // the original receiver is group node, check get response from each alive
      // node.
      if (onode->node_.role() == Node::GROUP) {
        for (auto &r : onode->group_) {
          if (r->alive_ && !r->sentReqTracker_.isFinished(ts)) {
            return;
          }
        }
        onode->sentReqTracker_.finish(ts);
      }
    }

    lk.unlock();

    if (it->second.callback_) {
      it->second.callback_();
      it->second.callback_ = Message::Callback();
    }

    sentReqCond_.notify_all();
  }
}

bool Executor::checkFinished(RemoteNode *rnode, int timestamp, bool sent) {
  CHECK(rnode);
  if (timestamp < 0) {
    return true;
  }
  auto &tracker = sent ? rnode->sentReqTracker_ : rnode->recvReqTracker_;
  if (!rnode->alive_ || tracker.isFinished(timestamp)) {
    return true;
  }
  if (rnode->node_.role() == Node::GROUP) {
    for (auto r : rnode->group_) {
      auto &rTracker = sent ? r->sentReqTracker_ : r->recvReqTracker_;
      if (r->alive_ && !rTracker.isFinished(timestamp)) {
        return false;
      }
      rTracker.finish(timestamp);
    }
    return true;
  }
  return false;
}

int Executor::numFinished(RemoteNode *rnode, int timestamp, bool send) {
  CHECK(rnode);
  if (timestamp < 0 || !rnode->alive_) {
    return 0;
  }

  auto &tracker = send ? rnode->sentReqTracker_ : rnode->recvReqTracker_;
  if (rnode->node_.role() == Node::GROUP) {
    int fin = 0;
    for (auto r : rnode->group_) {
      auto &indvTracker = send ? r->sentReqTracker_ : r->recvReqTracker_;
      if (r->alive_ && indvTracker.isFinished(timestamp)) {
        ++fin;
      }
    }

    return fin;
  } else {
    return tracker.isFinished(timestamp);
  }
}

void Executor::waitSentReq(int timestamp) {
  std::unique_lock<std::mutex> lk(nodeMu_);
  VLOG(1) << obj_.id() << ": wait sent request " << timestamp;
  const NodeID &recver = sentReqs_[timestamp].recver_;
  CHECK(recver.size());
  auto rnode = getRNode(recver);
  sentReqCond_.wait(lk, [this, rnode, timestamp]() {
    return checkFinished(rnode, timestamp, true);
  });
}

void Executor::waitRecvReq(int timestamp, const NodeID &sender) {
  std::unique_lock<std::mutex> lk(nodeMu_);
  VLOG(1) << obj_.id() << ": wait request " << timestamp << " from " << sender;
  auto rnode = getRNode(sender);
  recvReqCond_.wait(lk, [this, rnode, timestamp]() {
    return checkFinished(rnode, timestamp, false);
  });
}

int Executor::queryRecvReq(int timestamp, const mltools::NodeID &sender) {
  Lock l(nodeMu_);
  return numFinished(getRNode(sender), timestamp, false);
}

void Executor::finishRecvReq(int timestamp, const mltools::NodeID &sender) {
  std::unique_lock<std::mutex> lk(nodeMu_);
  VLOG(1) << obj_.id() << ": finish request " << timestamp << " from "
          << sender;
  auto rnode = getRNode(sender);
  rnode->recvReqTracker_.finish(timestamp);
  if (rnode->node_.role() == Node::GROUP) {
    for (auto r : rnode->group_) {
      r->recvReqTracker_.finish(timestamp);
    }
  }
  lk.unlock();
  recvReqCond_.notify_all();
  { Lock lk(msgMu_); }
  dagCond_.notify_all();
}

int Executor::submit(mltools::Message *msg) {
  CHECK(msg);
  CHECK(msg->recver_.size());
  Lock l(nodeMu_);
  int ts = msg->task_.has_time() ? msg->task_.time() : (time_ + 1);
  msg->task_.set_time(ts);
  msg->task_.set_request(true);
  msg->task_.set_customer_id(obj_.id());
  time_ = ts;
  auto &reqInfo = sentReqs_[ts];
  reqInfo.recver_ = msg->recver_;
  if (msg->callback) {
    reqInfo.callback_ = msg->callback;
  }

  // slice *msg*
  RemoteNode *rnode = getRNode(msg->recver_);
  std::vector<Message *> msgs(rnode->keys_.size());
  for (auto &m : msgs) {
    m = new Message(msg->task_);
  }
  obj_.slice(*msg, rnode->keys_, &msgs);
  CHECK_EQ(msgs.size(), rnode->group_.size());

  // sent out message one by one
  for (int i = 0; i < msgs.size(); ++i) {
    RemoteNode *r = CHECK_NOTNULL(rnode->group_[i]);
    Message *m = msgs[i];
    if (!m->valid_) {
      r->sentReqTracker_.finish(ts);
      continue;
    }
    r->encodeMessage(m);
    m->recver_ = r->node_.id();
    sys_.Queue(m);
  }

  return ts;
}

void Executor::reply(Message *request, Message *response) {
  const auto &req = CHECK_NOTNULL(request)->task_;
  if (!req.request()) {
    return;
  }

  auto &res = CHECK_NOTNULL(response)->task_;
  res.set_request(false);
  if (req.has_control()) {
    res.set_control(req.control());
  }
  if (req.has_customer_id()) {
    res.set_customer_id(req.customer_id());
  }
  res.set_time(req.time());
  response->recver_ = request->sender_;
  nodeMu_.lock();
  getRNode(response->recver_)->encodeMessage(response);
  nodeMu_.unlock();
  sys_.Queue(response);

  request->replied_ = true;
}

void Executor::accept(mltools::Message *msg) {
  {
    Lock lk(msgMu_);
    recvMsgs_.push_back(msg);
  }
  dagCond_.notify_one();
}

void Executor::replaceNode(const mltools::Node &oldNode,
                           const mltools::Node &newNode) {}

void Executor::removeNode(const mltools::Node &node) {
  VLOG(1) << obj_.id() << " remove node : " << node.ShortDebugString();
  auto id = node.id();
  if (remoteNode_.find(id) == remoteNode_.end()) {
    return;
  }
  auto r = getRNode(id);
  for (const NodeID &gid : groupIDs()) {
    remoteNode_[gid].removeGroupNode(r);
  }
  r->alive_ = false;
}
} // namespace mltools
