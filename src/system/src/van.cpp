/*
 * =====================================================================================
 *
 *       Filename:  van.cpp
 *
 *    Description:  implementation of van.h
 *
 *        Version:  1.0
 *        Created:  07/15/2018 10:28:29
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "system/van.h"
#include "system/manager.h"
#include "system/postoffice.h"
#include "util/dynamic_array_impl.h"
#include <libgen.h>
#include <string.h>
#include <zmq.h>

namespace mltools {
DEFINE_int32(bind_to, 0, "binding port");
DEFINE_bool(local, false, "run in local");

DECLARE_string(my_node);
DECLARE_string(scheduler);
DECLARE_int32(num_workers);
DECLARE_int32(num_servers);

Van::~Van() {
  statistics();
  for (auto &it : senders_) {
    zmq_close(it.second);
  }
  zmq_close(receiver_);
  zmq_ctx_destroy(context_);
}

Node Van::parseNode(const std::string &nodeStr) {
  Node node;
  CHECK(google::protobuf::TextFormat::ParseFromString(nodeStr, &node));
  if (!node.has_id()) {
    std::string str = node.hostname() + ":" + std::to_string(node.port());
    if (node.role() == Node::SCHEDULER) {
      str = "H";
    } else if (node.role() == Node::WORKER) {
      str = "W_" + str;
    } else if (node.role() == Node::SERVER) {
      str = "S_" + str;
    }
    node.set_id(str);
  }
  return node;
}

void Van::bind() {
  receiver_ = zmq_socket(context_, ZMQ_ROUTER);
  CHECK(receiver_ != nullptr)
      << "create receiver socket failed : " << zmq_strerror(errno);
  std::string addr = "tcp://*:";
  if (FLAGS_bind_to) {
    addr += std::to_string(FLAGS_bind_to);
  } else {
    CHECK(myNode_.has_port()) << myNode_.ShortDebugString();
    addr += std::to_string(myNode_.port());
  }

  if (FLAGS_local) {
    addr = "ipc:///tmp/" + myNode_.id();
  }
  CHECK(zmq_bind(receiver_, addr.c_str()) == 0)
      << "bind to " << addr << " failed:" << zmq_strerror(errno);
  VLOG(1) << "Bind to " << addr;
}

bool Van::connect(const Node &node) {
  CHECK(node.has_id()) << node.ShortDebugString();
  CHECK(node.has_port()) << node.ShortDebugString();
  CHECK(node.has_hostname()) << node.ShortDebugString();

  NodeID id = node.id();
  // If the node.id is the same as myNode_.id, then we need to update current
  // node information. This new node information generally comes from scheduler.
  if (id == myNode_.id()) {
    myNode_ = node;
  }
  if (senders_.find(id) != senders_.end()) {
    return true;
  }
  void *sender = zmq_socket(context_, ZMQ_DEALER);
  CHECK(sender != nullptr) << zmq_strerror(errno);
  std::string myId = myNode_.id();
  zmq_setsockopt(sender, ZMQ_IDENTITY, myId.data(), myId.size());

  std::string addr =
      "tcp://" + node.hostname() + ":" + std::to_string(node.port());
  if (FLAGS_local) {
    addr = "ipc:///tmp/" + node.id();
  }
  if (zmq_connect(sender, addr.c_str()) != 0) {
    LOG(WARNING) << "connect to " + addr + " failed: " + zmq_strerror(errno);
    return false;
  }

  senders_[id] = sender;
  hostnames_[id] = node.hostname();
  VLOG(1) << "Connect to " << id << " [" << addr << "]";
  return true;
}

void Van::monitor() {
  VLOG(1) << "starting monitoring...";
  void *s = CHECK_NOTNULL(zmq_socket(context_, ZMQ_PAIR));
  CHECK(!zmq_connect(s, "inproc://monitor"));
  while (true) {
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    if (zmq_msg_recv(&msg, s, 0) == -1) {
      if (errno == EINTR)
        continue;
      break;
    }
    uint8_t *data = (uint8_t *)zmq_msg_data(&msg);
    int event = *(reinterpret_cast<uint16_t *>(data));
    int value = *(reinterpret_cast<uint32_t *>(data + 2));

    if (event == ZMQ_EVENT_DISCONNECTED) {
      auto &manager = PostOffice::getInstance().manager();
      if(isScheduler()) {
        Lock l(fdToNodeIdMu_);
        if(fdToNodeId_.find(value) == fdToNodeId_.end()){
          LOG(WARNING) << "cannot find the node id for Fd = " << value;
          continue;
        }
        manager.nodeDisconnected(fdToNodeId_[value]);
      } else {
        manager.nodeDisconnected(scheduler_.id());
      }
    }
    
    if(event == ZMQ_EVENT_MONITOR_STOPPED) {
      break;
    }
  }
  zmq_close(s);
  VLOG(1) << "monitor stopped";
}

void Van::init() {
  scheduler_ = parseNode(FLAGS_scheduler);
  myNode_ = parseNode(FLAGS_my_node);
  LOG(INFO) << "I'm [" << myNode_.DebugString() << "]";

  context_ = zmq_ctx_new();
  CHECK(context_ != nullptr) << "Create 0mq context failed";

  zmq_ctx_set(context_, ZMQ_MAX_SOCKETS, 65536);
  bind();
  connect(scheduler_);
  
  if(isScheduler()) {
    CHECK(!zmq_socket_monitor(receiver_, "inproc://monitor", ZMQ_EVENT_ALL));
  } else {
    CHECK(!zmq_socket_monitor(senders_[scheduler_.id()], "inproc://monitor", ZMQ_EVENT_ALL));
  }
  monitorThread_ = new std::thread(&Van::monitor, this);
  monitorThread_->detach();
}

void Van::disconnect(const Node &node) {
  CHECK(node.has_id()) << node.ShortDebugString();
  NodeID id = node.id();
  if (senders_.find(id) != senders_.end()) {
    zmq_close(senders_[id]);
  }
  senders_.erase(id);
  VLOG(1) << "Disconnect from " << node.id();
}
} // namespace mltools
