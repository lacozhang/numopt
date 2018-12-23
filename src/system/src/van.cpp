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
    LOG(INFO) << "Use port from command line";
    addr += std::to_string(FLAGS_bind_to);
  } else {
    LOG(INFO) << "Use port from node specification";
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
      if (isScheduler()) {
        Lock l(fdToNodeIdMu_);
        if (fdToNodeId_.find(value) == fdToNodeId_.end()) {
          LOG(WARNING) << "cannot find the node id for Fd = " << value;
          continue;
        }
        manager.nodeDisconnected(fdToNodeId_[value]);
      } else {
        manager.nodeDisconnected(scheduler_.id());
      }
    }

    if (event == ZMQ_EVENT_MONITOR_STOPPED) {
      break;
    }
  }
  zmq_close(s);
  VLOG(1) << "monitor stopped";
}

void Van::init() {
  scheduler_ = parseNode(FLAGS_scheduler);
  myNode_ = parseNode(FLAGS_my_node);
  LOG(INFO) << "I'm \n[" << myNode_.DebugString() << "]";

  context_ = zmq_ctx_new();
  CHECK(context_ != nullptr) << "Create 0mq context failed";

  zmq_ctx_set(context_, ZMQ_MAX_SOCKETS, 65536);
  bind();
  connect(scheduler_);

  if (isScheduler()) {
    CHECK(!zmq_socket_monitor(receiver_, "inproc://monitor", ZMQ_EVENT_ALL));
  } else {
    CHECK(!zmq_socket_monitor(senders_[scheduler_.id()], "inproc://monitor",
                              ZMQ_EVENT_ALL));
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

bool Van::send(Message *msg, size_t *sendBytes) {
  CHECK_NOTNULL(msg);
  CHECK_NOTNULL(sendBytes);

  NodeID recverId = msg->recver_;
  auto it = senders_.find(recverId);
  if (it == senders_.end()) {
    LOG(WARNING) << "There is no socket to node " << recverId;
    return false;
  }
  void *socket = it->second;

  bool has_key = !msg->key_.empty();
  if (has_key) {
    msg->task_.set_has_key(has_key);
  } else {
    msg->task_.clear_has_key();
  }
  int n = has_key + msg->value_.size();
  size_t dataSize = 0;

  size_t taskSize = msg->task_.ByteSize();
  char *taskBuff = new char[taskSize + 5];
  std::memset(taskBuff, 0, taskSize + 5);
  if (taskBuff == nullptr) {
    LOG(FATAL) << "Failed to allocate memory";
  }
  CHECK(msg->task_.SerializeToArray(taskBuff, taskSize))
      << "failed to serialize task " << msg->task_.ShortDebugString();

  int tag = ZMQ_SNDMORE;
  if (n == 0) {
    tag = 0;
  }
  zmq_msg_t taskMsg;
  zmq_msg_init_data(&taskMsg, taskBuff, taskSize, freeData, nullptr);
  while (true) {
    if (zmq_msg_send(&taskMsg, socket, tag) == taskSize) {
      break;
    }
    if (errno == EINTR) {
      continue;
    }
    LOG(WARNING) << "failed to send to node " << recverId
                 << zmq_strerror(errno);
    return false;
  }
  dataSize += taskSize;

  for (int i = 0; i < n; ++i) {
    DArray<char> *data = new DArray<char>(
        (i == 0 && has_key) ? msg->key_ : msg->value_[i - has_key]);
    if (data == nullptr) {
      LOG(ERROR) << "Failed to allocate memory";
      return false;
    }
    zmq_msg_t dataMsg;
    zmq_msg_init_data(&dataMsg, data->data(), data->size(), freeData, data);
    if (i == n - 1) {
      tag = 0;
    }
    while (true) {
      if (zmq_msg_send(&dataMsg, socket, tag) == data->size()) {
        break;
      }
      if (errno == EINTR) {
        continue;
      }
      LOG(WARNING) << "failed to send data to " << recverId
                   << " errno : " << zmq_strerror(errno);
      return false;
    }
    dataSize += data->size();
  }

  *sendBytes += dataSize;
  if (hostnames_[recverId] == myNode_.hostname()) {
    sentToLocal_ += dataSize;
  } else {
    sentToOthers_ += dataSize;
  }
  VLOG(1) << "To " << msg->recver_ << " " << msg->ShortDebugString();
  return true;
}

bool Van::recv(mltools::Message *msg, size_t *recvBytes) {
  size_t dataSize = 0;
  msg->clear_data();
  for (int i = 0;; ++i) {
    zmq_msg_t *zmsg = new zmq_msg_t;
    CHECK(zmq_msg_init(zmsg) == 0) << zmq_strerror(errno);
    while (true) {
      if (zmq_msg_recv(zmsg, receiver_, 0) != -1) {
        break;
      }
      if (errno == EINTR) {
        continue;
      }
      LOG(WARNING) << "failed to receive message, error : "
                   << zmq_strerror(errno);
      return false;
    }
    char *buf = CHECK_NOTNULL((char *)zmq_msg_data(zmsg));
    size_t size = zmq_msg_size(zmsg);
    dataSize += size;

    if (i == 0) {
      // first message is the identity of the sender
      msg->sender_ = std::string(buf, size);
      msg->recver_ = myNode_.id();
      zmq_msg_close(zmsg);
      delete zmsg;
    } else if (i == 1) {
      CHECK(msg->task_.ParseFromArray(buf, size))
          << "failed to parse string from " << msg->sender_ << ". this is "
          << myNode_.id() << size;
      if (isScheduler() && msg->task_.control() &&
          msg->task_.ctrl().cmd() == Control::REQUEST_APP) {
        // In the start time of every work, they will send such message to
        // scheduler to get the config. The scheduler will store the file
        // descriptor for performance monitoring.
        int val[64];
        size_t valLen = sizeof(val);
        std::memset(val, 0, sizeof(val));
        CHECK(!zmq_getsockopt(receiver_, ZMQ_FD, (char *)val, &valLen))
            << "Failed to get the file descriptor of " << msg->sender_
            << ". with error: " << zmq_strerror(errno);
        CHECK_EQ(valLen, 4);
        int fd = val[0];
        VLOG(1) << "node [" << msg->sender_ << "] is on file descriptor " << fd;
        Lock l(fdToNodeIdMu_);
        fdToNodeId_[fd] = msg->sender_;
      }
      zmq_msg_close(zmsg);
      delete zmsg;
    } else {
      // keys & values from sender's message
      DArray<char> data(buf, size, false);
      data.pointer().reset(buf, [zmsg](char *) {
        zmq_msg_close(zmsg);
        delete zmsg;
      });
      if (i == 2 && msg->task_.has_key()) {
        msg->key_ = data;
      } else {
        msg->value_.push_back(data);
      }
    }

    if (!zmq_msg_more(zmsg)) {
      CHECK_GT(i, 0);
      break;
    }
  }
  *recvBytes += dataSize;
  if (hostnames_[msg->sender_] == myNode_.hostname()) {
    receivedFromLocal_ += dataSize;
  } else {
    receivedFromOthers_ += dataSize;
  }
  VLOG(1) << "FROM : " << msg->sender_ << msg->ShortDebugString();
  return true;
}

void Van::statistics() {
  auto gb = [](size_t x) { return x / 1e9; };
  LOG(INFO) << myNode_.id() << "\nReceive "
            << gb(receivedFromLocal_ + receivedFromOthers_) << " (local "
            << gb(receivedFromLocal_) << " ) GBytes,"
            << "Send    " << gb(sentToLocal_ + sentToOthers_) << " (local "
            << gb(sentToLocal_) << " ) GBytes";
}
} // namespace mltools
