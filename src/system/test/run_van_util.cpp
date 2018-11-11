/*
 * =====================================================================================
 *
 *       Filename:  run_van_util.cpp
 *
 *    Description:  implementation of utility function
 *
 *        Version:  1.0
 *        Created:  10/28/2018 09:53:17
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "run_van_util.h"
#include "util/dynamic_array_impl.h"
#include <gflags/gflags.h>

namespace mltools {
DECLARE_int32(bind_to);
DECLARE_bool(local);
DECLARE_string(my_node);
DECLARE_string(scheduler);
DECLARE_int32(num_workers);
DECLARE_int32(num_servers);

namespace {
const std::string kWorkerSpecString =
    "role:WORKER,hostname:'127.0.0.1',port:10001,id:'W0'";
const std::string kServerSpecString =
    "role:SERVER,hostname:'127.0.0.1',port:10002,id:'S0'";
const std::string kSchedulerSpecString =
    "role:SCHEDULER,hostname:'127.0.0.1',port:10003,id:'H'";
} // namespace

App *App::Create(const std::string &conf) { return new App(); }

void vanInitForWorker() {
  FLAGS_bind_to = 10001;
  FLAGS_local = true;
  FLAGS_my_node = kWorkerSpecString;
  FLAGS_scheduler = kSchedulerSpecString;
  FLAGS_num_workers = 1;
  FLAGS_num_servers = 1;
}

void vanInitForServer() {
  FLAGS_bind_to = 10002;
  FLAGS_local = true;
  FLAGS_my_node = kServerSpecString;
  FLAGS_scheduler = kSchedulerSpecString;
  FLAGS_num_workers = 1;
  FLAGS_num_servers = 1;
}

void vanInitForScheduler() {
  FLAGS_bind_to = 10003;
  FLAGS_local = true;
  FLAGS_my_node = kSchedulerSpecString;
  FLAGS_scheduler = kSchedulerSpecString;
  FLAGS_num_workers = 1;
  FLAGS_num_servers = 1;
}

std::shared_ptr<Van> createVanObjFromFunc(std::function<void()> func) {
  func();
  auto vanObj = std::shared_ptr<Van>(new Van());
  vanObj->init();
  return vanObj;
}

std::shared_ptr<Van> createWorkerVan() {
  return createVanObjFromFunc(std::function<void()>(vanInitForWorker));
}

std::shared_ptr<Van> createServerVan() {
  return createVanObjFromFunc(std::function<void()>(vanInitForServer));
}

std::shared_ptr<Van> createSchedulerVan() {
  return createVanObjFromFunc(std::function<void()>(vanInitForScheduler));
}

void fakeMessage(Message *msg, const std::string &sender,
                 const std::string &recver, const std::string strMessage) {
  if (msg == nullptr) {
    return;
  }
  msg->sender_ = sender;
  msg->recver_ = recver;
  DArray<int64> keys = {1, 2, 3, 4, 5};
  msg->set_key(keys);
  msg->task_.set_msg(strMessage);
  LOG(INFO) << "task: " << msg->task_.DebugString() << " of size " << msg->task_.ByteSize();
}

Node getNodeFromType(Node::Role role) {
  std::string str = "";
  switch (role) {
  case Node::WORKER:
    str = kWorkerSpecString;
    break;
  case Node::SERVER:
    str = kServerSpecString;
    break;
  case Node::SCHEDULER:
    str = kSchedulerSpecString;
    break;
  default:
    str = "";
    break;
  }

  Node node;
  if (::google::protobuf::TextFormat::ParseFromString(str, &node)) {
    return node;
  } else {
    LOG(WARNING) << "Failed to parse string " << str;
  }
  return Node::default_instance();
}

void printMessage(Message *recvMsg) {
  Message localMsg = *recvMsg;
  DArray<int64_t> keys(localMsg.key_);
  std::string strMessage(localMsg.task_.msg());
  LOG(INFO) << "sender: " << localMsg.sender_ << "\n"
            << "receiver: " << localMsg.recver_ << "\n"
            << "has key: " << localMsg.has_key() << "\n"
            << "keys: " << dbgstr<int64_t>(keys.data(), keys.size()) << "\n"
            << "task: " << localMsg.task_.DebugString()
            << " of size " << localMsg.task_.ByteSize();
}

} // namespace mltools
