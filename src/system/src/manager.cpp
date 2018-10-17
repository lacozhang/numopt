/*
 * =====================================================================================
 *
 *       Filename:  manager.cpp
 *
 *    Description:  implementation of manager.h
 *
 *        Version:  1.0
 *        Created:  07/15/2018 20:22:03
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "system/manager.h"
#include "system/customer.h"
#include "system/postoffice.h"

namespace mltools {
DECLARE_int32(num_servers);
DECLARE_int32(num_workers);
DECLARE_int32(num_replicas);
DECLARE_int32(report_interval);

DEFINE_string(app_conf, "", "the string conf of the main app");
DEFINE_string(app_file, "", "the path to the file of conf");

DEFINE_uint64(key_start, 0, "glocal key start");
DEFINE_uint64(key_end, kuint64max, "global key end");

Manager::Manager() {}

Manager::~Manager() {
  for (auto &it : customers_) {
    if (it.second.second) {
      delete it.second.first;
    }
  }

  if (nodeAssigner_ != nullptr) {
    delete nodeAssigner_;
  }

  if (app_ != nullptr) {
    delete app_;
  }
}

void Manager::init(char *argv0) {
  env_.init(argv0);
  van_.init();

  if (isScheduler()) {
    if (!FLAGS_logtostderr) {
      LOG(INFO) << "Start system. Logging to " << FLAGS_log_dir << "/"
                << basename(argv0) << ".log.*";
    }
    nodeAssigner_ = new NodeAssigner(
        FLAGS_num_servers, Range<Key>(FLAGS_key_start, FLAGS_key_end));
    if (!FLAGS_app_file.empty()) {
      CHECK(readFileToString(FLAGS_app_file, &appConf_))
          << " failed to read conf file " << FLAGS_app_file;
    }
    appConf_ += FLAGS_app_conf;
    createApp(appConf_);

    // init connection to self.
    addNode(van_.myNode());
  } else {
    // get app config from scheduler.
    Task tk = newControlTask(Control::REQUEST_APP);
    *tk.mutable_ctrl()->add_node() = van_.myNode();
    sendTask(van_.scheduler(), tk);
  }
}

void Manager::run() {
  // for non-scheduler, app_ object is created by PostOffice::recv_thread, but
  // manager is runned by main thread, need thread synchronization.
  while (!isMyNodeInited_) {
    usleep(500);
  }
  if (van_.myNode().role() == Node::WORKER) {
    waitServersReady();
  }

  VLOG(1) << "everything is ready, hand control to App object";
  CHECK_NOTNULL(app_)->run();
}

void Manager::addNode(const mltools::Node &node) {
  nodesMu_.lock();
  if (nodes_.find(node.id()) == nodes_.end()) {
    if (!isScheduler()) {
      // Scheduler already connect to each node when receive the request_app
      // van maintain the physical connection to every node.
      CHECK(van_.connect(node));
    }
    if (node.role() == Node::WORKER) {
      ++numWorkers_;
    }
    if (node.role() == Node::SERVER) {
      ++numServers_;
    }
    ++numActiveNodes_;
  }
  nodes_[node.id()] = node;
  nodesMu_.unlock();

  // let all customers notify the new connection.
  for (auto &it : customers_) {
    it.second.first->executor()->addNode(node);
  }

  if (isScheduler() && node.id() != van_.myNode().id()) {
    // send all existing node info to sender. Even include the newly added node.
    Task addNode = newControlTask(Control::ADD_NODE);
    for (auto &it : nodes_) {
      *addNode.mutable_ctrl()->add_node() = it.second;
    }
    sendTask(node, addNode);

    // broadcast new node information.
    for (auto &it : nodes_) {
      if (it.first == van_.myNode().id() || it.first == node.id()) {
        continue;
      }
      Task addNewNode = newControlTask(Control::ADD_NODE);
      *addNewNode.mutable_ctrl()->add_node() = node;
      sendTask(it.second, addNewNode);
    }
  }

  // Key point: Once get add_note command from scheduler, this variable will be set properly and start working on real work now.
  if (node.id() == van_.myNode().id()) {
    isMyNodeInited_ = true;
  }
  VLOG(1) << "add node : " << node.ShortDebugString();
}

Task Manager::newControlTask(Control::Command cmd) {
  Task task;
  task.set_control(true);
  task.set_request(true);
  task.set_time(isScheduler() ? time_ * 2 : time_ * 2 + 1);
  ++time_;
  task.mutable_ctrl()->set_cmd(cmd);
  return task;
}

void Manager::sendTask(const NodeID &recver, const Task &tk) {
  Message *msg = new Message(tk);
  msg->recver_ = recver;
  PostOffice::getInstance().Queue(msg);
}

void Manager::createApp(const std::string &conf) {
  app_ = App::Create(conf);
  CHECK(app_ != nullptr) << "failed to create app\n" << conf;
}

void Manager::waitServersReady() {
  while (numServers_ < FLAGS_num_servers) {
    usleep(500);
  }
}

void Manager::waitWorkersReady() {
  while (numWorkers_ < FLAGS_num_workers) {
    usleep(500);
  }
}

void Manager::stop() {
  if (isScheduler()) {
    while (numActiveNodes_ > 1) {
      usleep(500);
    }
    inExit_ = true;
    for (auto &it : nodes_) {
      Task tk = newControlTask(Control::EXIT);
      sendTask(it.second, tk);
    }
    usleep(1000);
    LOG(INFO) << "System stopped";
  } else {
    Task tk = newControlTask(Control::READY_TO_EXIT);
    sendTask(van_.scheduler(), tk);
    while (!done_) {
      usleep(500);
    }
  }
}

bool Manager::process(mltools::Message *msg) {
  const Task &tk = msg->task_;
  CHECK(tk.control());

  if (tk.request()) {
    Task reply;
    reply.set_control(true);
    reply.set_request(false);
    reply.set_time(tk.time());

    CHECK(tk.has_control());
    const auto &ctrl = tk.ctrl();
    switch (ctrl.cmd()) {
    case Control::REQUEST_APP: {
      CHECK(isScheduler());
      // connect the node at first
      CHECK_EQ(ctrl.node_size(), 1);
      CHECK(van_.connect(ctrl.node(0)));
      reply.mutable_ctrl()->set_cmd(Control::REQUEST_APP);
      reply.set_msg(appConf_);
      break;
    }
    case Control::REGISTER_NODE: {
      CHECK(isScheduler());
      CHECK_EQ(ctrl.node_size(), 1);
      Node sender = ctrl.node(0);
      CHECK_NOTNULL(nodeAssigner_)->assign(&sender);
      addNode(sender);
      break;
    }
    case Control::REPORT_PERF: {
      CHECK(isScheduler());
      break;
    }
    case Control::READY_TO_EXIT: {
      CHECK(isScheduler());
      --numActiveNodes_;
      break;
    }
    case Control::ADD_NODE:
    case Control::UPDATE_NODE: {
      for (int i = 0; i < ctrl.node_size(); ++i) {
        addNode(ctrl.node(i));
      }
      break;
    }
    case Control::REPLACE_NODE: {
      break;
    }
    case Control::REMOVE_NODE: {
      for (int i = 0; i < ctrl.node_size(); ++i) {
        removeNode(ctrl.node(i).id());
      }
      break;
    }
    case Control::EXIT: {
      done_ = true;
      return false;
    }
    }
    sendTask(msg->sender_, reply);
  } else {
    if (!tk.has_control()) {
      return false;
    }
    if (tk.ctrl().cmd() == Control::REQUEST_APP) {
      CHECK(tk.has_msg());
      createApp(tk.msg());
      // after the creation of app, we can enable the broadcast of this node
      // info.
      Task task = newControlTask(Control::REGISTER_NODE);
      *task.mutable_ctrl()->add_node() = van_.myNode();
      sendTask(van_.scheduler(), task);
    }
  }

  return true;
}

void Manager::removeNode(const mltools::NodeID &nodeId) {
  Node node;
  {
    Lock lk(nodesMu_);
    auto it = nodes_.find(nodeId);
    if (it == nodes_.end()) {
      return;
    }
    node = it->second;
    if (node.role() == Node::SERVER) {
      --numServers_;
    }
    if (node.role() == Node::WORKER) {
      --numWorkers_;
    }
    --numActiveNodes_;
    nodes_.erase(it);
  }
  for (auto &it : customers_) {
    it.second.first->executor()->removeNode(node);
  }

  if (isScheduler() && node.id() != van_.myNode().id()) {
    for (auto &it : nodes_) {
      if (it.first == van_.myNode().id() || it.first == node.id()) {
        continue;
      }
      Task rmNode = newControlTask(Control::REMOVE_NODE);
      *rmNode.mutable_ctrl()->add_node() = node;
      sendTask(it.second, rmNode);
    }
  }

  VLOG(1) << "remove node : " << node.ShortDebugString();
}

void Manager::nodeDisconnected(const NodeID &nodeId) {
  // unnecessary to do this.
  if (inExit_) {
    return;
  }

  for (auto &handle : nodeFailureHandlers_) {
    handle(nodeId);
  }
  if (isScheduler()) {
    LOG(INFO) << nodeId << " disconnected";
    removeNode(nodeId);
  } else {
    for (int i = 0; i < 1000; ++i) {
      usleep(1000);
      if (done_) {
        return;
      }
    }

    LOG(ERROR) << van_.myNode().id() << " disconnected to scheduler";
    string kill = "kill -9 " + std::to_string(getpid());
    system(kill.c_str());
  }
}

Customer *Manager::customer(int id) {
  auto it = customers_.find(id);
  CHECK(it != customers_.end()) << "customer " << id << " nonexists";
  return it->second.first;
}

void Manager::addCustomer(mltools::Customer *obj) {
  CHECK_EQ(customers_.count(obj->id()), 0)
      << "customer " << obj->id() << " added twice";
  customers_[obj->id()] = std::make_pair(obj, false);
  Lock lk(nodesMu_);
  for (const auto &it : nodes_) {
    obj->executor()->addNode(it.second);
  }
}

void Manager::removeCustomer(int cid) {
  auto it = customers_.find(cid);
  if (it != customers_.end()) {
    it->second.first = nullptr;
  }
}

int Manager::nextCustomerID() {
  int cid = 0;
  for (auto &it : customers_) {
    cid = std::max(cid, it.second.first->id() + 1);
  }
  return cid;
}
} // namespace mltools
