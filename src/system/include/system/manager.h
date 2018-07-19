/*
 * =====================================================================================
 *
 *       Filename:  manager.h
 *
 *    Description:  The command object behind the PostOffice interface.
 *
 *        Version:  1.0
 *        Created:  07/15/2018 20:21:05
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
#include "proto/task.pb.h"
#include "system/assigner.h"
#include "system/env.h"
#include "system/van.h"
#include "util/common.h"
#include "util/macro.h"

namespace mltools {
class App;
class Customer;

class Manager {
public:
  Manager();
  ~Manager();

  void init(char *argv0);
  void run();
  void stop();
  
  /// @brief key function to distribute incoming message.
  bool process(Message *msg);

  void addNode(const Node &node);
  void removeNode(const NodeID &nodeId);

  void nodeDisconnected(const NodeID &nodeId);

  /// @brief callback function when node disconnected
  typedef std::function<void(const NodeID &)> NodeFailureHandler;

  void addNodeFailureHandler(NodeFailureHandler handler) {
    nodeFailureHandlers_.push_back(handler);
  }

  // manager customer
  Customer *customer(int id);
  void addCustomer(Customer *obj);
  void removeCustomer(int id);
  int nextCustomerID();

  // workers and servers
  void waitServersReady();
  void waitWorkersReady();

  int numWorkers() const { return numWorkers_; }
  int numServers() const { return numServers_; }

  void addRequest(Message *msg) { delete msg; }
  void addResponse(Message *msg) {}

  Van &van() { return van_; }
  App *app() { return app_; }

private:
  bool isScheduler() { return van_.myNode().role() == Node::SCHEDULER; }
  Task newControlTask(Control::Command cmd);
  void sendTask(const NodeID &recver, const Task &tk);
  void sendTask(const Node &recver, const Task &tk) {
    sendTask(recver.id(), tk);
  }

  void createApp(const std::string &conf);
  
  /// @brief runed by main thread.
  App *app_ = nullptr;
  std::string appConf_;

  /// @brief manager the nodes available
  std::map<NodeID, Node> nodes_;
  std::mutex nodesMu_;
  int numWorkers_ = 0;
  int numServers_ = 0;
  int numActiveNodes_ = 0;
  std::vector<NodeFailureHandler> nodeFailureHandlers_;
  bool isMyNodeInited_ = false;

  // only available when node's role is scheduler
  NodeAssigner *nodeAssigner_ = nullptr;

  /// format: <id, <obj_ptr, is_deletable>>
  std::map<int, std::pair<Customer *, bool>> customers_;

  bool done_ = false;
  bool inExit_ = false;
  int time_ = 0;
  Van van_;
  Env env_;

  DISALLOW_COPY_AND_ASSIGN(Manager);
};
} // namespace mltools
