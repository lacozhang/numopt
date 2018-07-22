/*
 * =====================================================================================
 *
 *       Filename:  postoffice.h
 *
 *    Description:  Main interface for whole system
 *
 *        Version:  1.0
 *        Created:  07/15/2018 20:09:49
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "system/heartbeat_info.h"
#include "system/manager.h"
#include "system/message.h"
#include "util/common.h"
#include "util/macro.h"
#include "util/threadsafe_queue.h"

namespace mltools {
/**
 * @brief The interface to message delivery system
 *
 */
class PostOffice {
public:
  SINGLETON(PostOffice);
  ~PostOffice();

  /**
   * @brief Starts the system, entry point of system.
   */
  void Run(int *argc, char ***);

  /**
   * @brief stops the system
   */
  void Stop() { manager_.stop(); }

  /**
   * @brief deliver the message for delivery to other nodes. There is one
   * dedicated thread for this task.
   */
  void Queue(Message *msg);

  Manager &manager() { return manager_; }
  HeartbeatInfo &perf() { return perfMonitor_; }

private:
  PostOffice();
  void send();
  void recv();
  bool process(Message *msg);

  std::unique_ptr<std::thread> recvThread_;
  std::unique_ptr<std::thread> sendThread_;
  ThreadSafeQueue<Message *> sendingQueue_;

  Manager manager_;
  HeartbeatInfo perfMonitor_;

  // key: <sender, customer_id>, value: messages will be packed
  std::map<std::pair<NodeID, int>, std::vector<Message *>> pack_;
  std::mutex packMu_;

  DISALLOW_COPY_AND_ASSIGN(PostOffice);
};
} // namespace mltools
