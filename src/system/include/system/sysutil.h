/*
 * =====================================================================================
 *
 *       Filename:  sysutil.h
 *
 *    Description:  utility function for system related inferface
 *
 *        Version:  1.0
 *        Created:  07/22/2018 15:08:51
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "system/customer.h"

namespace mltools {
inline int NextCustomerID() {
  return PostOffice::getInstance().manager().nextCustomerID();
}

inline std::string SchedulerID() {
  return PostOffice::getInstance().manager().van().scheduler().id();
}

inline Node MyNode() {
  return PostOffice::getInstance().manager().van().myNode();
}

inline Node::Role MyRole() { return MyNode().role(); }

inline bool IsWorker() { return MyRole() == Node::WORKER; }

inline bool IsServer() { return MyRole() == Node::SERVER; }

inline bool IsScheduler() { return MyRole() == Node::SCHEDULER; }

inline std::string MyNodeID() { return MyNode().id(); }

inline int MyRank() { return MyNode().rank(); }

inline Range<Key> MyKeyRange() { return Range<Key>(MyNode().key()); }

inline void StartSystem(int argc, char *argv[]) {
  PostOffice::getInstance().Run(&argc, &argv);
}

inline void StopSystem() { PostOffice::getInstance().Stop(); }

inline int RunSystem(int argc, char *argv[]) {
  StartSystem(argc, argv);
  StopSystem();
  return 0;
}
} // namespace mltools
