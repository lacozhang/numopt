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

inline std::string MyNodeID() { return MyNode().id(); }
  
inline Range<Key> MyKeyRange() { return Range<Key>(MyNode().key()); }
} // namespace mltools
