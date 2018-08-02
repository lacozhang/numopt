/*
 * =====================================================================================
 *
 *       Filename:  filter.h
 *
 *    Description:  interface defined for all filters
 *
 *        Version:  1.0
 *        Created:  07/19/2018 08:55:25
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "proto/filter.pb.h"
#include "system/message.h"
#include "util/dynamic_array.h"

namespace mltools {
/**
 * @brief interface for all filters
 */
class Filter {
public:
  Filter() {}
  virtual ~Filter() {}

  static Filter *create(const FilterConfig &conf);

  virtual void encode(Message *msg) {}
  virtual void decode(Message *msg) {}

  static FilterConfig *find(FilterConfig::Type type, Message *msg) {
    return find(type, &(msg->task_));
  }
  static FilterConfig *find(FilterConfig::Type type, Task *task);
};
} // namespace mltools
