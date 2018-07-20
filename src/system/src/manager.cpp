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
} // namespace mltools
