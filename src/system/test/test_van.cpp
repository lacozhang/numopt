/*
 * =====================================================================================
 *
 *       Filename:  test_van.cpp
 *
 *    Description:  test implementation of van
 *
 *        Version:  1.0
 *        Created:  07/15/2018 10:30:37
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#include "run_van_util.h"
#include "system/sysutil.h"
#include "system/van.h"
#include "util/dynamic_array_impl.h"

namespace mltools {

TEST(Van, initVanObj) {
  auto schedulerVan = createSchedulerVan();
  auto workerVan = createWorkerVan();
  auto serverVan = createServerVan();
}
} // namespace mltools
