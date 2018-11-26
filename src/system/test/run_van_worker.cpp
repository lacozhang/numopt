/*
 * =====================================================================================
 *
 *       Filename:  run_van_worker.cpp
 *
 *    Description:  worker node
 *
 *        Version:  1.0
 *        Created:  10/28/2018 10:54:49
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "run_van_util.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

using namespace mltools;

int main(int argc, const char *argv[]) {
  auto workerVan = createWorkerVan();
  size_t sendBytesWorker = 0, recvBytesWorker = 0;
  Message msg;
  fakeMessage(&msg, workerVan->myNode().id(), workerVan->scheduler().id(),
              "Hello message to scheduler");
  printMessage(&msg);
  EXPECT_TRUE(workerVan->send(&msg, &sendBytesWorker));
  LOG(INFO) << "Message will send \n"
            << msg.DebugString() << " of size " << sendBytesWorker;
  workerVan->recv(&msg, &recvBytesWorker);
  LOG(INFO) << "message from scheduler with size: " << recvBytesWorker;
  printMessage(&msg);
  return 0;
}
