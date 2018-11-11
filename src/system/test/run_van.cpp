/*
 * =====================================================================================
 *
 *       Filename:  van_test.cpp
 *
 *    Description:  simulation program for Van
 *
 *        Version:  1.0
 *        Created:  10/28/2018 09:49:30
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "run_van_util.h"
#include <glog/logging.h>
using namespace mltools;

int main(int argc, const char *argv[]) {
  auto schedulerVan = createSchedulerVan();
  auto workerNode = getNodeFromType(Node::WORKER);
  schedulerVan->connect(workerNode);

  size_t recvBytes = 0, sendBytes = 0;
  Message *recvMsg = new Message();
  schedulerVan->recv(recvMsg, &recvBytes);
  LOG(INFO) << "Get the new message from worker " << recvMsg->sender_
            << " of size " << recvBytes;
  printMessage(recvMsg);

  Message *sendMsg = new Message();
  fakeMessage(sendMsg, schedulerVan->scheduler().id(), workerNode.id(),
              "This is ack from Scheduler");
  schedulerVan->send(sendMsg, &sendBytes);
  LOG(INFO) << "Scheduler send the message back with size " << sendBytes;
  return 0;
}
