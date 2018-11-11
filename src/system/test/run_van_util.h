/*
 * =====================================================================================
 *
 *       Filename:  run_van_util.h
 *
 *    Description:  some utility function for testing van
 *
 *        Version:  1.0
 *        Created:  10/28/2018 09:52:55
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#include "system/customer.h"
#include "system/message.h"
#include "system/van.h"

#include <string>
#pragma once
namespace mltools {
// follwing functions are faked for the testing purpose.
void vanInitForWorker();

void vanInitForServer();

void vanInitForScheduler();

std::shared_ptr<Van> createVanObjFromFunc(std::function<void()> func);

std::shared_ptr<Van> createWorkerVan();

std::shared_ptr<Van> createServerVan();

std::shared_ptr<Van> createSchedulerVan();

void fakeMessage(Message *msg, const std::string &sender,
                 const std::string &recver, const std::string strMessage);

void printMessage(Message *recvMsg);

Node getNodeFromType(Node::Role role);
} // namespace mltools
