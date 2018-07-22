/*
 * =====================================================================================
 *
 *       Filename:  postoffice.cpp
 *
 *    Description:  implementation of postoffice.h
 *
 *        Version:  1.0
 *        Created:  07/15/2018 20:10:19
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "system/postoffice.h"
#include "system/customer.h"
#include "util/file.h"

namespace mltools {
  DEFINE_int32(report_interval, 0, "set to 0 means disabled");
  
  DECLARE_string(interface);
  
  PostOffice::PostOffice() {}
  
  PostOffice::~PostOffice() {
    if(recvThread_) {
      recvThread_->join();
    }
    if(sendThread_) {
      Message *msg = new Message();
      msg->terminate_ = true;
      Queue(msg);
      sendThread_->join();
    }
  }
  
  void PostOffice::Run(int *argc, char ***argv) {
    google::InitGoogleLogging((*argv)[0]);
    google::ParseCommandLineFlags(argc, argv, true);
    
    manager_.init((*argv)[0]);
    if(FLAGS_report_interval > 0) {
      perfMonitor_.init(FLAGS_interface, manager_.van().myNode().hostname());
    }
    
    recvThread_ = std::unique_ptr<std::thread>(new std::thread(&PostOffice::recv, this));
    sendThread_ = std::unique_ptr<std::thread>(new std::thread(&PostOffice::send, this));
    
    manager_.run();
  }
  
  void PostOffice::send() {
    Message *msg = nullptr;
    while(true) {
      sendingQueue_.pop(msg);
      if(msg->terminate_) {
        break;
      }
      size_t sendBytes = 0;
      manager_.van().send(msg, &sendBytes);
      if(FLAGS_report_interval > 0) {
        perfMonitor_.incOutBytes(sendBytes);
      }
      if(msg->task_.request()) {
        manager_.addRequest(msg);
      } else {
        delete msg;
      }
    }
  }
  
  void PostOffice::recv() {
    while(true) {
      Message *msg = new Message();
      size_t recvBytes = 0;
      CHECK(manager_.van().recv(msg, &recvBytes));
      if(FLAGS_report_interval > 0) {
        perfMonitor_.incInBytes(recvBytes);
      }
      
      if(msg->task_.task_size()) {
        CHECK(!msg->has_data());
        for(int i=0; i<msg->task_.task_size(); ++i) {
          Message *unpackMsg = new Message();
          unpackMsg->sender_ = msg->sender_;
          unpackMsg->recver_ = msg->recver_;
          unpackMsg->task_ = msg->task_.task(i);
          if(!process(unpackMsg)) {
            break;
          }
        }
        delete msg;
      } else {
        if(!process(msg)) {
          break;
        }
      }
    }
  }
  
  bool PostOffice::process(mltools::Message *msg) {
    if(!msg->task_.request()) {
      manager_.addResponse(msg);
    }
    if(msg->task_.has_control()) {
      bool ret = manager_.process(msg);
      delete  msg;
      return ret;
    } else {
      int id = msg->task_.customer_id();
      manager_.customer(id)->executor()->accept(msg);
    }
    
    return true;
  }
  
  void PostOffice::Queue(mltools::Message *msg) {
    if(!msg->task_.has_more()) {
      sendingQueue_.push(msg);
    } else {
      CHECK(msg->task_.request());
      CHECK(msg->task_.has_customer_id());
      CHECK(!msg->has_data()) << " don't know how to pack";
      Lock lk(packMu_);
      auto key = std::make_pair(msg->recver_, msg->task_.customer_id());
      auto &value = pack_[key];
      value.push_back(msg);
      
      if(!msg->task_.more()) {
        Message *packMsg = new Message();
        packMsg->recver_ = msg->recver_;
        for(auto m: value) {
          m->task_.clear_more();
          *(packMsg->task_.add_task()) = m->task_;
          delete m;
        }
        value.clear();
        sendingQueue_.push(packMsg);
      }
    }
  }
}
