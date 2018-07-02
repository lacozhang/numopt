/*
 * =====================================================================================
 *
 *       Filename:  threadpool.cpp
 *
 *    Description:  implementation of thread pools
 *
 *        Version:  1.0
 *        Created:  07/01/2018 20:47:05
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "util/threadpool.h"

namespace mltools {
  ThreadPool::~ThreadPool(){
    if(!started_) {
      return;
    }
    
    mu_.lock();
    waitToFinish_=true;
    cv_.notify_all();
    mu_.unlock();
    
    stopOnBarrier();
    for(int i=0; i< numWorkers_; ++i) {
      workers_[i].join();
    }
  }
  
  void RunWorker(void *data) {
    ThreadPool *const ptr = reinterpret_cast<ThreadPool*>(data);
    auto task = ptr->getNextTask();
    while(task) {
      task();
      task = ptr->getNextTask();
    }
    ptr->stopOnBarrier();
  }
  
  void ThreadPool::startWorkers(){
    started_ = true;
    for(int i=0; i<numWorkers_; ++i) {
      workers_.push_back(std::move(std::thread(RunWorker, this)));
    }
  }
  
  
}
