/*
 * =====================================================================================
 *
 *       Filename:  threadpool.h
 *
 *    Description:  Using thread to finish lots of works
 *
 *        Version:  1.0
 *        Created:  07/01/2018 20:46:29
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once

#include <functional>
#include <list>
#include <string>
#include <vector>

#include "util/barrier.h"
#include "util/macro.h"
#include <condition_variable>
#include <mutex>
#include <thread>

#ifndef __THREAD_POOL_H__
#define __THREAD_POOL_H__

namespace mltools {
class ThreadPool {
public:
  explicit ThreadPool(int numWorkers)
      : numWorkers_(numWorkers), barrier_(numWorkers + 1) {}
  ~ThreadPool();

  typedef std::function<void()> Task;
  void add(const Task &task);

  void startWorkers();

  Task getNextTask();
  void stopOnBarrier() { barrier_.Block(); }

private:
  DISALLOW_COPY_AND_ASSIGN(ThreadPool);

  const int numWorkers_;
  std::list<Task> tasks_;
  std::mutex mu_;
  std::condition_variable cv_;
  Barrier barrier_;
  std::vector<std::thread> workers_;
  bool waitToFinish_, started_;
};
} // namespace mltools

#endif // __THREAD_POOL_H__
