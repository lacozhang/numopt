#pragma once

#include <boost/log/trivial.hpp>
#include <condition_variable>
#include <mutex>
#include <queue>

#ifndef __THREADSAFE_LIMITED_QUEUE_H__

template <typename T> class ThreadSafeLimitedQueue {
public:
  explicit ThreadSafeLimitedQueue(int capacity) {
    capacity_ = capacity;
    nomore_ = false;
  }

  void push(T &val, bool nomore) {
    if (nomore_)
      BOOST_LOG_TRIVIAL(error) << "Call push when done";
    std::unique_lock<std::mutex> lk(mut_);
    emptycond_.wait(lk, [this] { return dat_.size() < capacity_; });
    dat_.push(std::move(val));
    fullcond_.notify_all();
    nomore_ = nomore;
  }

  bool pop(T &val) {
    std::unique_lock<std::mutex> lk(mut_);
    if (nomore_ && dat_.empty())
      return false;
    fullcond_.wait(lk, [this] { return nomore_ || !dat_.empty(); });
    if (!dat_.empty()) {
      val = std::move(dat_.front());
      dat_.pop();
    }
    emptycond_.notify_all();
    return true;
  }

private:
  bool nomore_;
  int capacity_;
  std::queue<T> dat_;
  mutable std::mutex mut_;
  std::condition_variable emptycond_, fullcond_;
};

#endif // !__THREADSAFE_LIMITED_QUEUE_H__
