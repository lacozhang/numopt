#pragma once

#include <boost/log/trivial.hpp>
#include <condition_variable>
#include <glog/logging.h>
#include <mutex>
#include <queue>

template <typename T> class ThreadSafeLimitedQueue {
public:
  ThreadSafeLimitedQueue() {}
  explicit ThreadSafeLimitedQueue(int capacity) {
    setCapacity(capacity);
    size_ = 0;
  }

  void push(const T &val, size_t size, bool finished = false) {
    CHECK(!nomore_) << "The work is done, do not call push";
    if (size > capacity_) {
      LOG(WARNING) << "object size " << size << " exceed maxium capacity "
                   << capacity_ << ", you will be blocked forever";
    }
    if (!finished && size == 0) {
      LOG(INFO) << "insert object of size 0";
      return;
    }
    std::unique_lock<std::mutex> lk(mu_);
    fullcond_.wait(lk, [this, size]() { return (size_ + size) <= capacity_; });
    dat_.push(std::move(std::make_pair(val, size)));
    size_ += size;
    nomore_ = finished;
    emptycond_.notify_all();
  }

  bool pop(T &val) {
    std::unique_lock<std::mutex> lk(mu_);
    if (nomore_ && dat_.empty()) {
      return false;
    }
    emptycond_.wait(lk, [this] { return !dat_.empty(); });
    std::pair<T, size_t> item = std::move(dat_.front());
    if (item.second == 0) {
      CHECK(false);
      return false;
    }

    val = std::move(item.first);
    size_ -= item.second;
    dat_.pop();
    fullcond_.notify_all();
    return true;
  }

  void setCapacity(int capacity) { capacity_ = capacity; }

  size_t size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return size_;
  }

  bool empty() const { return size() == 0; }

private:
  bool nomore_ = false;
  int capacity_ = 0, size_ = 0;
  std::queue<std::pair<T, size_t>> dat_;
  mutable std::mutex mu_;
  std::condition_variable emptycond_, fullcond_;
};
