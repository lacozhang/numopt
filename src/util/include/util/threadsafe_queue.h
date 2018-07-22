#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T> class ThreadSafeQueue {
public:
  ThreadSafeQueue() {}

  void push(T &val) {
    std::lock_guard<std::mutex> lk(mut_);
    dat_.push(std::move(val));
    cond_.notify_all();
  }

  bool try_pop(T &val) {
    std::lock_guard<std::mutex> lk(mut_);
    if (dat_.empty())
      return false;
    val = std::move(dat_.front());
    dat_.pop();
    return true;
  }

  void pop(T &val) {
    std::unique_lock<std::mutex> lk(mut_);
    cond_.wait(lk, [this] { return !dat_.empty(); });
    val = std::move(dat_.front());
    dat_.pop();
  }

  int size() {
    std::lock_guard<std::mutex> lk(mut_);
    return dat_.size();
  }

  bool empty() { return size() == 0; }

private:
  mutable std::mutex mut_;
  std::condition_variable cond_;
  std::queue<T> dat_;
};
