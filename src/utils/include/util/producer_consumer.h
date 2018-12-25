#pragma once

#include "barrier.h"
#include "threadsafe_limited_queue.h"
#include <boost/log/trivial.hpp>
#include <functional>
#include <glog/logging.h>
#include <thread>
#include <vector>

namespace mltools {

template <typename T> class ProducerConsumer {
public:
  ProducerConsumer() : blocker_(2) {
    setCapacity(1000);
    consumers_ = 1;
  }
  ProducerConsumer(int capacity, int consumers) : blocker_(consumers + 1) {
    consumers_ = consumers;
    setCapacity(capacity);
  }

  void StartProducer(std::function<bool(T &, size_t &)> &func) {
    producer_thr_ = std::move(std::thread([this, &func]() {
      T entry;
      bool done = false;
      while (!done) {
        size_t size = 0;
        done = !func(entry, size);
        queue_.push(entry, size, done);
      }
    }));
    producer_thr_.detach();
  }

  void StartProducer(std::function<bool(T &)> &func) {
    producer_thr_ = std::move(std::thread([this, func]() {
      T val;
      bool done = false;
      while (!done) {
        done = !func(val);
        queue_.push(val, 0, done);
      }
    }));
  }

  void StartConsumer(const std::function<void(const T &)> &func) {
    consumers_thrs_.emplace_back(std::move(std::thread([this, func]() {
      T entry;
      while (pop(entry)) {
        func(entry);
      }
    })));
  }

  template <typename V>
  void StartConsumer(std::function<void(T &, V &)> &processor,
                     std::function<void(V &)> &updater) {
    auto worker = [this, &processor, &updater]() {
      T val;
      V output;
      while (queue_.pop(val)) {
        processor(val, output);
        updater(output);
      }

      BlockConsumer();
#ifdef _DEBUG
      BOOST_LOG_TRIVIAL(info) << "Thread done";
#endif // _DEBUG
    };

    consumers_thrs_.resize(consumers_);
    for (int i = 0; i < consumers_; ++i) {
      consumers_thrs_[i] = std::move(std::thread(worker));
    }
  }

  void BlockConsumer() { blocker_.Block(); }

  void JoinConsumer() {
    for (auto &item : consumers_thrs_) {
      if (item.joinable())
        item.join();
    }
  }

  void BlockProducer() {
    if (producer_thr_.joinable()) {
      producer_thr_.join();
    } else {
      LOG(ERROR) << "Producer thread not joinable";
    }
  }

  void setCapacity(int size) { queue_.setCapacity(size * 4096); }

  bool pop(T &data) { return queue_.pop(data); }

  void push(const T &entry, size_t size = 1, bool finished = false) {
    queue_.push(entry, size, finished);
  }

private:
  int consumers_;
  std::thread producer_thr_;
  std::vector<std::thread> consumers_thrs_;
  ThreadSafeLimitedQueue<T> queue_;
  mltools::Barrier blocker_;
};
} // namespace mltools
