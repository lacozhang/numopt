#pragma once
#include <condition_variable>
#include <mutex>

#ifndef __BARRIER_H__
#define __BARRIER_H__

namespace mltools {
class Barrier {
public:
  Barrier(int numthreads) { num_exits_ = num_block_ = numthreads; }

  bool Block() {
    std::unique_lock<std::mutex> lk(mut_);
    --num_block_;
    if (num_block_ > 0) {
      cond_.wait(lk, [this] { return this->num_block_ == 0; });
    } else {
      cond_.notify_all();
    }

    --num_exits_;
    if (num_exits_ == 0)
      return true;
    else
      return false;
  }

private:
  int num_exits_, num_block_;
  std::mutex mut_;
  std::condition_variable cond_;
};
} // namespace mltools

#endif // !__BARRIER_H__
