/*
 * =====================================================================================
 *
 *       Filename:  countmin.h
 *
 *    Description:  count-min filter
 *
 *        Version:  1.0
 *        Created:  07/22/2018 17:32:47
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "util/dynamic_array_impl.h"
#include "util/sketch.h"
#include <math.h>

namespace mltools {

/// @brief use data array of size *n_* to store counting probabilistically.
template <typename K, typename V> class CountMin : public Sketch {
public:
  bool empty() const { return n_ == 0; }

  void clear() {
    data_.clear();
    n_ = 0;
  }

  void resize(int n, int k, V vmax) {
    n_ = std::max(n, 64);
    data_.resize(n_);
    data_.setZero();
    k_ = std::min(30, std::max(1, k));
    vmax_ = vmax;
  }

  void insert(const K &key, const V &count) {
    uint32 h = hash(key);
    const uint32 delta = (h >> 17) | (h << 15);
    // store the count information in k_ locations by addition of count.
    for (int j = 0; j < k_; ++j) {
      V v = data_[h % n_];
      // store the max count.
      data_[h % n_] = count > (vmax_ - v) ? vmax_ : (v + count);
      h += delta;
    }
  }

  V query(const K &key) const {
    V res = vmax_;
    uint32 h = hash(key);
    const uint32 delta = (h >> 17) | (h << 15);
    for (int j = 0; j < k_; ++j) {
      res = std::min(res, data_[h % n_]);
      h += delta;
    }
    return res;
  }

private:
  DArray<V> data_;
  int n_ = 0;
  int k_ = 1;
  V vmax_ = 0;
};
} // namespace mltools
