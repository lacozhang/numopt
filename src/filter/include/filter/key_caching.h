/*
 * =====================================================================================
 *
 *       Filename:  key_caching.h
 *
 *    Description:  key caching
 *
 *        Version:  1.0
 *        Created:  07/22/2018 15:54:17
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "filter/filter.h"
#include "util/crc32c.h"

namespace mltools {

class KeyCachingFilter : public Filter {
public:
  void encode(Message *msg) {
    auto conf = find(FilterConfig::KEY_CACHING, msg);
    if (!conf) {
      return;
    }
    if (!msg->has_key()) {
      conf->clear_signature();
      return;
    }
    const auto &key = msg->key_;
    auto sig = crc32c::Value(key.data(), std::min(key.size(), maxSigLen_));
    conf->set_signature(sig);
    auto cacheK = std::make_pair(msg->task_.key_channel(),
                                 Range<Key>(msg->task_.key_range()));
    Lock lk(mu_);
    auto &cacheV = cache_[cacheK];
    bool hit = (cacheV.first == sig) && (cacheV.second.size() == key.size());
    if (hit) {
      msg->clear_key();
    } else {
      cacheV.first = sig;
      cacheV.second = key;
    }
    if (conf->clear_cache_if_done() && isDone(msg->task_)) {
      cache_.erase(cacheK);
    }
  }

private:
  bool isDone(const Task &task) {
    return (!task.request() || (task.has_param() && task.param().push()));
  }

  std::unordered_map<std::pair<int, Range<Key>>,
                     std::pair<uint32_t, DArray<char>>>
      cache_;

  // calculate the signature using the first maxSigLen_*4 bytes to accelerate
  // the computation.
  const size_t maxSigLen_ = 2048;
  std::mutex mu_;
};
} // namespace mltools
