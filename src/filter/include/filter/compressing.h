/*
 * =====================================================================================
 *
 *       Filename:  compressing.h
 *
 *    Description:  compression filter
 *
 *        Version:  1.0
 *        Created:  07/22/2018 15:52:58
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

namespace mltools {
class CompressingFilter : public Filter {
public:
  void encode(Message *msg) {
    auto conf = find(FilterConfig::COMPRESSING, msg);
    if (!conf) {
      return;
    }
    conf->clear_uncompressed_size();
    if (msg->has_key()) {
      conf->add_uncompressed_size(msg->key_.size());
      msg->key_ = msg->key_.compressTo();
    }
    for (auto &v : msg->value_) {
      conf->add_uncompressed_size(v.size());
      v = v.compressTo();
    }
  }

  void decode(Message *msg) {
    auto conf = find(FilterConfig::COMPRESSING, msg);
    if (!conf) {
      return;
    }
    int hasKey = msg->has_key();
    CHECK_EQ(conf->uncompressed_size_size(), hasKey + msg->value_.size());
    if (hasKey) {
      DArray<char> raw(conf->uncompressed_size(0));
      raw.uncompressFrom(msg->key_);
      msg->key_ = raw;
    }
    for (int i = 0; i < msg->value_.size(); ++i) {
      DArray<char> raw(conf->uncompressed_size(i + hasKey));
      raw.uncompressFrom(msg->value_[i]);
      msg->value_[i] = raw;
    }
  }
};
} // namespace mltools
