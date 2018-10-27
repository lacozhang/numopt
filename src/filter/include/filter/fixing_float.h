/*
 * =====================================================================================
 *
 *       Filename:  fixing_float.h
 *
 *    Description:  quantization of floating point numbers.
 *
 *        Version:  1.0
 *        Created:  07/22/2018 15:53:18
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
#include <time.h>

namespace mltools {
class FixingFloatFilter : public Filter {
public:
  void encode(Message *msg) { convert(msg, true); }

  void decode(Message *msg) { convert(msg, false); }

private:
  /// @brief a fast random function
  static bool boolrand(int *seed) {
    *seed = (214013 * *seed + 2531011);
    return ((*seed >> 16) & 0x1) == 0;
  }

  template <typename V>
  DArray<char> convert(const DArray<char> &array, bool encode, int nbytes,
                       FilterConfig::FixedFloatConfig *conf) {
    CHECK_GT(nbytes, 0);
    CHECK_LT(nbytes, 8);
    double ratio = static_cast<double>(1 << (nbytes * 8)) - 2;

    if (encode) {
      if (!conf->has_min_value()) {
        conf->set_min_value(DArray<V>(array).eigenArray().minCoeff());
      }
      if (!conf->has_max_value()) {
        conf->set_max_value(DArray<V>(array).eigenArray().maxCoeff() + 1e-6);
      }
    }

    CHECK(conf->has_min_value());
    double minV = static_cast<double>(conf->min_value());
    CHECK(conf->has_max_value());
    double maxV = static_cast<double>(conf->max_value());
    double bin = maxV - minV;
    CHECK_GT(bin, 0);

    if (encode) {
      // float/double to nbytes * 8 int
      DArray<V> orig(array);
      DArray<uint8> code(orig.size() * nbytes);
      uint8 *codePtr = code.data();
      int seed = time(NULL);
      for (int i = 0; i < orig.size(); ++i) {
        double proj = orig[i] > maxV ? maxV : orig[i] < minV ? minV : orig[i];
        double tmp = ((proj - minV) / bin * 1.0) * ratio;
        uint64 r = static_cast<uint64>(floor(tmp) + boolrand(&seed));
        for (int j = 0; j < nbytes; ++j) {
          *(codePtr++) = static_cast<uint8>(r & 0xFF);
          r = r >> 8;
        }
      }
      return DArray<char>(code);
    } else {
      // nbytes*8 int to float/double
      uint8 *codePtr = reinterpret_cast<uint8 *>(array.data());
      DArray<V> orig(array.size() / nbytes);
      for (int i = 0; i < orig.size(); ++i) {
        double r = 0;
        for (int j = 0; j < nbytes; ++j) {
          r += static_cast<uint64>(*(codePtr++)) << 8 * j;
        }
        orig[i] = static_cast<V>(r / ratio * bin + minV);
      }
      return DArray<char>(orig);
    }
  }

  void convert(Message *msg, bool encode) {
    auto filterConf = CHECK_NOTNULL(find(FilterConfig::FIXING_FLOAT, msg));
    if (filterConf->num_bytes() == 0) {
      return;
    }
    int n = msg->value_.size();
    CHECK_EQ(n, msg->task_.value_type_size());
    int k = 0;
    for (int i = 0; i < n; ++i) {
      if (msg->value_[i].size() == 0) {
        continue;
      }
      auto type = msg->task_.value_type(i);
      if (filterConf->fixed_point_size() <= k) {
        filterConf->add_fixed_point();
      }

      if (type == DataType::FLOAT) {
        msg->value_[i] =
            convert<float>(msg->value_[i], encode, filterConf->num_bytes(),
                           filterConf->mutable_fixed_point(k++));
      }
      if (type == DataType::DOUBLE) {
        msg->value_[i] =
            convert<double>(msg->value_[i], encode, filterConf->num_bytes(),
                            filterConf->mutable_fixed_point(k++));
      }
    }
  }
};
} // namespace mltools
