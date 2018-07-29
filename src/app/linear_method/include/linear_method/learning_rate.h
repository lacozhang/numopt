/*
 * =====================================================================================
 *
 *       Filename:  learning_rate.h
 *
 *    Description:  learning rate schema
 *
 *        Version:  1.0
 *        Created:  07/29/2018 10:02:12
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "proto/linear.pb.h"

namespace mltools {
namespace linear {

template <typename V> class LearningRate {
public:
  LearningRate(const LearningRateConfig &conf) : conf_(conf) {
    CHECK_GT(alpha(), 0);
    CHECK_GE(beta(), 0);
  }

  ~LearningRate() {}

  V eval(V x = 0) const {
    if (conf_.type() == LearningRateConfig::CONSTANT) {
      return alpha();
    } else {
      return alpha() / (x + beta());
    }
  }

  V alpha() const { return conf_.alpha(); }

  V beta() const { return conf_.beta(); }

private:
  LearningRateConfig conf_;
};
} // namespace linear
} // namespace mltools
