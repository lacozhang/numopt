#pragma once

#include "lossbase.h"
#include "nnmacros.h"
#include "util/typedef.h"

#ifndef __CROSS_ENTROPY_H__
#define __CROSS_ENTROPY_H__

namespace NNModel {

class CrossEntropyLoss
    : public LossLayerBase<RealVector, double, int, RealVector> {
public:
  CrossEntropyLoss(int labelsize) {
    labelsize_ = labelsize;
    grad_ = boost::make_shared<RealVector>(labelsize_);
    softmax_.resize(labelsize);
  }

  double FAST_EXP(double val) {
    const double EXP_A = (1048576 / 0.6931471806);
    const double EXP_C = 60801;
    const double BOUND = 700.0;
    double r = 0.0;
    val = std::min(val, BOUND);
    val = std::max(val, -BOUND);
    *(int *)((unsigned char *)&r + 4) =
        static_cast<int>(EXP_A * (val) + (1072693248 - EXP_C));
    return r;
  }

  double Forward(const RealVector &input, const int &golden);
  void BackWard(const RealVector &input, const int &golden,
                boost::shared_ptr<RealVector> &grads);

private:
  int labelsize_;
  RealVector softmax_;
};

} // namespace NNModel

#endif // !__CROSS_ENTROPY_H__
