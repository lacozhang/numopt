#pragma once

#include "nnmacros.h"
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#ifndef __LOSS_BASE_H__
#define __LOSS_BASE_H__

namespace NNModel {

template <typename InputType, typename TrueValueType, typename LossValueType,
          typename GradOutType>
class LossLayerBase {
public:
  LossLayerBase() {}
  ~LossLayerBase() {}

  LossValueType Forward(const InputType &input, const TrueValueType &golden) {
    NNForbidOperation;
  }
  void BackWard(const InputType &input, const TrueValueType &golden,
                boost::shared_ptr<GradOutType> &grads) {
    NNForbidOperation;
  }

protected:
  boost::shared_ptr<GradOutType> grad_;
};
} // namespace NNModel

#endif // !__LOSS_BASE_H__
