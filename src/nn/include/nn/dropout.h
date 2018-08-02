#pragma once
#include "nnmodule.h"
#include <random>

#ifndef __DROP_OUT_H__
#define __DROP_OUT_H__

namespace NNModel {
class DropoutLayer
    : public NNLayerBase<RealVector, RealVector, RealVector, RealVector> {
public:
  typedef NNLayerBase<RealVector, RealVector, RealVector, RealVector> BaseType;

  DropoutLayer(double rate, int size);

  void Forward(const RealVector &input, boost::shared_ptr<RealVector> &output);
  void Backward(const RealVector &input,
                const boost::shared_ptr<RealVector> &gradin,
                boost::shared_ptr<RealVector> &gradout) {
    InputGrad(input, gradin, gradout);
  }
  void Backward(const RealVector &input,
                const boost::shared_ptr<RealVector> &gradin) {
    BaseType::Backward(input, gradin);
  }
  void ResetParamGrad() {}

protected:
  void ParamGrad(const RealVector &input,
                 const boost::shared_ptr<RealVector> &gradin) {
    BaseType::ParamGrad(input, gradin);
  }
  void InputGrad(const RealVector &input,
                 const boost::shared_ptr<RealVector> &gradin,
                 boost::shared_ptr<RealVector> &gradout);

private:
  std::vector<bool> gatestate_;
  std::default_random_engine engine_;
  std::uniform_real_distribution<double> rndgen_;
  int size_;
  double rate_;
};
} // namespace NNModel

#endif // __DROP_OUT_H__