#pragma once

#include "nnmodule.h"

#ifndef __LINEAR_LAYER_H__
#define __LINEAR_LAYER_H__

namespace NNModel {

class LinearLayer
    : public NNLayerBase<RealVector, RealVector, RealVector, RealVector> {
public:
  LinearLayer(double *parambase, double *gradbase, int inputsize,
              int outputsize);
  ~LinearLayer() {}

  // dense input like linear layer
  void Forward(const RealVector &input, boost::shared_ptr<RealVector> &output);
  void Backward(const RealVector &input,
                const boost::shared_ptr<RealVector> &gradin,
                boost::shared_ptr<RealVector> &gradout) {
    ParamGrad(input, gradin);
    InputGrad(input, gradin, gradout);
  }
  void Backward(const RealVector &input,
                const boost::shared_ptr<RealVector> &gradin) {
    NNForbidOperation;
  }
  void ResetParamGrad() { grad_.setZero(); }

protected:
  void ParamGrad(const RealVector &input,
                 const boost::shared_ptr<RealVector> &gradin);
  void InputGrad(const RealVector &input,
                 const boost::shared_ptr<RealVector> &gradin,
                 boost::shared_ptr<RealVector> &gradout);

private:
  Eigen::Map<RowRealMatrix> param_, grad_;
  int inputsize_, outputsize_;
};

} // namespace NNModel

#endif // !__LINEAR_LAYER_H__
