#pragma once

#include "nnmodule.h"

#ifndef __VECTOR_SUM_H__
#define __VECTOR_SUM_H__

namespace NNModel {

class VectorSumLayer
    : public NNLayerBase<std::vector<boost::shared_ptr<RealVector>>, RealVector,
                         RealVector, RealVector> {
public:
  typedef NNLayerBase<std::vector<boost::shared_ptr<RealVector>>, RealVector,
                      RealVector, RealVector>
      BaseType;

  VectorSumLayer(int inputsize, int vectorsize)
      : inputsize_(inputsize), vectorsize_(vectorsize) {
    output_ = boost::make_shared<RealVector>();
    output_->resize(vectorsize);
  }

  ~VectorSumLayer() {}

  void Forward(const std::vector<boost::shared_ptr<RealVector>> &input,
               boost::shared_ptr<RealVector> &output);

  void Backward(const std::vector<boost::shared_ptr<RealVector>> &input,
                const boost::shared_ptr<RealVector> &gradin,
                boost::shared_ptr<RealVector> &gradout) {
    InputGrad(input, gradin, gradout);
  }
  void Backward(const std::vector<boost::shared_ptr<RealVector>> &input,
                const boost::shared_ptr<RealVector> &gradin) {
    BaseType::Backward(input, gradin);
  }

  void ResetParamGrad() { BaseType::ResetParamGrad(); }

protected:
  void ParamGrad(const std::vector<boost::shared_ptr<RealVector>> &input,
                 const boost::shared_ptr<RealVector> &gradin) {
    BaseType::ParamGrad(input, gradin);
  }
  void InputGrad(const std::vector<boost::shared_ptr<RealVector>> &input,
                 const boost::shared_ptr<RealVector> &gradin,
                 boost::shared_ptr<RealVector> &gradout);

private:
  int inputsize_, vectorsize_;
};

} // namespace NNModel

#endif // __VECTOR_SUM_H__