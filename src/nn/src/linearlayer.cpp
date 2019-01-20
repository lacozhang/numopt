#include "nn/linearlayer.h"

namespace NNModel {
LinearLayer::LinearLayer(double *parambase, double *gradbase, int inputsize,
                         int outputsize)
    : param_(parambase, outputsize, inputsize),
      grad_(gradbase, outputsize, inputsize) {
  inputsize_ = inputsize;
  outputsize_ = outputsize;
  output_ = boost::make_shared<RealVector>();
  inputgrad_ = boost::make_shared<RealVector>();
  output_->resize(outputsize_);
  inputgrad_->resize(inputsize_);
}

void LinearLayer::Forward(const RealVector &input,
                          boost::shared_ptr<RealVector> &output) {
  *output_ += param_ * input;
  output = output_;
}

void LinearLayer::ParamGrad(const RealVector &input,
                            const boost::shared_ptr<RealVector> &gradin) {
  if (!gradin) {
    LOG(ERROR) << "Input Gradient is empty";
    std::abort();
  }
  grad_ += *gradin * input.transpose();
}

void LinearLayer::InputGrad(const RealVector &input,
                            const boost::shared_ptr<RealVector> &gradin,
                            boost::shared_ptr<RealVector> &gradout) {
  if (!gradin) {
    LOG(ERROR) << "Input Gradient is empty";
    std::abort();
  }

  if (!gradout) {
    LOG(ERROR) << "Output Gradient is empty";
    std::abort();
  }

  *inputgrad_ += param_.colwise().sum().transpose();
}
} // namespace NNModel
