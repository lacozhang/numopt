#include "nn/textpooling.h"

namespace NNModel {

void TextMaxPoolingLayer::Forward(const RowRealMatrix &input,
                                  boost::shared_ptr<RealVector> &output) {
  if (input.cols() != convsize_) {
    NNForbidOperationMsg("Input column size not equal to convolution size");
  }
  output_->setZero();
  poolingidx_.setZero();
  for (int i = 0; i < convsize_ * stack_; ++i) {
    output_->coeffRef(i) = std::numeric_limits<double>::min();
    poolingidx_.coeffRef(i) = -1;
  }

  for (int i = 0; i < stack_; ++i) {
    for (int idx = 0; idx < input.rows(); idx += i + 1) {
      for (int j = 0; j < convsize_; ++j) {
        if (input.coeff(idx, j) > output_->coeff(i * convsize_ + j)) {
          output_->coeffRef(i * convsize_ + j) = input.coeff(idx, j);
          poolingidx_.coeffRef(i * convsize_ + j) = idx;
        }
      }
    }
  }

  output = output_;
}

void TextMaxPoolingLayer::InputGrad(const RowRealMatrix &input,
                                    const boost::shared_ptr<RealVector> &gradin,
                                    boost::shared_ptr<RowRealMatrix> &gradout) {
  inputgrad_->resizeLike(input);
  inputgrad_->setZero();
  if (gradin->size() != output_->size()) {
    NNForbidOperationMsg("Error, size not equal");
  }
  for (int i = 0; i < stack_; ++i) {
    for (int j = 0; j < convsize_; ++j) {
      int row = poolingidx_.coeff(i * convsize_ + j);
      inputgrad_->coeffRef(row, j) += gradin->coeff(i * convsize_ + j);
    }
  }

  gradout = inputgrad_;
}
} // namespace NNModel