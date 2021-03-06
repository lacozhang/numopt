#include "optimizer/proximalgd.h"

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
void ProxGradientDescent<ParameterType, SampleType, LabelType,
                         SparseGradientType, DenseGradientType>::Train() {
  ParameterType &param = this->model_.GetParameters();
  DenseGradientType grad;
  int itercnt = 0;
  double funcval = 0.0;

  grad.resize(param.size());
  grad.setZero();
  tempdir_.resize(param.size());
  tempdir_.setZero();

  while (itercnt < this->learn_.maxiter_) {

    LOG(INFO) << "iter " << itercnt;
    this->model_.Learn(this->trainiter_->GetAllData(),
                       this->trainiter_->GetAllLabel(), grad);
    if (this->learn_.l2_ > 0) {
      grad += this->learn_.l2_ * param;
    }

    if (this->fixedstepsize_) {
      SoftThresholding(param, grad, tempdir_, this->stepsize_,
                       this->learn_.l1_);
      tempdir_.swap(param);
    } else {
      BOOST_ASSERT_MSG(false, "Proximal gradient descent not implemented yet");
    }

    funcval = OptMethodBaseType::EvaluateOnSet(this->trainiter_->GetAllData(),
                                               this->trainiter_->GetAllLabel());
    if (this->testiter_->IsValid()) {
      OptMethodBaseType::EvaluateOnSet(this->testiter_->GetAllData(),
                                       this->testiter_->GetAllLabel());
    }

    if (this->learn_.l2_ > 0) {
      funcval += param.norm();
    }

    if (this->learn_.l1_ > 0) {
      funcval += param.template lpNorm<1>();
    }
    LOG(INFO) << "Grad norm  : " << grad.norm();
    LOG(INFO) << "Param norm : " << param.norm();
    LOG(INFO) << "Objective value " << funcval;

    ++itercnt;
  }
  this->ResultStats(param);
}

template class ProxGradientDescent<DenseVector, DataSamples, LabelVector,
                                   SparseVector, DenseVector>;
template class ProxGradientDescent<DenseVector, LccrfSamples, LccrfLabels,
                                   SparseVector, DenseVector>;
