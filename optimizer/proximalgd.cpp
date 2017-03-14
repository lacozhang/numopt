#include "proximalgd.h"

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void ProxGradientDescent<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::Train()
{
	ParameterType& param = this->model_.GetParameters();
	DenseGradientType grad;
	int itercnt = 0;
	double funcval = 0.0;

	grad.resize(param.size());
	grad.setZero();
	tempdir_.resize(param.size());
	tempdir_.setZero();

	while (itercnt < this->learn_.maxiter_) {
		
		BOOST_LOG_TRIVIAL(info) << "iter " << itercnt;
		this->model_.Learn(this->trainiter_->GetAllData(), this->trainiter_->GetAllLabel(), grad);
		if (this->learn_.l2_ > 0) {
			grad += this->learn_.l2_ * param;
		}

		if (this->fixedstepsize_) {
			SoftThresholding(param, grad, tempdir_, this->stepsize_, this->learn_.l1_);
			tempdir_.swap(param);
		}
		else {
			BOOST_ASSERT_MSG(false, "Proximal gradient descent not implemented yet");
		}

		funcval = EvaluateOnSet(this->trainiter_->GetAllData(), this->trainiter_->GetAllLabel());
		if (this->testiter_->IsValid()) {
			EvaluateOnSet(this->testiter_->GetAllData(), this->testiter_->GetAllLabel());
		}

		if (this->learn_.l2_ > 0) {
			funcval += param.norm();
		}

		if (this->learn_.l1_ > 0) {
			funcval += param.lpNorm<1>();
		}
		BOOST_LOG_TRIVIAL(info) << "Grad norm  : " << grad.norm();
		BOOST_LOG_TRIVIAL(info) << "Param norm : " << param.norm();
		BOOST_LOG_TRIVIAL(info) << "Objective value " << funcval;

		++itercnt;
	}
	ResultStats(param);
}

template class ProxGradientDescent<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>;
template class ProxGradientDescent<DenseVector, LccrfSamples, LccrfLabels, SparseVector, DenseVector>;