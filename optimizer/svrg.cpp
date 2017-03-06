#include "svrg.h"
#include "../util.h"


template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void StochasticVRG<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::InitFromCmd(int argc, const char * argv[])
{
	boost::program_options::options_description overall;
	overall.add(this->basedesc_);
	overall.add(this->svrgdesc_);

	auto vm = ParseArgs(argc, argv, overall, true);
	if (this->learn_.learningrate_ < 0) {
		BOOST_LOG_TRIVIAL(fatal) << "Learning rate is negative, set to default 1e-7";
		this->learn_.learningrate_ = 1e-7;
	}
	if (this->learn_.l1_ > 0) {
		BOOST_LOG_TRIVIAL(info) << "--l1 l1 regularization not enabled for svrg";
	}
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void StochasticVRG<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::Train()
{
	ParameterType& param = this->model_.GetParameters();
	ParameterType avgparam;
	SparseGradientType grad, avgrad;
	DenseGradientType gradcache;
	SampleType minibatchdata;
	LabelType minibatchlabel;
	int iter = 0;
	bool svrgenable = false;
	timeutil timer;

	gradcache.resize(param.size());
	gradcache.setZero();
	grad.resize(param.size());
	grad.setZero();
	avgrad.resize(param.size());
	avgrad.setZero();
	avgparam.resize(param.size());
	avgparam.setZero();

	while (iter <= this->learn_.maxiter_) {
		++iter;
		this->trainiter_->ResetBatch();
		BOOST_LOG_TRIVIAL(info) << "Start iteration " << iter;
		timer.tic();
		if (iter % this->svrginterval_ == 0) {
			svrgenable = true;
			BOOST_LOG_TRIVIAL(info) << "Enable SVRG iterations";
			avgparam = param;
			this->model_.Learn(this->trainiter_->GetAllData(), this->trainiter_->GetAllLabel(), gradcache);
		}

		while (this->trainiter_->GetNextBatch(minibatchdata, minibatchlabel)) {

			this->model_.Learn(minibatchdata, minibatchlabel, grad);
			if (svrgenable) {
				param.swap(avgparam);
				this->model_.Learn(minibatchdata, minibatchlabel, avgrad);
				param.swap(avgparam);
			}

			if (this->L2RegVal() > 0) {
#pragma omp parallel for
				for (int featidx = 0; featidx < param.size(); ++featidx) {
					param.coeffRef(featidx) *= (1 - this->L2RegVal() * this->learn_.learningrate_);
				}
			}

			for (SparseGradientType::InnerIterator it(grad); it; ++it) {
				param.coeffRef(it.index()) -= this->learn_.learningrate_ * it.value();
			}

			if (svrgenable) {
				for (SparseGradientType::InnerIterator it(avgrad); it; ++it) {
					param.coeffRef(it.index()) += this->learn_.learningrate_ * it.value();
				}
				param -= this->learn_.learningrate_ * gradcache;
			}
		}
		double secs = timer.toc();
		BOOST_LOG_TRIVIAL(info) << "batch costs " << secs;
		BOOST_LOG_TRIVIAL(info) << "evaluate on train set";
		EvaluateOnSet(this->trainiter_->GetAllData(), this->trainiter_->GetAllLabel());

		if (this->testiter_->IsValid()) {
			BOOST_LOG_TRIVIAL(info) << "evaluate on test set";
			EvaluateOnSet(this->testiter_->GetAllData(), this->testiter_->GetAllLabel());
		}

		BOOST_LOG_TRIVIAL(info) << "param norm : " << param.norm();
	}
}


template class StochasticVRG<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>;
template class StochasticVRG<DenseVector, LccrfSamples, LccrfLabels, SparseVector, DenseVector>;