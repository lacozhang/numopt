#include <iostream>
#include "sgd.h"
#include "util.h"

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
const char* StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::kLearningRateOption = "sgd.lr";

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
const char* StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::kLearningRateDecayOption = "sgd.lrd";

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
const char* StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::kAverageGradientOption = "sgd.avg";

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::~StochasticGD() {}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::InitFromCmd(int argc, const char * argv[])
{
	boost::program_options::options_description alloptions("Available options for SGD optimizer");
	alloptions.add(this->basedesc_);
	alloptions.add(this->sgdesc_);

	auto vm = ParseArgs(argc, argv, alloptions, true);
	this->learn_.l1_ = vm[this->kBaseL1RegOption].template as<double>();
	this->learn_.l2_ = vm[this->kBaseL2RegOption].template as<double>();
	this->learn_.learningrate_ = vm[kLearningRateOption].as<double>();
	this->learn_.learningratedecay_ = vm[kLearningRateDecayOption].as<double>();
	this->learn_.maxiter_ = vm[this->kBaseMaxItersOption].template as<int>();
	this->learn_.funceps_ = vm[this->kBaseFunctionEpsOption].template as<double>();
	this->learn_.gradeps_ = vm[this->kBaseGradEpsOption].template as<double>();
	this->learn_.averge_ = vm[kAverageGradientOption].as<bool>();
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::Train()
{
	learnrateiter_ = this->LearningRate();
	itercount_ = 0;
	timeutil timer;

	for (epochcount_ = 0; epochcount_ < this->MaxIter(); ++epochcount_) {
		BOOST_LOG_TRIVIAL(info) << "epoch " << epochcount_ << " start";
		this->trainiter_->ResetBatch();

		timer.tic();
		TrainOneEpoch();
		double secs = timer.toc();
		BOOST_LOG_TRIVIAL(info) << "batch costs " << secs;

		BOOST_LOG_TRIVIAL(info) << "evaluate on train set";
		EvaluateOnSet(this->trainiter_->GetAllData(), this->trainiter_->GetAllLabel());

		if (this->testiter_->IsValid()) {
			BOOST_LOG_TRIVIAL(info) << "evaluate on test set";
			EvaluateOnSet(this->testiter_->GetAllData(), this->testiter_->GetAllLabel());
		}
	}
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
boost::program_options::options_description StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::Options()
{
	boost::program_options::options_description all("Combined options for SGD");
	all.add(this->basedesc_);
	all.add(this->sgdesc_);
	return all;
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::EvaluateOnSet(SampleType & samples, LabelType& labels)
{
	std::string message;
	double correct = 0, total = 0;
	this->model_.Evaluate(samples, labels, message);
	BOOST_LOG_TRIVIAL(info) << message << std::endl;
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::TrainOneEpoch()
{
	ParameterType& param = this->model_.GetParameters();
	SparseGradientType paramgrad;
	SampleType minibatchdata;
	LabelType minibatchlabel;
	size_t epochsize = this->trainiter_->GetSampleSize();

	if (this->learn_.averge_ && epochcount_ == 1) {
		BOOST_LOG_TRIVIAL(trace) << "copy param to averaged param, start to averaging";
		avgparam_.resizeLike(param);
		avgparam_.setZero();
		avgparam_ = param;
	}
	else if (this->learn_.averge_ && epochcount_ > 1) {
		avgparam_.swap(param);
	}

	while (this->trainiter_->GetNextBatch(minibatchdata, minibatchlabel)) {

		this->model_.Learn(minibatchdata, minibatchlabel, paramgrad);
		if (this->learn_.averge_) {
			learnrateiter_ = this->LearningRate() / std::pow(1 + this->LearningRateDecay()*itercount_, 0.75);
		}
		else {
			learnrateiter_ = this->LearningRate() / (1 + this->LearningRateDecay() * itercount_);
		}
		if (this->L2RegVal() > 0) {

#pragma omp parallel for
			for (int featidx = 0; featidx < param.size(); ++featidx) {
				param.coeffRef(featidx) *= (1 - this->L2RegVal() * learnrateiter_);
			}
		}

		// for sparse data, accelerate the speed
		for (SparseVector::InnerIterator it(paramgrad); it; ++it) {
			param.coeffRef(it.index()) -= learnrateiter_ * it.value();
		}

		// get average work now.
		if (this->learn_.averge_ && epochcount_ >= 1) {
			double mu = 1.0 / (1 + this->LearningRateDecay() * (itercount_ - epochsize));
#pragma omp parallel for
			for (int featidx = 0; featidx < avgparam_.size(); ++featidx) {
				avgparam_.coeffRef(featidx) += mu * (param.coeff(featidx) - avgparam_.coeff(featidx));
			}
		}

		itercount_ += 1;
	}

	if (this->learn_.averge_ && epochcount_ >= 1) {
		param.swap(avgparam_);
	}

	BOOST_LOG_TRIVIAL(info) << "param norm : " << param.norm();
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::InitCmdDescription()
{
	sgdesc_.add_options()
		(kLearningRateOption, boost::program_options::value<double>()->default_value(1e-6), "Learning rate for sgd")
		(kLearningRateDecayOption, boost::program_options::value<double>()->default_value(1e-6), "Learning rate decay for learning rate")
		(kAverageGradientOption, boost::program_options::value<bool>()->default_value(false), "Enable gradient averaging or not");
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::ResetState()
{
	learnrateiter_ = 0;
	itercount_ = 0;
	epochcount_ = 0;
}

template class StochasticGD<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>;