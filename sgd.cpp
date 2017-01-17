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
	alloptions.add(basedesc_);
	alloptions.add(sgdesc_);

	auto vm = ParseArgs(argc, argv, alloptions, true);
	learn_.l1_ = vm[kBaseL1RegOption].as<double>();
	learn_.l2_ = vm[kBaseL2RegOption].as<double>();
	learn_.learningrate_ = vm[kLearningRateOption].as<double>();
	learn_.learningratedecay_ = vm[kLearningRateDecayOption].as<double>();
	learn_.maxiter_ = vm[kBaseMaxItersOption].as<int>();
	learn_.funceps_ = vm[kBaseFunctionEpsOption].as<double>();
	learn_.gradeps_ = vm[kBaseGradEpsOption].as<double>();
	learn_.averge_ = vm[kAverageGradientOption].as<bool>();
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::Train()
{
	learnrateiter_ = LearningRate();
	itercount_ = 0;
	timeutil timer;

	for (epochcount_ = 0; epochcount_ < MaxIter(); ++epochcount_) {
		BOOST_LOG_TRIVIAL(info) << "epoch " << epochcount_ << " start";
		trainiter_->ResetBatch();

		timer.tic();
		TrainOneEpoch();
		double secs = timer.toc();
		BOOST_LOG_TRIVIAL(info) << "batch costs " << secs;

		BOOST_LOG_TRIVIAL(info) << "evaluate on train set";
		EvaluateOnSet(trainiter_->GetAllData(), trainiter_->GetAllLabel());

		if (testiter_->IsValid()) {
			BOOST_LOG_TRIVIAL(info) << "evaluate on test set";
			EvaluateOnSet(testiter_->GetAllData(), testiter_->GetAllLabel());
		}
	}
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
boost::program_options::options_description StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::Options()
{
	boost::program_options::options_description all("Combined options for SGD");
	all.add(basedesc_);
	all.add(sgdesc_);
	return all;
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::EvaluateOnSet(SampleType & samples, LabelType& labels)
{
	std::string message;
	double correct = 0, total = 0;
	model_.Evaluate(samples, labels, message);
	BOOST_LOG_TRIVIAL(info) << message << std::endl;
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::TrainOneEpoch()
{
	ParameterType& param = model_.GetParameters();
	SparseGradientType paramgrad;
	SampleType minibatchdata;
	LabelType minibatchlabel;
	size_t epochsize = trainiter_->GetSampleSize();

	if (learn_.averge_ && epochcount_ == 1) {
		BOOST_LOG_TRIVIAL(trace) << "copy param to averaged param, start to averaging";
		avgparam_.resizeLike(param);
		avgparam_.setZero();
		avgparam_ = param;
	}
	else if (learn_.averge_ && epochcount_ > 1) {
		avgparam_.swap(param);
	}

	while (trainiter_->GetNextBatch(minibatchdata, minibatchlabel)) {

		model_.Learn(minibatchdata, minibatchlabel, paramgrad);
		if (learn_.averge_) {
			learnrateiter_ = LearningRate() / std::pow(1 + LearningRateDecay()*itercount_, 0.75);
		}
		else {
			learnrateiter_ = LearningRate() / (1 + LearningRateDecay() * itercount_);
		}
		if (L2RegVal() > 0) {

#pragma omp parallel for
			for (int featidx = 0; featidx < param.size(); ++featidx) {
				param.coeffRef(featidx) *= (1 - L2RegVal() * learnrateiter_);
			}
		}

		// for sparse data, accelerate the speed
		for (SparseVector::InnerIterator it(paramgrad); it; ++it) {
			param.coeffRef(it.index()) -= learnrateiter_ * it.value();
		}

		// get average work now.
		if (learn_.averge_ && epochcount_ >= 1) {
			double mu = 1.0 / (1 + LearningRateDecay() * (itercount_ - epochsize));
#pragma omp parallel for
			for (int featidx = 0; featidx < avgparam_.size(); ++featidx) {
				avgparam_.coeffRef(featidx) += mu * (param.coeff(featidx) - avgparam_.coeff(featidx));
			}
		}

		itercount_ += 1;
	}

	if (learn_.averge_ && epochcount_ >= 1) {
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