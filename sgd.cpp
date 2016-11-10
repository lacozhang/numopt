#include <iostream>
#include "sgd.h"
#include "util.h"

StochasticGD::StochasticGD(LearnParameters& learn, DataIterator& trainiter, DataIterator& testiter, BinaryLinearModel& model) :
	OptMethodBase(learn), trainiter_(trainiter), testiter_(testiter), model_(model)
{
	learnrateiter_ = 0;
	itercount_ = 0;
	epochcount_ = 0;
}

StochasticGD::~StochasticGD() {}

void StochasticGD::Train()
{
	learnrateiter_ = LearningRate();
	itercount_ = 0;
	timeutil timer;

	for (epochcount_ = 0; epochcount_ < MaxIter(); ++epochcount_) {
		BOOST_LOG_TRIVIAL(info) << "epoch " << epochcount_ << " start";
		trainiter_.ResetBatch();

		timer.tic();
		TrainOneEpoch();
		double secs = timer.toc();
		BOOST_LOG_TRIVIAL(info) << "batch costs " << secs;

		BOOST_LOG_TRIVIAL(info) << "evaluate on train set";
		EvaluateOnSet(trainiter_.GetAllData(), trainiter_.GetAllLabel());

		if (testiter_.IsValid()) {
			BOOST_LOG_TRIVIAL(info) << "evaluate on test set";
			EvaluateOnSet(testiter_.GetAllData(), testiter_.GetAllLabel());
		}
	}
}

void StochasticGD::EvaluateOnSet(DataSamples & samples, LabelVector & labels)
{
	std::string message;
	double correct = 0, total = 0;
	model_.Evaluate(samples, labels, message);
	BOOST_LOG_TRIVIAL(info) << message << std::endl;
}

void StochasticGD::TrainOneEpoch()
{
	DenseVector& param = model_.GetParameters();
	SparseVector paramgrad;
	DataSamples minibatchdata;
	LabelVector minibatchlabel;
	size_t epochsize = trainiter_.GetAllData().rows();

	if (learn_.averge_ && epochcount_ == 1) {
		BOOST_LOG_TRIVIAL(trace) << "copy param to averaged param, start to averaging";
		avgparam_.resizeLike(param);
		avgparam_.setZero();
		avgparam_ = param;
	}
	else if (learn_.averge_ && epochcount_ > 1) {
		avgparam_.swap(param);
	}

	while (trainiter_.GetNextBatch(minibatchdata, minibatchlabel)) {

		model_.Learn(minibatchdata, minibatchlabel, paramgrad);
		if (learn_.averge_) {
			learnrateiter_ = LearningRate() / std::pow(1 + LearningRateDecay()*itercount_, 0.75);
		}
		else {
			learnrateiter_ = LearningRate() / (1 + LearningRateDecay() * itercount_);
		}
		if (L2RegVal() > 0) {
			param *= (1 - L2RegVal() * learnrateiter_);
		}
		param -= learnrateiter_ * paramgrad;

		// get average work now.
		if (learn_.averge_ && epochcount_ >= 1) {
			double mu = 1.0 / (1 + LearningRateDecay() * (itercount_ - epochsize));
			avgparam_ += mu * (param - avgparam_);
		}

		itercount_ += 1;
	}

	if (learn_.averge_ && epochcount_ >= 1) {
		param.swap(avgparam_);
	}

	BOOST_LOG_TRIVIAL(info) << "param norm : " << param.norm();
}
