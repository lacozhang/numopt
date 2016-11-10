#include <iostream>
#include "sgd.h"
#include "util.h"

StochasticGD::StochasticGD(LearnParameters& learn, DataIterator& trainiter, DataIterator& testiter, BinaryLinearModel& model) :
	OptMethodBase(learn), trainiter_(trainiter), testiter_(testiter), model_(model)
{
	learnrateiter_ = 0;
	itercount_ = 0;
}

StochasticGD::~StochasticGD() {}

void StochasticGD::Train()
{
	learnrateiter_ = LearningRate();
	itercount_ = 0;
	timeutil timer;

	for (int i = 0; i < MaxIter(); ++i) {
		BOOST_LOG_TRIVIAL(info) << "epoch " << i << " start";
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
	while (trainiter_.GetNextBatch(minibatchdata, minibatchlabel)) {

		model_.Learn(minibatchdata, minibatchlabel, paramgrad);
		learnrateiter_ = LearningRate() / (1 + LearningRateDecay() * itercount_);
		if (L2RegVal() > 0) {
			param *= (1 - L2RegVal() * learnrateiter_);
		}
		param -= learnrateiter_ * paramgrad;
		itercount_ += 1;
	}
	BOOST_LOG_TRIVIAL(info) << "param norm : " << param.norm();
}
