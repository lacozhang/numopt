#ifndef __SGD_H__
#define __SGD_H__
#include "opt.h"
#include "DataIterator.h"
#include "linearmodel.h"

class StochasticGD : public OptMethodBase {
public:
	StochasticGD(LearnParameters &learn, DataIterator& trainiter, DataIterator& testiter, BinaryLinearModel& model);
	~StochasticGD();

	void Train();
	void EvaluateOnSet(DataSamples& samples, LabelVector& labels);

private:
	void TrainOneEpoch();

	double learnrateiter_;
	size_t itercount_;
	DataIterator& trainiter_;
	DataIterator& testiter_;
	BinaryLinearModel& model_;
};

#endif // __SGD_H__