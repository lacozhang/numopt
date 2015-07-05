#ifndef __SGD_H__
#define __SGD_H__
#include "opt.h"

class StochasticGD : public OptMethodBase{
public:
	StochasticGD(LearnParameters& learn, bool ratedecay);
	~StochasticGD();
	void trainDenseGradient(modelbase& model);
	void trainSparseGradient(modelbase& model);

private:

	bool ratedecay_;
};

#endif // __SGD_H__