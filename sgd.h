#ifndef __SGD_H__
#define __SGD_H__
#include "opt.h"

class StochasticGD : public OptMethodBase{
public:
	StochasticGD(int maxIters, double gradeps, double funceps, bool decay, double initsize);
	~StochasticGD();
	void trainDenseGradient(modelbase& model);
	void trainSparseGradient(modelbase& model);
	DenseVector w_;
	int iternum_;
	bool decay_;
	double stepsize_;
	double initsize_;
};

#endif // __SGD_H__