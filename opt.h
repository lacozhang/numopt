#ifndef __OPT_H__
#define __OPT_H__

// header file for optimization algorithm like GD, SGD, CG, LBFGS, Proximal SGD, GD

#include "typedef.h"
#include "parameter.h"
#include "model.h"

class OptMethodBase {
public:
	OptMethodBase(LearnParameters& learn);
	virtual ~OptMethodBase();
	virtual void trainDenseGradient(modelbase& model) = 0;
	virtual void trainSparseGradient(modelbase& model) = 0;

	int maxIter() const {
		return learn_.maxiter_;
	}

	double learningRate() const {
		return learn_.learningrate_;
	}

	double functionEpsilon() const {
		return learn_.funceps_;
	}

	double gradEpsilon() const {
		return learn_.gradeps_;
	}

	double batchSize() const {
		return learn_.batchsize_;
	}

	double l2RegVal() const {
		return learn_.l2_;
	}

	double l1RegVal() const {
		return learn_.l1_;
	}

	void ProximalGradient(DenseVector& w, double lambda);
	void ProximalGradient(SparseVector& w, double lambda);

protected:
	LearnParameters learn_;
};

#endif // __OPT_H__