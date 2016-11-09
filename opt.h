#ifndef __OPT_H__
#define __OPT_H__

// header file for optimization algorithm like GD, SGD, CG, LBFGS, Proximal SGD, GD

#include "typedef.h"
#include "parameter.h"

class OptMethodBase {
public:
	OptMethodBase(LearnParameters& learn);
	virtual ~OptMethodBase();

	int MaxIter() const {
		return learn_.maxiter_;
	}

	double LearningRate() const {
		return learn_.learningrate_;
	}

	double LearningRateDecay() const {
		return learn_.learningratedecay_;
	}

	double FunctionEpsilon() const {
		return learn_.funceps_;
	}

	double GradEpsilon() const {
		return learn_.gradeps_;
	}

	double BatchSize() const {
		return learn_.batchsize_;
	}

	double L2RegVal() const {
		return learn_.l2_;
	}

	double L1RegVal() const {
		return learn_.l1_;
	}

protected:
	LearnParameters learn_;
};

#endif // __OPT_H__