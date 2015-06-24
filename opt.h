#ifndef __OPT_H__
#define __OPT_H__

// header file for optimization algorithm like GD, SGD, CG, LBFGS, Proximal SGD, GD

#include "typedef.h"
#include "model.h"

enum OptMethod {
	GD = 2, // Gradient Descent
	SGD, // Stochastic Gradient Descent
	CG, // Conjugate Gradient
	LBFGS, // Limited BFGS
	PGD, // Proximal Gradient Descent
	CD, // coordinate descent
	BCD // block coordinate descent
};

class OptMethodBase {
public:
	OptMethodBase(int maxEpochs, double gradeps, double funceps);
	virtual ~OptMethodBase();
	virtual void trainDenseGradient(modelbase& model) = 0;
	virtual void trainSparseGradient(modelbase& model) = 0;

protected:

	int maxiters_;
	double gradeps_;
	double funceps_;
};

#endif // __OPT_H__