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
	virtual void train(modelbase& model) = 0;

protected:

	int maxiters_;
	double gradeps_;
	double funceps_;
};


class StochasticGD: public OptMethodBase{
public:
	StochasticGD(int maxIters, double gradeps, double funceps, bool decay, double initsize);
	~StochasticGD();
	void train(modelbase& model);
	DenseVector w_;
	DenseVector grad_;
	int iternum_;
	bool decay_;
	double stepsize_;
	double initsize_;
};

#endif // __OPT_H__