#include "opt.h"

OptMethodBase::OptMethodBase(int maxIters, double gradeps, double funceps){
	maxiters_ = maxIters;
	gradeps_ = gradeps;
	funceps_ = funceps;
}

StochasticGD::StochasticGD(int maxIters, double gradeps, double funceps, 
	bool decay, double initsize)
:OptMethodBase(maxIters, gradeps, funceps){
	w_.resize(0);
	grad_.resize(0);
	decay_ = decay;
	initsize_ = initsize;
}

void StochasticGD::train(modelbase& model){
	iternum_ = 0;

	w_.resize(model.featsize());
	w_.setZero();
	grad_.resize(model.featsize());

	stepsize_ = initsize_;

	double func0 = abs(model.funcval(w_));

	while (iternum_ < maxiters_){

		model.startbatch(1);

		while (model.nextbatch()){
			model.grad(w_, grad_);
			w_ -= stepsize_ * grad_;
		}

		if (decay_){
			stepsize_ = initsize_ * (1.0 / (1 + iternum_));
		}

		iternum_++;
	}
}