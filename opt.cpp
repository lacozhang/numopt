#include <iostream>
#include "opt.h"

OptMethodBase::OptMethodBase(int maxIters, double gradeps, double funceps){
	maxiters_ = maxIters;
	gradeps_ = gradeps;
	funceps_ = funceps;
}

OptMethodBase::~OptMethodBase(){

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

	std::cout << "start training" << std::endl;
	std::cout << "feat size : " << model.featsize() << std::endl;
	w_.resize(model.featsize());
	w_.setZero();
	grad_.resize(model.featsize());

	stepsize_ = initsize_;
	std::cout << "step size : " << stepsize_ << std::endl;
	double func0 = abs(model.lossval(w_));

	while (iternum_ < maxiters_){
		std::cout << "epochs " << iternum_ << std::endl;
		model.startbatch(1);

		int samplecnt = 0;
		while (model.nextbatch()){
			samplecnt += 1;
			model.grad(w_, grad_);
			w_ -= stepsize_ * grad_;

			if (samplecnt % 100 == 0){
				std::cout << '.';
			}
			else if (samplecnt % 1000 == 0){
				std::cout << "x" << std::endl;
			}
		}

		std::cout << "func value : " << model.lossval(w_) << std::endl;
		model.grad(w_, grad_);
		std::cout << "grad norm  : " << grad_.norm() << std::endl;

		if (decay_){
			stepsize_ = initsize_ * (1.0 / (1 + iternum_));
		}

		iternum_++;
	}
}

StochasticGD::~StochasticGD(){

}