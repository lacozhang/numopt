#include <iostream>
#include "sgd.h"
#include "util.h"

StochasticGD::StochasticGD(int maxIters, double gradeps, double funceps,
	bool decay, double initsize)
	:OptMethodBase(maxIters, gradeps, funceps){
	decay_ = decay;
	initsize_ = initsize;
}

void StochasticGD::trainDenseGradient(modelbase& model){

	iternum_ = 0;
	DenseVector grad;
	std::cout << "start training" << std::endl;
	std::cout << "feat size : " << model.featsize() << std::endl;
	w_.resize(model.featsize());
	w_.setZero();
	grad.resize(model.featsize());

	stepsize_ = initsize_;
	std::cout << "step size : " << stepsize_ << std::endl;
	double func0 = abs(model.lossval());
	std::cout << "F0 value  : " << func0
		<< std::endl;

	timeutil t;
	

	while (iternum_ < maxiters_){
		std::cout << "epochs " << iternum_ << std::endl;
		int samplecnt = 0;

		t.tic();
		model.startbatch(1);
		while (model.nextbatch()){
			samplecnt += 1;

			t.tic();
			model.grad(grad);
			std::cout << "grad interface costs :" << t.toc() << std::endl;
			t.tic();
			w_ -= stepsize_ * grad;
			std::cout << "grad desc costs :" << t.toc() << std::endl;

			if (samplecnt % 100 == 0){
				std::cout << '.';
			}
			else if (samplecnt % 1000 == 0){
				std::cout << "x" << std::endl;
			}
		}
		std::cout << "epoch learning costs : " << t.toc()
			<< std::endl;

		std::cout << "func value : " << model.lossval(w_) << std::endl;
		model.grad(w_, grad);
		std::cout << "grad norm  : " << grad.norm() << std::endl;

		if (decay_){
			stepsize_ = initsize_ * (1.0 / (1 + iternum_));
		}

		iternum_++;
	}
}

void StochasticGD::trainSparseGradient(modelbase& model){
	iternum_ = 0;
	SparseVector grad;
	std::cout << "start training" << std::endl;
	std::cout << "feat size : " << model.featsize() << std::endl;
	w_.resize(model.featsize());
	w_.setZero();
	grad.resize(model.featsize());

	stepsize_ = initsize_;
	std::cout << "step size : " << stepsize_ << std::endl;
	double func0 = abs(model.lossval(w_));
	std::cout << "F0 value  : " << func0
		<< std::endl;

	timeutil t;


	while (iternum_ < maxiters_){
		std::cout << "epochs " << iternum_ << std::endl;
		int samplecnt = 0;

		t.tic();
		model.startbatch(1);
		while (model.nextbatch()){
			samplecnt += 1;

			model.grad(w_, grad);
			for (SparseVector::InnerIterator iter(grad); iter; ++iter){
				w_.coeffRef(iter.index()) -= stepsize_ * iter.value();
			}

			if (samplecnt % 100000 == 0){
				std::cout << 'x' << std::endl;
			}
			else if (samplecnt % 10000 == 0){
				std::cout << ".";
			}
		}

		std::cout << "epoch learning costs : " << t.toc()
			<< std::endl;

		std::cout << "func value : " << model.lossval(w_) << std::endl;
		std::cout << "grad norm  : " << w_.norm() << std::endl;

		if (decay_){
			stepsize_ = initsize_ * (1.0 / (1 + iternum_));
		}

		iternum_++;
	}
}

StochasticGD::~StochasticGD(){

}