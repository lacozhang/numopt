#include <iostream>
#include "sgd.h"
#include "util.h"

StochasticGD::StochasticGD(LearnParameters& learn, bool ratedecay)
	:OptMethodBase(learn){
	ratedecay_ = ratedecay;
}

void StochasticGD::trainDenseGradient(modelbase& model){
	std::cerr << "Gradient Descent for DenseGradient Vector implemented" << std::endl;
}

void StochasticGD::trainSparseGradient(modelbase& model){


	int iternum_ = 0;
	SparseVector grad;
	DenseVector& params = model.param();
	params.setZero();


	std::cout << "start training" << std::endl;
	std::cout << "feat size : " << model.featsize() << std::endl;
	grad.resize(model.featsize());

	double initsize = learningRate();
	double stepsize = initsize;
	std::cout << "step size : " << stepsize << std::endl;
	double func0 = abs(model.lossval());
	std::cout << "F0 value  : " << func0
		<< std::endl;

	timeutil t;


	while (iternum_ < maxIter()){
		std::cout << "epochs " << iternum_ << std::endl;
		int samplecnt = 0;

		t.tic();
		model.startbatch(1);
		while (model.nextbatch()){
			samplecnt += 1;

			model.grad(grad);
			for (SparseVector::InnerIterator iter(grad); iter; ++iter){
				params.coeffRef(iter.index()) -= stepsize * iter.value();
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

		std::cout << "func value : " << model.lossval() << std::endl;
		std::cout << "grad norm  : " << params.norm() << std::endl;

		if (ratedecay_){
			stepsize = initsize * (1.0 / (1 + iternum_));
		}

		iternum_++;
	}
}

StochasticGD::~StochasticGD(){

}