#include <iostream>
#include "opt.h"

OptMethodBase::OptMethodBase(LearnParameters& learn){
	learn_ = learn;
}

OptMethodBase::~OptMethodBase(){
}


void OptMethodBase::ProximalGradient(DenseVector& w, double lambda){

	for (int i = 0; i < w.size(); ++i){
		if (w.coeff(i) > lambda){
			w.coeffRef(i) = lambda;
		}
		else if (w.coeff(i) < -lambda){
			w.coeffRef(i) = -lambda;
		}
		else {
			w.coeffRef(i) = 0;
		}
	}
}

void OptMethodBase::ProximalGradient(SparseVector& w, double lambda){

	for (SparseVector::InnerIterator iter(w); iter; ++iter){
		if (iter.value() > lambda){
			iter.valueRef() = lambda;
		}
		else if (iter.value() < -lambda){
			iter.valueRef() = -lambda;
		}
		else {
			iter.valueRef() = 0;
		}
	}
}