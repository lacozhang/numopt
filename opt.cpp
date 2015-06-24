#include <iostream>
#include "opt.h"

OptMethodBase::OptMethodBase(int maxIters, double gradeps, double funceps){
	maxiters_ = maxIters;
	gradeps_ = gradeps;
	funceps_ = funceps;
}

OptMethodBase::~OptMethodBase(){

}
