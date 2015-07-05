#include <iostream>
#include "opt.h"

OptMethodBase::OptMethodBase(LearnParameters& learn){
	learn_ = learn;
}

OptMethodBase::~OptMethodBase(){
}
