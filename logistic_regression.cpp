#include <iostream>
#include "logisticmodel.h"
#include "parameter.h"
#include "sgd.h"
#include "cmdline.h"

int main(int argc, char* argv[]){

	Parameter param;
	bool ret = cmd_line_parse(argc, argv, param);

	if (!ret){
		return 0;
	}

	LogisticModel lr(param.io_.train_);
	StochasticGD sgd(200, 1e-3, 1e-3, false, param.learn_.learningRate_);
	sgd.trainSparseGradient(lr);

	return 0;
}