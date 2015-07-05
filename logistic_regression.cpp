#include <iostream>
#include "linearmodel.h"
#include "parameter.h"
#include "sgd.h"
#include "cmdline.h"

int main(int argc, char* argv[]){


	Parameter param;
	bool ret = cmd_line_parse(argc, argv, param);

	if (!ret){
		return 0;
	}

	LinearModel lr(param.io_, param.loss_);
	StochasticGD sgd(param.learn_, false);
	sgd.trainSparseGradient(lr);

	return 0;
}