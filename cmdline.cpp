#include "cmdline.h"

void usage(){
	std::cout << "logreg -m model -i trainging.dat -t test.tsv -1 lambda_1 -2 lambda_2 -l hinge -s GD -r 0.03"
		<< std::endl;
}

LossFunc parselossfunc(char* loss){
	if (std::strcmp(loss, "squared")){
		return LossFunc::Squared;
	}
	else if (std::strcmp(loss, "hinge")){
		return LossFunc::Hinge;
	}
	else if (std::strcmp(loss, "logistic")){
		return LossFunc::Logistic;
	}
	else {
		return LossFunc::SquaredHinge;
	}
}

OptMethod parseopt(char* opt){
	if (std::strcmp(opt, "gd")){
		return OptMethod::GD;
	}
	else if (std::strcmp(opt, "sgd")){
		return OptMethod::SGD;
	}
	else if (std::strcmp(opt, "cg")){
		return OptMethod::CG;
	}
	else if (std::strcmp(opt, "lbfgs")){
		return OptMethod::LBFGS;
	}
	else if (std::strcmp(opt, "pgd")){
		return OptMethod::PGD;
	}
	else if (std::strcmp(opt, "cd")){
		return OptMethod::CD;
	}
	else {
		return OptMethod::BCD;
	}
}

bool cmd_line_parse(int argc, char* argv[], Parameter& p){

	int i = 1;
	while (i < argc){
		switch (argv[i][1])
		{
		case 'm':
			p.model_ = argv[++i];
			break;
		case 'i':
			p.train_ = argv[++i];
			break;
		case 't':
			p.test_ = argv[++i];
			break;
		case '1':
			p.l1_ = atof(argv[++i]);
			break;
		case '2':
			p.l2_ = atof(argv[++i]);
			break;
		case 'l':
			p.loss_ = parselossfunc(argv[++i]);
			break;
		case 's':
			p.algo_ = parseopt(argv[++i]);
			break;
		case 'r':
			p.learningRate_ = atof(argv[++i]);
			break;
		case 'h':
		default:
			usage();
			return false;
		}
		++i;
	}

	std::cout << p << std::endl;
	return true;
}