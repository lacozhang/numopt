#include "cmdline.h"

LossFunc parselossfunc(const char *loss) {
  if (!std::strcmp(loss, "squared")) {
    return LossFunc::Squared;
  } else if (!std::strcmp(loss, "hinge")) {
    return LossFunc::Hinge;
  } else if (!std::strcmp(loss, "logistic")) {
    return LossFunc::Logistic;
  } else if (!std::strcmp(loss, "squaredhinge")) {
    return LossFunc::SquaredHinge;
  } else {
    std::cerr << "unsupported loss function " << loss << std::endl;
    return LossFunc::Squared;
  }
}

OptMethod parseopt(const char *opt) {
  if (!std::strcmp(opt, "gd")) {
    return OptMethod::GD;
  } else if (!std::strcmp(opt, "sgd")) {
    return OptMethod::SGD;
  } else if (!std::strcmp(opt, "cg")) {
    return OptMethod::CG;
  } else if (!std::strcmp(opt, "lbfgs")) {
    return OptMethod::LBFGS;
  } else if (!std::strcmp(opt, "pgd")) {
    return OptMethod::PGD;
  } else if (!std::strcmp(opt, "cd")) {
    return OptMethod::CD;
  } else if (!std::strcmp(opt, "bcd")) {
    return OptMethod::BCD;
  } else {
    return OptMethod::SGD;
  }
}

void usage() {
	std::cout << "logreg -m model -i trainging.dat -t test.tsv -1 lambda_1_reg -2 lambda_2 -l<Loss Function> hinge -s<Optimization Method> GD -r<Learning Rate> 0.03 -b<Batch Size> 10 -x<Init Seed> 0 -e<Max Iters> 5 -d<learning Rate Decy> 1e-3"
		<< std::endl;
}

bool cmd_line_parse(int argc, const char *argv[], Parameter &p) {

  int i = 1;

  if (argc < 2) {
    usage();
    exit(-1);
  }

  while (i < argc) {
    switch (argv[i][1]) {
    case 'm':
      p.io_.model_ = argv[++i];
      break;
    case 'i':
      p.io_.train_ = argv[++i];
      break;
    case 't':
      p.io_.test_ = argv[++i];
      break;
    case '1':
      p.learn_.l1_ = atof(argv[++i]);
      break;
    case '2':
      p.learn_.l2_ = atof(argv[++i]);
      break;
    case 'l':
      p.loss_ = parselossfunc(argv[++i]);
      break;
    case 's':
      p.opt_ = parseopt(argv[++i]);
      break;
    case 'r':
      p.learn_.learningrate_ = atof(argv[++i]);
      break;
    case 'b':
      p.learn_.batchsize_ = atoi(argv[++i]);
      break;
    case 'e':
      p.learn_.maxiter_ = atoi(argv[++i]);
      break;
	case 'x':
		p.learn_.seed_ = atoi(argv[++i]);
		break;
	case 'd':
		p.learn_.learningratedecay_ = atof(argv[++i]);
		break;
	case 'a':
		p.learn_.averge_ = true;
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