#include "util/cmdline.h"
#include <cstring>

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
	}
	else if (!std::strcmp(opt, "sgd")) {
		return OptMethod::SGD;
	}
	else if (!std::strcmp(opt, "cg")) {
		return OptMethod::CG;
	}
	else if (!std::strcmp(opt, "lbfgs")) {
		return OptMethod::LBFGS;
	}
	else if (!std::strcmp(opt, "pgd")) {
		return OptMethod::PGD;
	}
	else if (!std::strcmp(opt, "cd")) {
		return OptMethod::CD;
	}
	else if (!std::strcmp(opt, "bcd")) {
		return OptMethod::BCD;
	}
	else if(!std::strcmp(opt, "svrg")) {
		return OptMethod::SVRG;
	}
	else {
		return OptMethod::None;
	}
}

ModelType parsemodel(const char* model) {
	if (!std::strcmp(model, "linear")) {
		return ModelType::Linear;
	}
	else if (!std::strcmp(model, "lccrf")) {
		return ModelType::LCCRF;
	}
	else if (!std::strcmp(model, "smcrf")) {
		return ModelType::SMCRF;
	}
	else {
		return ModelType::None;
	}
}
