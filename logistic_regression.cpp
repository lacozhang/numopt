#include <iostream>
#include "linearmodel.h"
#include "parameter.h"
#include "sgd.h"
#include "cmdline.h"

int main(int argc, const char *argv[]) {

  Parameter param;
  bool ret = cmd_line_parse(argc, argv, param);

  if (!ret) {
    return 0;
  }

  LinearModel lr(param.io_, param.loss_);
  switch (param.opt_) {
  case SGD: // Stochastic Gradient Descent
  {
    StochasticGD sgd(param.learn_, false);
    sgd.trainSparseGradient(lr);
  } break;
  case PGD: // Proximal Gradient Descent
  {
    StochasticGD sgd(param.learn_, false);
    sgd.trainSparseGradient(lr);
  } break;
  case GD: // Gradient Descent
    break;
  case CG: // Conjugate Gradient
    break;
  case LBFGS: // Limited BFGS
    break;
  case CD: // Coordinate Descent
    break;
  case BCD: // Block Coordinate Descent
    break;
  default:
    break;
  }

  return 0;
}