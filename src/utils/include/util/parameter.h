
#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include "util/lossfunc.h"
#include <iostream>
#include <string>

enum class OptMethod {
  GD = 2, // Gradient Descent
  SGD,    // Stochastic Gradient Descent
  CG,     // Conjugate Gradient
  LBFGS,  // Limited BFGS
  PGD,    // Proximal Gradient Descent
  CD,     // coordinate descent
  BCD,    // block coordinate descent
  SVRG,   // stochastic variance reduction gradient
  None
};

enum class ModelType { Linear, LCCRF, SMCRF, None };

struct LearnParameters {
  double l1_, l2_;
  double learningrate_;
  double learningratedecay_;
  int batchsize_;
  int maxiter_;
  int maxlinetries_;
  double funceps_, gradeps_;
  int seed_;
  bool averge_;

public:
  LearnParameters() {
    l1_ = 0;
    l2_ = 1.0;

    learningrate_ = 1e-3;
    learningratedecay_ = 1e-4;
    batchsize_ = -1;
    maxiter_ = 5;
    funceps_ = gradeps_ = 1e-3;
    averge_ = false;
  }
};

struct IOParameters {
  std::string train_, model_, test_;
};

class Parameter {

public:
  Parameter();

  LearnParameters learn_;
  IOParameters io_;
  OptMethod opt_;
  LossFunc loss_;
};

template <class T>
std::basic_ostream<T> &operator<<(std::basic_ostream<T> &sink, Parameter &p) {

  sink << "Parameter for optimization parameter" << std::endl;

  sink << "Using optimization algorithm    : ";
  switch (p.opt_) {
  case OptMethod::GD:
    sink << "GD" << std::endl;
    break;
  case OptMethod::SGD:
    sink << "SGD" << std::endl;
    break;
  case OptMethod::CG:
    sink << "CG" << std::endl;
    break;
  case OptMethod::LBFGS:
    sink << "LBFGS" << std::endl;
    break;
  case OptMethod::PGD:
    sink << "Proximal Gradient Descent" << std::endl;
    break;
  case OptMethod::CD:
    sink << "Coordinate Descent" << std::endl;
    break;
  case OptMethod::BCD:
    sink << "Block Coordinate Descent" << std::endl;
    break;
  default:
    sink << "Unknow Optimization Algorithm" << std::endl;
  }

  sink << "Loss function used for modeling : ";
  switch (p.loss_) {
  case LossFunc::Hinge:
    sink << "Hinge" << std::endl;
    break;
  case LossFunc::Logistic:
    sink << "Logistic" << std::endl;
    break;
  case LossFunc::Squared:
    sink << "Squared" << std::endl;
    break;
  case LossFunc::SquaredHinge:
    sink << "Squared Hinge" << std::endl;
    break;
  default:
    sink << "Error, Unknown loss function" << std::endl;
  }

  sink << "L1 regularization parameter     : ";
  sink << p.learn_.l1_ << std::endl;

  sink << "L2 regularization parameter     : ";
  sink << p.learn_.l2_ << std::endl;

  sink << "Learning rate for optimization  : ";
  sink << p.learn_.learningrate_ << std::endl;

  sink << "Learning rate decay             : ";
  sink << p.learn_.learningratedecay_ << std::endl;

  sink << "Average Parameter                : ";
  sink << p.learn_.averge_ << std::endl;

  sink << "mini-batch size for optimization: ";
  sink << p.learn_.batchsize_ << std::endl;

  sink << "max iteration of learning       : ";
  sink << p.learn_.maxiter_ << std::endl;

  sink << "Training data file              : " << p.io_.train_ << std::endl;
  sink << "Testing data file               : " << p.io_.test_ << std::endl;
  sink << "Model output                    : " << p.io_.model_ << std::endl;

  return sink;
}

#endif
