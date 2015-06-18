
#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include <string>
#include <iostream>
#include "opt.h"
class Parameter {
  
public:
  Parameter();

  double l1_, l2_;
  OptMethod algo_;
  double learningRate_;
  LossFunc loss_;
  
  std::string train_, model_, test_;
};

template<class T>
std::basic_ostream<T>& operator<< (std::basic_ostream<T>& sink, Parameter& p) {
    
    sink << "Parameter for optimization parameter" << std::endl;
    
    sink << "Using optimization algorithm    : ";
    switch(p.algo_){
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
    switch(p.loss_){
    case LossFunc::Hinge:
        sink << "Hinge" << std::endl;
        break;
	case LossFunc::Logistic:
        sink << "Logistic" << std::endl;
        break;
	case LossFunc::Squared:
        sink << "Squared" << std::endl;
        break;
    default:
        sink << "Error, Unknown loss function"
             << std::endl;
    }

    sink << "L1 regularization parameter     : ";
    sink << p.l1_ << std::endl;

    sink << "L2 regularization parameter     : ";
    sink << p.l2_ << std::endl;

    sink << "Learning rate for optimization  : ";
    sink << p.learningRate_ << std::endl;

    sink << "Training data file              : "
         << p.train_ << std::endl;
	sink << "Testing data file               : "
		<< p.test_ << std::endl;
    sink << "Model output                    : "
         << p.model_ << std::endl;
    
    return sink;
}

#endif
