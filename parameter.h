
#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include <string>
#include <iostream>

class Parameter {

 public:
  enum OptAlgo {
    GD = 2,
    SGD,
    CG,
    LBFGS
  };

  enum LossFunc {
    Squared = 2,
    Hinge,
    Logistic
  };
  
  Parameter();

  double l1_, l2_;
  OptAlgo algo_;
  double learningRate_;
  LossFunc loss_;
  
  std::string train_, model_;
};

template<class T>
std::basic_ostream<T>& operator<< (std::basic_ostream<T>& sink, Parameter& p) {
    
    sink << "Parameter for optimization parameter" << std::endl;
    
    sink << "Using optimization algorithm    : ";
    switch(p.algo_){
    case Parameter::GD:
        sink << "GD" << std::endl;
        break;
    case Parameter::SGD:
        sink << "SGD" << std::endl;
        break;
    case Parameter::CG:
        sink << "CG" << std::endl;
        break;
    case Parameter::LBFGS:
        sink << "LBFGS" << std::endl;
        break;
    default:
        sink << "Unknow Optimization Algorithm" << std::endl;
    }

    sink << "Loss function used for modeling : ";
    switch(p.loss_){
    case Parameter::Hinge:
        sink << "Hinge" << std::endl;
        break;
    case Parameter::Logistic:
        sink << "Logistic" << std::endl;
        break;
    case Parameter::Squared:
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
    sink << "Model output                    : "
         << p.model_ << std::endl;
    
    return sink;
}

#endif
