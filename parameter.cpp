
#include "parameter.h"

Paramter::Parameter(){

  l1_ = 0f;
  l2_ = 0f;
  
  algo_ = GD;
  loss_ = Squared;
  learningRate_ = 1e-3;

  train_ = model_ = "";
}


template<class T>
std::basic_ostream<T>& operator <<(std::basic_ostream<T>& sink, Parameter& p){

  sink << "Parameter for optimization parameter" << std::endl;
  
}
