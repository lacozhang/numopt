
#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

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
std::basic_ostream<T>& operator<<(std::basic_ostream<T>& sink, Parameter& p);


#endif
