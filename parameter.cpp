#include <iostream>
#include "parameter.h"

Parameter::Parameter(){
    
    l1_ = 0;
    l2_ = 0;
  
    algo_ = GD;
    loss_ = Squared;
    learningRate_ = 1e-3;

    train_ = model_ = "";
}