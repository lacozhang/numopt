#include <iostream>
#include "parameter.h"

Parameter::Parameter(){
    
    opt_ = OptMethod::GD;
    loss_ = LossFunc::Squared;
}