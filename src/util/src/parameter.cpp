#include "util/parameter.h"
#include <iostream>

Parameter::Parameter() {
  opt_ = OptMethod::GD;
  loss_ = LossFunc::Squared;
}
