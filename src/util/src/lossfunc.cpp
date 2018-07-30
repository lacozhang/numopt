#include "util/lossfunc.h"

// logloss(a,y) = log(1+exp(-a*y))
double LogLoss::loss(double a, double y) {
  double z = a * y;
  if (z > 18)
    return exp(-z);
  if (z < -18)
    return -z;
  return log(1 + exp(-z));
}

// dloss(a,y)/da
double LogLoss::dloss(double a, double y) {
  double z = a * y;
  if (z > 18)
    return -y * exp(-z);
  if (z < -18)
    return -y;
  return -y / (1 + exp(z));
}

// hingeloss(a,y) = max(0, 1-a*y)
double HingeLoss::loss(double a, double y) {
  double z = a * y;
  if (z > 1)
    return 0;
  return 1 - z;
}

// dloss(a,y)/da
double HingeLoss::dloss(double a, double y) {
  double z = a * y;
  if (z > 1)
    return 0;
  return -y;
}

// squaredhingeloss(a,y) = 1/2 * max(0, 1-a*y)^2
double SquaredHingeLoss::loss(double a, double y) {
  double z = a * y;
  if (z > 1)
    return 0;
  double d = 1 - z;
  return 0.5 * d * d;
}

// dloss(a,y)/da
double SquaredHingeLoss::dloss(double a, double y) {

  double z = a * y;
  if (z > 1)
    return 0;
  return -y * (1 - z);
}

// smoothhingeloss(a,y) = ...
double SmoothHingeLoss::loss(double a, double y) {

  double z = a * y;
  if (z > 1)
    return 0;
  if (z < 0)
    return 0.5 - z;
  double d = 1 - z;
  return 0.5 * d * d;
}

// dloss(a,y)/da
double SmoothHingeLoss::dloss(double a, double y) {

  double z = a * y;
  if (z > 1)
    return 0;
  if (z < 0)
    return -y;
  return -y * (1 - z);
}

double SquaredLoss::loss(double a, double t) { return 0.5 * (a - t) * (a - t); }

double SquaredLoss::dloss(double a, double t) { return (a - t); }
