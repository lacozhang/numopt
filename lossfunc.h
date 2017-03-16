#ifndef __LOSS_FUNC_H__
#define __LOSS_FUNC_H__

#include <cmath>

enum LossFunc {
	Squared = 2,
	Hinge,
	Logistic,
	SquaredHinge,
	MLECRF
};

struct lossbase {
	virtual double loss(double a, double y) = 0;
	virtual double dloss(double a, double y) = 0;
};

struct SquaredLoss : public lossbase
{
	// squaredloss(a,t) = 1/2 ( a - t) ^ 2
	double loss(double a, double t);

	// dsquaredloss(a,t)/da = (a-t)
	double dloss(double a, double t);
};

struct LogLoss : public lossbase
{
	// logloss(a,y) = log(1+exp(-a*y))
    double loss(double a, double y);

	// dloss(a,y)/da
	double dloss(double a, double y);
};

struct HingeLoss: public lossbase
{
	// hingeloss(a,y) = max(0, 1-a*y)
	double loss(double a, double y);
	
	// dloss(a,y)/da
    double dloss(double a, double y);
};

struct SquaredHingeLoss : public lossbase
{
	// squaredhingeloss(a,y) = 1/2 * max(0, 1-a*y)^2
	double loss(double a, double y);

	// dloss(a,y)/da
	double dloss(double a, double y);
};

struct SmoothHingeLoss : public lossbase
{
	// smoothhingeloss(a,y) = ...
    double loss(double a, double y);

	// dloss(a,y)/da
	double dloss(double a, double y);
};

#endif // __LOSS_FUNC_H__
