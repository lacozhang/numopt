#ifndef __LOSS_FUNC_H__
#define __LOSS_FUNC_H__

#include <cmath>

struct lossbase {
	virtual double loss(double a, double y) = 0;
	virtual double dloss(double a, double y) = 0;
};

struct LogLoss : public lossbase
{
	// logloss(a,y) = log(1+exp(-a*y))
    double loss(double a, double y){
		double z = a * y;
		if (z > 18) 
			return exp(-z);
		if (z < -18)
			return -z;
		return log(1 + exp(-z));
	}
	// dloss(a,y)/da
	double dloss(double a, double y){
		double z = a * y;
		if (z > 18) 
			return -y * exp(-z);
		if (z < -18)
			return -y;
		return -y / (1 + exp(z));
	}
};

struct HingeLoss: public lossbase
{
	// hingeloss(a,y) = max(0, 1-a*y)
	double loss(double a, double y){
		double z = a * y;
		if (z > 1) 
			return 0;
		return 1 - z;
	}
	
	// dloss(a,y)/da
    double dloss(double a, double y){
		double z = a * y;
		if (z > 1) 
			return 0;
		return -y;
	}
};

struct SquaredHingeLoss : public lossbase
{
	// squaredhingeloss(a,y) = 1/2 * max(0, 1-a*y)^2
	double loss(double a, double y){
		double z = a * y;
		if (z > 1)
			return 0;
		double d = 1 - z;
		return 0.5 * d * d;
	}

	// dloss(a,y)/da
	double dloss(double a, double y){

		double z = a * y;
		if (z > 1) 
			return 0;
		return -y * (1 - z);
	}
};

struct SmoothHingeLoss : public lossbase
{
	// smoothhingeloss(a,y) = ...
    double loss(double a, double y){

		double z = a * y;
		if (z > 1)
			return 0;
		if (z < 0)
			return 0.5 - z;
		double d = 1 - z;
		return 0.5 * d * d;
	}

	// dloss(a,y)/da
	double dloss(double a, double y){
		
		double z = a * y;
		if (z > 1) 
			return 0;
		if (z < 0)
			return -y;
		return -y * (1 - z);
	}
};

#endif // __LOSS_FUNC_H__
