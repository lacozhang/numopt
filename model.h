
#ifndef __MODEL_H__ 
#define __MODEL_H__

#include "typedef.h"

class modelbase{
public:
	modelbase();
	~modelbase();

	virtual double funcval(DenseVector& param) = 0;
	virtual void grad(DenseVector& param, DenseVector& g) = 0;

	virtual int samplesize() const = 0;
	virtual int featsize() const = 0;

	// batch size = -1 means all the training data; batch size = 1, means SGD;
	virtual void startbatch(int batchsize) = 0;
	virtual bool nextbatch() = 0;
};

#endif // __MODEL_H__