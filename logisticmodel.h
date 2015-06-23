#ifndef __LOGISTIC_MODEL_H__
#define __LOGISTIC_MODEL_H__
#include <boost/shared_ptr.hpp>
#include "model.h"

// current only support binary class logistic regression model.
class LogisticModel: public modelbase
{
public:
	LogisticModel();
	LogisticModel(std::string dat);
	~LogisticModel();
	double lossval(DenseVector& param);
	double funcval(DenseVector& param, SparseVector& sample);
	void grad(DenseVector& param, DenseVector& g);
	void setparameter(DenseVector& param);

	int samplesize() const;

	int featsize() const;

	// batch size = -1 means all the training data; batch size = 1, means SGD;
	void startbatch(int batchsize);
	bool nextbatch();
private:
	void dataload(std::string dat);
	LogLoss loss_;
	boost::shared_ptr<DataSamples> trainsamples_;
	boost::shared_ptr<ClsLabelVector> trainlabels_;
	int batchsize_; // size of each batch, -1 menas all the training data.
	int sampleidx_; // for each epocs, current index of samples.
	int epochbatch_; // the number of batchs in each epocs. epochbatch_ = trainsamples->rows() / batchsize_;
	std::vector<double> hypouts_;
	DenseVector param_;
};


#endif // __LOGISTIC_MODEL_H__