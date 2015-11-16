#ifndef __LOGISTIC_MODEL_H__
#define __LOGISTIC_MODEL_H__
#include <boost/shared_ptr.hpp>
#include "model.h"
#include "parameter.h"
#include "lossfunc.h"

// current only support binary class logistic regression model.
class LinearModel: public modelbase
{
public:
	LinearModel(IOParameters& io, LossFunc loss);
	~LinearModel();
	void setloss(LossFunc loss);
	void setio(IOParameters& io);
	double lossval();
	double funcval(SparseVector& sample);
	void grad(DenseVector& g);
	void grad(SparseVector& g);
	void setparameter(DenseVector& param);
	DenseVector& param() const;

	int samplesize() const;

	int featsize() const;

	// batch size = -1 means all the training data; batch size = 1, means SGD;
	void startbatch(int batchsize);
	bool nextbatch();

	// model evaluation
	double getaccu();
private:
	void loadtrain(std::string dat);
	void loadtest(std::string dat);
	void savemodel(std::string model);
	boost::shared_ptr<lossbase> loss_;

	// training data
	boost::shared_ptr<DataSamples> trainsamples_;
	boost::shared_ptr<ClsLabelVector> trainlabels_;

	// test data
	boost::shared_ptr<DataSamples> testsamples_;
	boost::shared_ptr<ClsLabelVector> testlabels_;

	int batchsize_; // size of each batch, -1 menas all the training data.
	int sampleidx_; // for each epocs, current index of samples.
	int epochbatch_; // the number of batchs in each epocs. epochbatch_ = trainsamples->rows() / batchsize_;

	// parameters
	boost::shared_ptr<DenseVector> param_;

	// IO parameters
	IOParameters io_;
};


#endif // __LOGISTIC_MODEL_H__