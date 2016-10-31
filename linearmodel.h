#ifndef __LOGISTIC_MODEL_H__
#define __LOGISTIC_MODEL_H__
#include <boost/shared_ptr.hpp>
#include "parameter.h"
#include "lossfunc.h"
#include "typedef.h"

class LinearModel
{
public:
	LinearModel(LossFunc loss, size_t featdim, size_t numclasses);
	~LinearModel();

	void SetLoss(LossFunc loss);

	DenseVector& GetParameters() const;
	size_t FeatureDimension() const;
	size_t NumClasses() const;

	bool LoadModel(std::string model);
	bool SaveModel(std::string model, bool binary);

	void RetrieveGradient(DataSamples& samples, LabelVector& labels, DenseVector& grad);
	void RetrieveGradient(SparseVector& sample, int label, DenseVector& grad);
	void RetrieveGradient(DataSamples& samples, LabelVector& labels, SparseVector& grad);
	void RetrieveGradient(SparseVector& sample, int label, SparseVector& grad);

private:
	boost::shared_ptr<lossbase> loss_;
	boost::shared_ptr<DenseVector> param_;
	size_t numclasses_;
}

#endif // __LOGISTIC_MODEL_H__