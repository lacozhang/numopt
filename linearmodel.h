#ifndef __LOGISTIC_MODEL_H__
#define __LOGISTIC_MODEL_H__
#include <boost/shared_ptr.hpp>
#include "parameter.h"
#include "lossfunc.h"
#include "typedef.h"

class BinaryLinearModel
{
public:
	BinaryLinearModel(LossFunc loss, size_t featdim, float bias);
	~BinaryLinearModel();

	void SetLoss(LossFunc loss);

	DenseVector& GetParameters() const {
		return *param_;
	}

	size_t FeatureDimension() const {
		return featdim_;
	}

	bool LoadModel(std::string model);
	bool SaveModel(std::string model);

	void Learn(DataSamples& samples, LabelVector& labels, SparseVector& grad);
	void Learn(DataSamples& samples, LabelVector& labels, DenseVector& grad);

	void Inference(DataSamples& samples, LabelVector& labels);

	void Evaluate(DataSamples& samples, LabelVector& labels, std::string& summary);

private:
	void Init();

	boost::shared_ptr<lossbase> loss_;
	boost::shared_ptr<DenseVector> param_;
	size_t featdim_;
};

#endif // __LOGISTIC_MODEL_H__