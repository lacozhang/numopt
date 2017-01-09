#pragma once
#ifndef __LOGISTIC_MODEL_H__
#define __LOGISTIC_MODEL_H__
#include <boost/shared_ptr.hpp>
#include "AbstractModel.h"
#include "typedef.h"
#include "DataIterator.h"
#include "parameter.h"
#include "lossfunc.h"

class BinaryLinearModel : public AbstractModel<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>
{
public:

	typedef AbstractModel<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector> BaseModelType;

	BinaryLinearModel();
	~BinaryLinearModel();

	void SetLoss(LossFunc loss);

	virtual void InitFromCmd(int argc, const char* argv[]) override;
	virtual void InitFromData(DataIterator& iterator) override;

	virtual DenseVector& GetParameters() const override {
		return *param_;
	}

	virtual DenseVector& GetParameters() override {
		return *param_;
	}

	size_t FeatureDimension() const {
		return featdim_;
	}

	virtual bool LoadModel(std::string model) override;
	virtual bool SaveModel(std::string model) override;

	virtual void Learn(DataSamples& samples, LabelVector& labels, SparseVector& grad) override;
	virtual void Learn(DataSamples& samples, LabelVector& labels, DenseVector& grad) override;

	virtual void Inference(DataSamples& samples, LabelVector& labels) override;

	virtual void Evaluate(DataSamples& samples, LabelVector& labels, std::string& summary) override;

private:

	boost::shared_ptr<lossbase> loss_;
	boost::shared_ptr<DenseVector> param_;
	size_t featdim_;
	double bias_;

	static const char* kLossOption;
	static const char* kBiasOption;
};

#endif // __LOGISTIC_MODEL_H__