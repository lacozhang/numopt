#pragma once
#include <boost/program_options.hpp>
#include "typedef.h"
#include "DataIterator.h"

template <class ParameterType,
	class SampleType, class LabelType, 
	class SparseGradientType, class DenseGradientType>
class AbstractModel {
public:

	typedef DataIteratorBase<SampleType, LabelType> DataIterator;

	AbstractModel() : optionsdesc_("Model spec options") {}

	virtual void InitFromCmd(int argc, const char* argv[]) = 0;
	virtual void InitFromData(DataIterator& iterator) = 0;
	virtual boost::program_options::options_description& Options() {
		return optionsdesc_;
	}

	virtual ParameterType& GetParameters() = 0;
	virtual ParameterType& GetParameters() const = 0;

	virtual size_t FeatureDimension() const = 0;

	virtual bool LoadModel(std::string model) = 0;
	virtual bool SaveModel(std::string model) = 0;

	// calculate the gradient with respect to data samples
	virtual void Learn(SampleType& samples, LabelType& labels, SparseGradientType& grad) = 0;
	virtual void Learn(SampleType& samples, LabelType& labels, DenseGradientType& grad) = 0;

	// inference model on new samples
	virtual void Inference(SampleType& samples, LabelType& labels) = 0;

	// evaluate model performance
	virtual void Evaluate(SampleType& samples, LabelType& labels, std::string& summary) = 0;

protected:
	boost::program_options::options_description optionsdesc_;
};

template class AbstractModel<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>;
template class AbstractModel<DenseVector, LccrfSamples, LccrfLabels, SparseVector, DenseVector>;