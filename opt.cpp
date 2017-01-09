#include <iostream>
#include "opt.h"


template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
const char* OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::kBaseL1RegOption = "l1";

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
const char* OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::kBaseL2RegOption = "l2";

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
const char* OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::kBaseMaxItersOption = "maxiter";

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
const char* OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::kBaseFunctionEpsOption = "feps";

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
const char* OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::kBaseGradEpsOption = "geps";

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::~OptMethodBase(){
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::ConstructBaseCmdOptions()
{
	basedesc_.add_options()
		(kBaseL1RegOption, boost::program_options::value<double>()->default_value(0.0), "L1 regularization parameter")
		(kBaseL2RegOption, boost::program_options::value<double>()->default_value(1e-5), "L2 regularization parameter")
		(kBaseMaxItersOption, boost::program_options::value<int>()->default_value(5), "Max number of iterations")
		(kBaseFunctionEpsOption, boost::program_options::value<double>()->default_value(1e-7), "Stop criteria for function value change")
		(kBaseGradEpsOption, boost::program_options::value<double>()->default_value(1e-7), "Stop criteria for gradient value change");
}


template OptMethodBase<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>;