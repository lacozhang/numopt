#include <iostream>
#include "opt.h"

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::~OptMethodBase(){
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::ConstructBaseCmdOptions()
{
	basedesc_.add_options()
		(this->kBaseL1RegOption, boost::program_options::value<double>(&this->learn_.l1_)->default_value(0.0), "L1 regularization parameter")
		(this->kBaseL2RegOption, boost::program_options::value<double>(&this->learn_.l2_)->default_value(1e-5), "L2 regularization parameter")
		(this->kBaseMaxItersOption, boost::program_options::value<int>(&this->learn_.maxiter_)->default_value(5), "Max number of iterations")
		(this->kBaseFunctionEpsOption, boost::program_options::value<double>(&this->learn_.funceps_)->default_value(1e-7), "Stop criteria for function value change")
		(this->kBaseGradEpsOption, boost::program_options::value<double>(&this->learn_.gradeps_)->default_value(1e-7), "Stop criteria for gradient value change");
}


template class OptMethodBase<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>;
template class OptMethodBase<DenseVector, LccrfSamples, LccrfLabels, SparseVector, DenseVector>;