#include "lbfgs.h"





template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
boost::program_options::options_description LBFGS<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::Options()
{
	boost::program_options::options_description combined;
	combined.add(this->basedesc_);
	combined.add(this->lbfgsdesc_);
	return combined;
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void LBFGS<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::InitCmdDescription()
{
	this->lbfgsdesc_.add_options()
		(this->kLineSearchOption,
			boost::program_options::value<std::string>(&this->lsfuncstr_).default_value("backtrack"), "line search function: backtrack, full")
			(this->kHistoryOption,
				boost::program_options::value<int>(&this->historycnt_).default_value(10), "count of available history")
				(this->kLineSearchStopOption,
					boost::program_options::value<std::string>(&this->lsconfstr_),
					"stop criteria of line search: backtrack, wolfe, strongwolfe");
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void LBFGS<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::ResetState()
{
	this->lsfuncstr_.clear();
	this->lsconfstr_.clear();
	this->gradhistory_.clear();
	this->paramhistory_.clear();
	this->alphas_.clear();
	this->betas_.clear();
	this->rhos_.clear();
	this->historycnt_ = 0;
}
