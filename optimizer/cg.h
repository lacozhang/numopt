#pragma once
#ifndef __CG_H__
#define __CG_H__
#include "opt.h"
#include "linesearch.h"

enum class ConjugateGenerationMethod {
	None,
	FR, // Fletcher and Reeves formula beta = \delta{f(x_{k+1}} . \delta{f(x_{k+1})}
	PR, // Polak and Ribiere formula   beta = \delta{f(x_{k+1})} . (\delta{f(x_{k+1})} - \delta{f(x_{k})})
};

template<class ParameterType,
	class SampleType, class LabelType,
	class SparseGradientType, class DenseGradientType>
	class ConjugateGradient : public OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType> {	
	public:
		typedef OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType> OptMethodBaseType;
		typedef DataIteratorBase<SampleType, LabelType> DataIterator;

		ConjugateGradient(typename OptMethodBaseType::ModelSpecType& model)
			: OptMethodBaseType(model), cgdesc_("Options for conjugate gradient") {
			InitCmdDescription();
			ResetState();
		}

		virtual void InitFromCmd(int argc, const char* argv[]) override;
		virtual void Train() override;
		virtual boost::program_options::options_description Options() override;

	private:

		double EvaluateValueAndGrad(ParameterType& modelparam, DenseGradientType& grad) {
			this->model_.GetParameters().swap(modelparam);
			double funcval = this->model_.Learn(this->trainiter_->GetAllData(), this->trainiter_->GetAllLabel(), grad);
			this->model_.GetParameters().swap(modelparam);
			if (this->learn_.l2_ > 0) {
				grad += this->learn_.l2_ * modelparam;
				funcval += 0.5*this->learn_.l2_ * modelparam.dot(modelparam);
			}
			return funcval;
		}

		void InitCmdDescription();
		void ResetState();
		boost::program_options::options_description cgdesc_;

		DenseGradientType pastgrad_;
		int restartcounter_;
		std::string methodstr_, lsfuncstr_;
		ConjugateGenerationMethod method_;
		boost::shared_ptr<LineSearcher> lsearch_;

};


#endif // !__CG_H__
