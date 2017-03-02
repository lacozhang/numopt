#pragma once
#ifndef __LBFGS_H__
#define __LBFGS_H__
#include "opt.h"
#include "linesearch.h"

template<class ParameterType,
	class SampleType, class LabelType,
	class SparseGradientType, class DenseGradientType>
	class LBFGS : public OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType> {
	public:

		typedef OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType> OptMethodBaseType;
		typedef DataIteratorBase<SampleType, LabelType> DataIterator;

		LBFGS(typename OptMethodBaseType::ModelSpecType& model)
			: OptMethodBaseType(model), lbfgsdesc_("") {
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

			if (this->learn_.l1_ > 0) {
				BOOST_ASSERT(false);
				funcval += this->learn_.l1_ * modelparam.lpNorm<1>();
			}
			return funcval;
		}

		void InitCmdDescription();
		void ResetState();

		static const char* kLineSearchOption;
		static const char* kHistoryOption;
		static const char* kLineSearchStopOption;

		std::string lsfuncstr_;
		std::string lsconfstr_;
		int historycnt_;

		std::vector<DenseGradientType> gradhistory_;
		std::vector<ParameterType> paramhistory_;
		std::vector<double> alphas_;
		std::vector<double> betas_;
		std::vector<double> rhos_;
		boost::shared_ptr<LineSearcher> lsearch_;
		int itercnt_;

		boost::program_options::options_description lbfgsdesc_;
};

template<class ParameterType,
	class SampleType, class LabelType,
	class SparseGradientType, class DenseGradientType>
	const char* LBFGS<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::kLineSearchOption = "lbfgs.ls";

template<class ParameterType,
	class SampleType, class LabelType,
	class SparseGradientType, class DenseGradientType>
	const char* LBFGS<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::kHistoryOption = "lbfgs.hist";

template<class ParameterType,
	class SampleType, class LabelType,
	class SparseGradientType, class DenseGradientType>
	const char* LBFGS<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::kLineSearchStopOption = "lbfgs.lscond";
#endif // !__LBFGS_H__

