#pragma once
#ifndef __LBFGS_H__
#define __LBFGS_H__
#include "opt.h"

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

		enum class LineSearchOption {
			BackTracking,
			FullLineSearch
		};

		enum class LineSearchStopCondition {
			SufficientDecrease,
			Wolfe,
			StrongWolfe
		};

	private:
		void InitCmdDescription();
		void ResetState();

		static const char* kLineSearchOption;
		static const char* kHistoryOption;
		static const char* kLineSearchStopOption;

		std::string lsfuncstr_;
		std::string lsconfstr_;
		int historycnt_;

		std::vector<boost::shared_ptr<DenseGradientType>> gradhistory_;
		std::vector<boost::shared_ptr<ParameterType>> paramhistory_;
		std::vector<double> alphas_;
		std::vector<double> betas_;
		std::vector<double> rhos_;

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

