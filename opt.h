#ifndef __OPT_H__
#define __OPT_H__

// header file for optimization algorithm like GD, SGD, CG, LBFGS, Proximal SGD, GD
#include <boost/program_options.hpp>
#include "typedef.h"
#include "parameter.h"
#include "AbstractModel.h"
#include "DataIterator.h"

template<class ParameterType,
	class SampleType, class LabelType,
	class SparseGradientType, class DenseGradientType>
	class OptMethodBase {
	public:

		typedef AbstractModel<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType> ModelSpecType;
		typedef DataIteratorBase<SampleType, LabelType> DataIteratorType;

		OptMethodBase(ModelSpecType& model) : model_(model), basedesc_("General command line options for optimimizatioin method") {
			ConstructBaseCmdOptions();
		}

		virtual ~OptMethodBase();

		virtual void InitFromCmd(int argc, const char* argv[]) = 0;

		virtual void Train() = 0;
		virtual boost::program_options::options_description Options() = 0;

		void SetTrainData(boost::shared_ptr<DataIteratorType> dat) {
			trainiter_ = dat;
		}

		void SetTestData(boost::shared_ptr<DataIteratorType> dat) {
			testiter_ = dat;
		}

		int MaxIter() const {
			return learn_.maxiter_;
		}

		double LearningRate() const {
			return learn_.learningrate_;
		}

		double LearningRateDecay() const {
			return learn_.learningratedecay_;
		}

		double FunctionEpsilon() const {
			return learn_.funceps_;
		}

		double GradEpsilon() const {
			return learn_.gradeps_;
		}

		double L2RegVal() const {
			return learn_.l2_;
		}

		double L1RegVal() const {
			return learn_.l1_;
		}

	protected:
		LearnParameters learn_;
		ModelSpecType& model_;
		boost::program_options::options_description basedesc_;
		boost::shared_ptr<DataIteratorType> trainiter_;
		boost::shared_ptr<DataIteratorType> testiter_;

		static const char* kBaseL1RegOption;
		static const char* kBaseL2RegOption;
		static const char* kBaseMaxItersOption;
		static const char* kBaseFunctionEpsOption;
		static const char* kBaseGradEpsOption;

	private:
		void ConstructBaseCmdOptions();
};

#endif // __OPT_H__