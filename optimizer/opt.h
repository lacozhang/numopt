#ifndef __OPT_H__
#define __OPT_H__

// header file for optimization algorithm like GD, SGD, CG, LBFGS, Proximal SGD, GD
#include <boost/program_options.hpp>
#include "../typedef.h"
#include "../parameter.h"
#include "../AbstractModel.h"
#include "../DataIterator.h"

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

		inline int MaxIter() const {
			return learn_.maxiter_;
		}

		inline double LearningRate() const {
			return learn_.learningrate_;
		}

		inline double LearningRateDecay() const {
			return learn_.learningratedecay_;
		}

		inline double FunctionEpsilon() const {
			return learn_.funceps_;
		}

		inline double GradEpsilon() const {
			return learn_.gradeps_;
		}

		inline double L2RegVal() const {
			return learn_.l2_;
		}

		inline double L1RegVal() const {
			return learn_.l1_;
		}

		double EvaluateOnSet(SampleType& samples, LabelType& labels) {
			std::string message;
			double funcval = this->model_.Evaluate(samples, labels, message);
			BOOST_LOG_TRIVIAL(info) << message << std::endl;
			return funcval;
		}

		virtual void ResultStats(const ParameterType& param) {
			double total = param.size(), zeros = 0;
			for (int i = 0; i < param.size(); ++i) {
				if (param.coeff(i) == 0) {
					zeros += 1;
				}
			}
			BOOST_LOG_TRIVIAL(info) << "sparsity rate " << (zeros / total);
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
		static const char* kBaseMaxLineSearchTriesOption;

	private:
		void ConstructBaseCmdOptions();
};

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
const char* OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::kBaseMaxLineSearchTriesOption = "mlt";
#endif // __OPT_H__
