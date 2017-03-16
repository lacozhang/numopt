#include <string>
#include <map>
#include <functional>

#include "opt.h"
#include "sgd.h"
#include "lbfgs.h"
#include "cg.h"
#include "svrg.h"
#include "proximalgd.h"
#include "sdca.h"
#include "../typedef.h"
#include "../AbstractModel.h"
#include "../LccrfModel.h"

#ifndef __OPTIM_FACTORY_H__
#define __OPTIM_FACTORY_H__

template<class OptimizerType, class ParameterType,
	class SampleType, class LabelType,
	class SparseGradientType, class DenseGradientType>
	OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>* CreateOptimizer(
		AbstractModel<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>& model) {
	return new OptimizerType(model);
}


template<class ParameterType,
	class SampleType, class LabelType,
	class SparseGradientType, class DenseGradientType>
	class BaseOptimizerFactory {
	public:

		typedef OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType> BaseOptimizerType;
		typedef AbstractModel<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType> BaseModelType;
		typedef std::function<BaseOptimizerType*(BaseModelType&)> OptimizerCreationFn;

		BaseOptimizerFactory() {
		}

		template<class OptimizerType>
		void Register(std::string name) {
			if (!funcobjmap_.count(name)) {
				funcobjmap_[name] = std::bind(&CreateOptimizer<OptimizerType, ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>, std::placeholders::_1);
			}
		}

		BaseOptimizerType* Create(std::string name, BaseModelType& model) {
			if (funcobjmap_.count(name)) {
				return funcobjmap_[name](model);
			}
			return nullptr;
		}

	private:
		static std::map<std::string, OptimizerCreationFn> funcobjmap_;

};

template<class ParameterType,
	class SampleType, class LabelType,
	class SparseGradientType, class DenseGradientType>
	std::map<std::string, typename BaseOptimizerFactory<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::OptimizerCreationFn> BaseOptimizerFactory<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::funcobjmap_;

class LinearFactory : public BaseOptimizerFactory<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector> {
public:
	LinearFactory() {
		std::string name = "sgd";
		Register<StochasticGD<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>>(name);

		name = "cg";
		Register<ConjugateGradient<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>>(name);

		name = "lbfgs";
		Register<LBFGS<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>>(name);

		name = "svrg";
		Register<StochasticVRG<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>>(name);

		name = "pgd";
		Register<ProxGradientDescent<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>>(name);

		name = "sdca";
		Register<StochasticDCA<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>>(name);
	}
};

class LccrfFactory : public BaseOptimizerFactory<DenseVector, LccrfSamples, LccrfLabels, SparseVector, DenseVector> {
public:
	LccrfFactory() {
		std::string name = "sgd";
		Register<StochasticGD<DenseVector, LccrfSamples, LccrfLabels, SparseVector, DenseVector>>(name);

		name = "cg";
		Register<ConjugateGradient<DenseVector, LccrfSamples, LccrfLabels, SparseVector, DenseVector>>(name);

		name = "lbfgs";
		Register<LBFGS<DenseVector, LccrfSamples, LccrfLabels, SparseVector, DenseVector>>(name);

		name = "svrg";
		Register<StochasticVRG<DenseVector, LccrfSamples, LccrfLabels, SparseVector, DenseVector>>(name);

		name = "pgd";
		Register<ProxGradientDescent<DenseVector, LccrfSamples, LccrfLabels, SparseVector, DenseVector>>(name);
	}
};

#endif // __OPTIM_FACTORY_H__