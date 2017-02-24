#pragma once
#ifndef __CG_H__
#define __CG_H__
#include "../opt.h"
template<class ParameterType,
	class SampleType, class LabelType,
	class SparseGradientType, class DenseGradientType>
	class ConjugateGradient : public OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType> {	
	public:
		typedef OptMethodBase<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType> OptMethodBaseType;
		typedef DataIteratorBase<SampleType, LabelType> DataIterator;

		ConjugateGradient(typename OptMethodBaseType::ModelSpecType& model)
			: OptMethodBaseType(model), cgdesc_("Options for conjugate gradient") {

		}

		virtual void InitFromCmd(int argc, const char* argv[]) override;
		virtual void Train() override;
		virtual boost::program_options::options_description Options() override;

	private:
		void InitCmdDescription();
		void ResetState();
		boost::program_options::options_description cgdesc_;
};


#endif // !__CG_H__
