#pragma once

#ifndef __SVRG_H__
#define __SVRG_H__
#include "opt.h"

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
class StochasticVRG
    : public OptMethodBase<ParameterType, SampleType, LabelType,
                           SparseGradientType, DenseGradientType> {
public:
  typedef OptMethodBase<ParameterType, SampleType, LabelType,
                        SparseGradientType, DenseGradientType>
      OptMethodBaseType;
  typedef DataIteratorBase<SampleType, LabelType> DataIterator;

  StochasticVRG(typename OptMethodBaseType::ModelSpecType &model)
      : OptMethodBaseType(model),
        svrgdesc_("Options for Stochastic Variance Reduction Gradient") {
    InitCmdDescription();
    ResetState();
  }

  ~StochasticVRG() {}

  virtual void InitFromCmd(int argc, const char *argv[]) override;
  virtual void Train() override;

  virtual boost::program_options::options_description Options() override {
    boost::program_options::options_description alldesc;
    alldesc.add(this->basedesc_);
    alldesc.add(this->svrgdesc_);
    return alldesc;
  }

private:
  void InitCmdDescription() {
    this->svrgdesc_.add_options()(
        this->kSVRGLearningRateOption,
        boost::program_options::value<double>(&this->learn_.learningrate_)
            ->default_value(1e-5),
        "learning rate for svrg")(
        this->kSVRGFrequencyOption,
        boost::program_options::value<int>(&this->svrginterval_)
            ->default_value(2),
        "Update optimial estimated parameter");
  }
  void ResetState() {}
  int svrginterval_;
  boost::program_options::options_description svrgdesc_;

  static const char *kSVRGLearningRateOption;
  static const char *kSVRGFrequencyOption;
};

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
const char
    *StochasticVRG<ParameterType, SampleType, LabelType, SparseGradientType,
                   DenseGradientType>::kSVRGLearningRateOption = "svrg.lr";

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
const char
    *StochasticVRG<ParameterType, SampleType, LabelType, SparseGradientType,
                   DenseGradientType>::kSVRGFrequencyOption = "svrg.iter";
#endif // !__SVRG_H__
