#pragma once
#include "opt.h"
#include "util/util.h"

#ifndef __PROXIMAL_GRADIENT_H__
#define __PROXIMAL_GRADIENT_H__

// proximal gradient used for batch method with l1 regularization

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
class ProxGradientDescent
    : public OptMethodBase<ParameterType, SampleType, LabelType,
                           SparseGradientType, DenseGradientType> {
public:
  typedef OptMethodBase<ParameterType, SampleType, LabelType,
                        SparseGradientType, DenseGradientType>
      OptMethodBaseType;
  typedef DataIteratorBase<SampleType, LabelType> DataIterator;

  ProxGradientDescent(typename OptMethodBaseType::ModelSpecType &model)
      : OptMethodBaseType(model),
        proxgdesc_("Options for Proximal Gradient Descent") {
    InitCmdDescription();
    ResetState();
  }
  ~ProxGradientDescent() {}

  virtual void InitFromCmd(int argc, const char *argv[]) override {
    boost::program_options::options_description combined;
    combined.add(this->basedesc_);
    combined.add(this->proxgdesc_);

    auto vm = ParseArgs(argc, argv, combined, true);
    if (this->fixedstepsize_) {
      LOG(INFO) << "Using fixed stepsize for proximal gradient descent";
      if (this->stepsize_ == 0.0) {
        LOG(INFO) << "Using default value 0.1";
        this->stepsize_ = 0.1;
      } else {
        LOG(INFO) << "Using fixed stepsize " << this->stepsize_;
      }
    } else {
      LOG(INFO) << "Using line search to determine stepsize";
    }

    if (this->learn_.l1_ == 0) {
      LOG(WARNING) << "Do not use L1 regularization for Proximal "
                      "gradient descent is like GD";
    }
  }
  virtual void Train() override;

  virtual boost::program_options::options_description Options() override {
    boost::program_options::options_description alldesc;
    alldesc.add(this->basedesc_);
    alldesc.add(this->proxgdesc_);
    return alldesc;
  }

private:
  void InitCmdDescription() {
    this->proxgdesc_.add_options()(
        this->kConstStepSizeOption,
        boost::program_options::value<bool>(&this->fixedstepsize_)
            ->default_value(false),
        "whether used fixed step size for proximal gradient descent")(
        this->kStepSizeOption,
        boost::program_options::value<double>(&this->stepsize_)
            ->default_value(0),
        "when using fixed step size strategy, using this value");
  }

  void ResetState() {
    fixedstepsize_ = false;
    stepsize_ = 0.0;
  }

  void SoftThresholding(ParameterType &base, DenseGradientType &grad,
                        ParameterType &res, double stepsize, double l1reg) {
    res = base - (1.0 / stepsize) * grad;
    double threshold = (1.0 / stepsize) * l1reg;
#pragma omp parallel for
    for (int i = 0; i < res.size(); ++i) {
      if (res.coeff(i) > threshold) {
        res.coeffRef(i) -= threshold;
      } else if (res.coeff(i) < -threshold) {
        res.coeffRef(i) += threshold;
      } else {
        res.coeffRef(i) = 0.0;
      }
    }
  }

  boost::program_options::options_description proxgdesc_;
  bool fixedstepsize_;
  double stepsize_;
  ParameterType tempdir_;

  static const char *kConstStepSizeOption;
  static const char *kStepSizeOption;
};

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
const char *ProxGradientDescent<ParameterType, SampleType, LabelType,
                                SparseGradientType,
                                DenseGradientType>::kConstStepSizeOption =
    "pgd.const";

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
const char *ProxGradientDescent<ParameterType, SampleType, LabelType,
                                SparseGradientType,
                                DenseGradientType>::kStepSizeOption =
    "pgd.stepsize";

#endif // !__PROXIMAL_GRADIENT_H__
