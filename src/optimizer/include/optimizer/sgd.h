#ifndef __SGD_H__
#define __SGD_H__
#include "dataop/DataIterator.h"
#include "model/linearmodel.h"
#include "optimizer/opt.h"

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
class StochasticGD
    : public OptMethodBase<ParameterType, SampleType, LabelType,
                           SparseGradientType, DenseGradientType> {
public:
  typedef OptMethodBase<ParameterType, SampleType, LabelType,
                        SparseGradientType, DenseGradientType>
      OptMethodBaseType;
  typedef DataIteratorBase<SampleType, LabelType> DataIterator;

  StochasticGD(typename OptMethodBaseType::ModelSpecType &model)
      : OptMethodBaseType(model), sgdesc_("Options for Sgd") {
    InitCmdDescription();
    ResetState();
  }

  ~StochasticGD();

  virtual void InitFromCmd(int argc, const char *argv[]) override;
  virtual void Train() override;
  virtual boost::program_options::options_description Options() override;

private:
  void TrainOneEpoch();
  void InitCmdDescription();
  void ResetState();

  double learnrateiter_;
  size_t itercount_;
  size_t epochcount_;
  DenseVector avgparam_;

  static const char *kLearningRateOption;
  static const char *kLearningRateDecayOption;
  static const char *kAverageGradientOption;

  boost::program_options::options_description sgdesc_;
};

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
const char
    *StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType,
                  DenseGradientType>::kLearningRateOption = "sgd.lr";

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
const char
    *StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType,
                  DenseGradientType>::kLearningRateDecayOption = "sgd.lrd";

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
const char
    *StochasticGD<ParameterType, SampleType, LabelType, SparseGradientType,
                  DenseGradientType>::kAverageGradientOption = "sgd.avg";

#endif // __SGD_H__