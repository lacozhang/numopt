#ifndef __SDCA_H__
#define __SDCA_H__
#include "opt.h"

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
class StochasticDCA
    : public OptMethodBase<ParameterType, SampleType, LabelType,
                           SparseGradientType, DenseGradientType> {

public:
  typedef OptMethodBase<ParameterType, SampleType, LabelType,
                        SparseGradientType, DenseGradientType>
      OptMethodBaseType;
  typedef DataIteratorBase<SampleType, LabelType> DataIterator;

  StochasticDCA(typename OptMethodBaseType::ModelSpecType &model)
      : OptMethodBaseType(model), sdcadesc_("Options for Sgd") {
    InitCmdOptions();
  }

  virtual void InitFromCmd(int argc, const char *argv[]) override;
  virtual void Train() override;

  virtual boost::program_options::options_description Options() override {
    boost::program_options::options_description alldesc;
    alldesc.add(this->basedesc_);
    alldesc.add(this->sdcadesc_);
    return alldesc;
  }

private:
  void InitCmdOptions() {
    this->sdcadesc_.add_options()(
        this->kDualityGapOption,
        boost::program_options::value<double>(&this->dualgap_)
            ->default_value(1e-3),
        "stop criteria used by SDCA. i.e. duality gap");
  }

  void UpdateDataNorms(DataSamples &dat, DenseVector &res, int epochsize);

  bool TrainOneEpoch(DataSamples &datas, LabelVector &labels,
                     DenseVector &param, DenseVector &dualcoef,
                     DenseVector &proxcoef, std::vector<int> &rndptrs,
                     DenseVector &datanorms, int epochsize, LossFunc loss,
                     double lambda, double alpha, double beta);

  boost::program_options::options_description sdcadesc_;
  ParameterType dualcoef_;
  ParameterType proxcoef_;
  double dualgap_;

  static const char *kDualityGapOption;
};

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
const char
    *StochasticDCA<ParameterType, SampleType, LabelType, SparseGradientType,
                   DenseGradientType>::kDualityGapOption = "sdca.gap";

#endif // !__SDCA_H__
