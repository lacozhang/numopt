#include "optimizer/svrg.h"
#include "util/util.h"

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
void StochasticVRG<ParameterType, SampleType, LabelType, SparseGradientType,
                   DenseGradientType>::InitFromCmd(int argc,
                                                   const char *argv[]) {
  boost::program_options::options_description overall;
  overall.add(this->basedesc_);
  overall.add(this->svrgdesc_);

  auto vm = ParseArgs(argc, argv, overall, true);
  if (this->learn_.learningrate_ < 0) {
    LOG(FATAL) << "Learning rate is negative, set to default 1e-7";
    this->learn_.learningrate_ = 1e-7;
  }
  if (this->learn_.l1_ > 0) {
    LOG(INFO) << "--l1 l1 regularization not enabled for svrg";
  }
}

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
void StochasticVRG<ParameterType, SampleType, LabelType, SparseGradientType,
                   DenseGradientType>::Train() {
  ParameterType &param = this->model_.GetParameters();
  ParameterType avgparam;
  SparseGradientType grad, avgrad;
  DenseGradientType gradcache;
  SampleType minibatchdata;
  LabelType minibatchlabel;
  int iter = 0;
  size_t epochsize = this->trainiter_->GetSampleSize();
  bool svrgenable = false;
  timeutil timer;

  gradcache.resize(param.size());
  gradcache.setZero();
  grad.resize(param.size());
  grad.setZero();
  avgrad.resize(param.size());
  avgrad.setZero();
  avgparam.resize(param.size());
  avgparam.setZero();

  while (iter <= this->learn_.maxiter_) {
    ++iter;
    LOG(INFO) << "Start iteration " << iter;
    timer.tic();
    svrgenable = true;
    LOG(INFO) << "Enable SVRG iterations";
    avgparam = param;
    this->model_.Learn(this->trainiter_->GetAllData(),
                       this->trainiter_->GetAllLabel(), gradcache);
    gradcache += this->learn_.l2_ * avgparam;

    for (size_t sampleidx = 0; sampleidx < epochsize; ++sampleidx) {

      this->trainiter_->GetRandomBatch(minibatchdata, minibatchlabel);
      this->model_.Learn(minibatchdata, minibatchlabel, grad);
      if (svrgenable) {
        param.swap(avgparam);
        this->model_.Learn(minibatchdata, minibatchlabel, avgrad);
        param.swap(avgparam);
      }

      if (this->learn_.l2_ > 0) {
#pragma omp parallel for
        for (int featidx = 0; featidx < param.size(); ++featidx) {
          param.coeffRef(featidx) *=
              (1 - this->learn_.l2_ * this->learn_.learningrate_ * epochsize);
        }
      }

      for (typename SparseGradientType::InnerIterator it(grad); it; ++it) {
        param.coeffRef(it.index()) -= this->learn_.learningrate_ * it.value();
      }

      if (svrgenable) {
        for (typename SparseGradientType::InnerIterator it(avgrad); it; ++it) {
          param.coeffRef(it.index()) += this->learn_.learningrate_ * it.value();
        }

#pragma omp parallel for
        for (int featidx = 0; featidx < param.size(); ++featidx) {
          param.coeffRef(featidx) += this->learn_.l2_ *
                                     this->learn_.learningrate_ * epochsize *
                                     avgparam.coeff(featidx);
          param.coeffRef(featidx) -=
              this->learn_.learningrate_ * gradcache.coeff(featidx);
        }
      }
    }
    double secs = timer.toc();
    LOG(INFO) << "batch costs " << secs;
    LOG(INFO) << "evaluate on train set";
    this->EvaluateOnSet(this->trainiter_->GetAllData(),
                        this->trainiter_->GetAllLabel());

    if (this->testiter_->IsValid()) {
      LOG(INFO) << "evaluate on test set";
      this->EvaluateOnSet(this->testiter_->GetAllData(),
                          this->testiter_->GetAllLabel());
    }

    LOG(INFO) << "param norm : " << param.norm();
  }
}

template class StochasticVRG<DenseVector, DataSamples, LabelVector,
                             SparseVector, DenseVector>;
template class StochasticVRG<DenseVector, LccrfSamples, LccrfLabels,
                             SparseVector, DenseVector>;
