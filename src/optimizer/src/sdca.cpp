#include "optimizer/sdca.h"
#include "util/util.h"

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
void StochasticDCA<ParameterType, SampleType, LabelType, SparseGradientType,
                   DenseGradientType>::InitFromCmd(int argc,
                                                   const char *argv[]) {
  boost::program_options::options_description alldesc;
  alldesc.add(this->basedesc_);
  alldesc.add(this->sdcadesc_);

  auto vm = ParseArgs(argc, argv, alldesc, true);

  BOOST_ASSERT_MSG(this->learn_.l1_ >= 0,
                   "l1 regularization must larger or equal to 0");
  BOOST_ASSERT_MSG(this->learn_.l2_ >= 0,
                   "l2 regularization must larget or equal to 0");
  if (this->dualgap_ <= 0) {
    BOOST_LOG_TRIVIAL(warning)
        << "Duality gap can be negative or 0, set to default value(1e-3)";
    this->dualgap_ = 1e-3;
  }
}

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
void StochasticDCA<ParameterType, SampleType, LabelType, SparseGradientType,
                   DenseGradientType>::Train() {
  int itercnt = 0;
  double funcval = 0.0;
  size_t epochsize = this->trainiter_->GetSampleSize();
  ParameterType &param = this->model_.GetParameters();
  ParameterType datanorms;

  dualcoef_.resize(epochsize);
  dualcoef_.setZero();
  proxcoef_.resize(epochsize);
  proxcoef_.setZero();

  datanorms.resize(epochsize);
  datanorms.setZero();
  UpdateDataNorms(this->trainiter_->GetAllData(), datanorms, epochsize);

  double lambda = 0, alpha = 0, beta = 0;
  if ((this->learn_.l2_ == 0) && (this->learn_.l1_ != 0)) {
    BOOST_LOG_TRIVIAL(info)
        << "For Prox-SDCA to work, need to adjust the parameters";
    lambda = this->dualgap_ * this->learn_.l1_ * this->learn_.l1_;
    beta = 1.0 / (this->learn_.l1_ * this->dualgap_);
    alpha = 1.0;
    this->dualgap_ *= 0.5;
  } else if (this->learn_.l2_ != 0) {
    lambda = this->learn_.l2_ * 2;
    alpha = 1.0;
    beta = this->learn_.l1_ / (this->learn_.l2_ * 2);
  } else {
    lambda = 0.0;
    alpha = beta = 0.0;
    BOOST_ASSERT_MSG(false, "Optimiza with no regularization at all");
    return;
  }

  LossFunc loss = this->model_.LossFunction();
  switch (loss) {
  case LossFunc::Hinge:
  case LossFunc::SquaredHinge:
    break;
  default:
    BOOST_LOG_TRIVIAL(fatal) << "Loss function not supported";
    return;
    BOOST_ASSERT_MSG(false, "Loss function not supported");
  }

  BOOST_LOG_TRIVIAL(info) << "Optimize with follow regularization: lambda "
                             "*(alpha/2 * l2_reg + beta * l1_reg)";
  BOOST_LOG_TRIVIAL(info) << "lambda : " << lambda;
  BOOST_LOG_TRIVIAL(info) << "alpha  : " << alpha;
  BOOST_LOG_TRIVIAL(info) << "beta   : " << beta;

  this->dualcoef_.resize(epochsize);
  this->dualcoef_.setZero();
  std::vector<int> rndptrs;
  for (int i = 0; i < epochsize; ++i) {
    rndptrs.push_back(i);
  }

  while (itercnt < this->learn_.maxiter_) {
    BOOST_LOG_TRIVIAL(info) << "Start iteration " << itercnt;
    std::random_shuffle(rndptrs.begin(), rndptrs.end());

    if (TrainOneEpoch(this->trainiter_->GetAllData(),
                      this->trainiter_->GetAllLabel(), param, dualcoef_,
                      proxcoef_, rndptrs, datanorms, epochsize, loss, lambda,
                      alpha, beta)) {
      BOOST_LOG_TRIVIAL(info) << "Converged";
      break;
    }

    funcval = this->EvaluateOnSet(this->trainiter_->GetAllData(),
                                  this->trainiter_->GetAllLabel());
    if (this->testiter_->IsValid()) {
      this->EvaluateOnSet(this->testiter_->GetAllData(),
                          this->testiter_->GetAllLabel());
    }

    if (this->learn_.l2_ > 0) {
      funcval += this->learn_.l2_ * param.norm();
    }

    if (this->learn_.l1_ > 0) {
      funcval += this->learn_.l1_ * param.template lpNorm<1>();
    }
    BOOST_LOG_TRIVIAL(info) << "Param norm : " << param.norm();
    BOOST_LOG_TRIVIAL(info) << "Objective value " << funcval;
    itercnt++;
  }

  this->EvaluateOnSet(this->trainiter_->GetAllData(),
                      this->trainiter_->GetAllLabel());
  if (this->testiter_->IsValid()) {
    this->EvaluateOnSet(this->testiter_->GetAllData(),
                        this->testiter_->GetAllLabel());
  }
  this->ResultStats(param);
}

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
void StochasticDCA<ParameterType, SampleType, LabelType, SparseGradientType,
                   DenseGradientType>::UpdateDataNorms(DataSamples &dat,
                                                       DenseVector &res,
                                                       int epochsize) {
  for (int i = 0; i < epochsize; ++i) {
    res.coeffRef(i) = dat.row(i).squaredNorm();
  }
}

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
bool StochasticDCA<ParameterType, SampleType, LabelType, SparseGradientType,
                   DenseGradientType>::TrainOneEpoch(DataSamples &datas,
                                                     LabelVector &labels,
                                                     DenseVector &param,
                                                     DenseVector &dualcoef,
                                                     DenseVector &proxcoef,
                                                     std::vector<int> &rndptrs,
                                                     DenseVector &datanorms,
                                                     int epochsize,
                                                     LossFunc loss,
                                                     double lambda,
                                                     double alpha,
                                                     double beta) {
  double primal = 0, dual = 0;
  double regval = 0;
  double scale = 1.0 / (epochsize * lambda);
  for (int i = 0; i < epochsize; ++i) {
    int index = rndptrs[i];

    double pred = 0.0;
    for (DataSamples::InnerIterator iter(datas, index); iter; ++iter) {
      pred += param.coeff(iter.col()) * iter.value();
    }
    double label = labels.coeff(index);
    double norm = datanorms.coeff(index);
    double update = 0;
    switch (loss) {
    case LossFunc::Hinge: {
      update =
          (1 - pred * label) / (norm * scale) + dualcoef.coeff(index) * label;
      update = std::min(update, 1.0);
      update = std::max(0.0, update);
      update = label * update - dualcoef.coeff(index);
      primal += std::max(0.0, 1 - pred * label);
      dual += label * update;
    } break;
    case LossFunc::SquaredHinge: {
      update = -(pred - label + dualcoef.coeff(index)) / (0.5 + norm * scale);
      if ((1 - label * pred) > 0) {
        primal += (1 - label * pred) * (1 - label * pred);
      }
      if (dualcoef.coeff(index) + update < 0)
        update = -dualcoef.coeff(index);
    } break;
    }

    // update with only l2 regularization
    if (beta == 0) {
      for (DataSamples::InnerIterator iter(datas, index); iter; ++iter) {
        double delta = iter.value() * update * scale;
        regval += delta * (2 * dualcoef.coeff(iter.col()) + delta);
        dualcoef.coeffRef(iter.col()) += delta;
      }
    } // update with l1 & l2 regularization
    else {
      for (DataSamples::InnerIterator iter(datas, index); iter; ++iter) {
        double delta = iter.value() * update * scale;
        regval -= param.coeff(iter.col()) * proxcoef.coeff(iter.col());
        proxcoef.coeffRef(iter.col()) += delta;
        if (proxcoef.coeff(iter.col()) > beta) {
          param.coeffRef(iter.col()) = proxcoef.coeff(iter.col()) - beta;
        } else if (proxcoef.coeff(iter.col()) < -beta) {
          param.coeffRef(iter.col()) = proxcoef.coeff(iter.col()) + beta;
        } else {
          param.coeffRef(iter.col()) = 0.0;
        }
        regval += param.coeff(iter.col()) * proxcoef.coeff(iter.col());
      }
    }
  }

  double gap = primal - dual + regval;
  gap /= epochsize;
  BOOST_LOG_TRIVIAL(info) << "Duality gap " << gap;
  BOOST_LOG_TRIVIAL(info) << "Desired gap " << this->dualgap_;
  if (gap < this->dualgap_) {
    return true;
  } else {
    return false;
  }
}

template class StochasticDCA<DenseVector, DataSamples, LabelVector,
                             SparseVector, DenseVector>;
