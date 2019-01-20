#include "optimizer/lbfgs.h"
#include "util/util.h"
#include <algorithm>
#include <boost/make_shared.hpp>
#include <functional>

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
void LBFGS<ParameterType, SampleType, LabelType, SparseGradientType,
           DenseGradientType>::InitFromCmd(int argc, const char *argv[]) {
  boost::program_options::options_description alldesc;
  alldesc.add(this->basedesc_);
  alldesc.add(this->lbfgsdesc_);

  auto vm = ParseArgs(argc, argv, alldesc, true);
  if (this->historycnt_ < 1) {
    LOG(ERROR) << "History count less than 1";
    return;
  }
  this->gradhistory_.resize(this->historycnt_);
  this->paramhistory_.resize(this->historycnt_);
  this->alphas_.resize(this->historycnt_, 0);
  this->betas_.resize(this->historycnt_, 0);
  this->rhos_.resize(this->historycnt_, 0);
  itercnt_ = 0;
  lsearch_.reset(new LineSearcher(this->lsfuncstr_, this->lsconfstr_,
                                  this->learn_.maxlinetries_));
  if (lsearch_.get() == nullptr) {
    LOG(FATAL) << "Can't allocate object";
    return;
  }
}

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
void LBFGS<ParameterType, SampleType, LabelType, SparseGradientType,
           DenseGradientType>::Train() {
  itercnt_ = 1;
  ParameterType &param = this->model_.GetParameters();
  ParameterType pastparam;
  ParameterType paramdiff;

  ParameterType direction;

  DenseGradientType grad;
  DenseGradientType projgrad;
  DenseGradientType pastgrad;
  DenseGradientType gradiff;

  double funcval, paramnorm, gradnorm, stepsize;
  bool lsgood = false;
  int index = 0;

  pastparam.resize(param.size());
  pastparam.setZero();
  paramdiff.resize(param.size());
  paramdiff.setZero();
  grad.resize(param.size());
  grad.setZero();
  pastgrad.resize(param.size());
  pastgrad.setZero();
  gradiff.resize(param.size());
  gradiff.setZero();
  direction.resize(param.size());
  direction.setZero();
  if (this->learn_.l1_ > 0) {
    projgrad.resize(param.size());
    projgrad.setZero();
    workorthant_.resize(param.size());
    workorthant_.setZero();
    paramnew_.resize(param.size());
    paramnew_.setZero();
  }

  funcval = EvaluateValueAndGrad(param, grad);
  paramnorm = param.norm();
  LOG(INFO) << "Param norm " << paramnorm;
  paramnorm = std::max(paramnorm, 1.0);

  if (this->learn_.l1_ > 0) {
    OwlqnPGradient(param, grad, projgrad);
    gradnorm = projgrad.norm();
  } else {
    gradnorm = grad.norm();
  }
  LOG(INFO) << "gradient norm " << gradnorm;

  std::function<double(ParameterType &, DenseGradientType &)> evaluator =
      std::bind(&LBFGS<ParameterType, SampleType, LabelType, SparseGradientType,
                       DenseGradientType>::EvaluateValueAndGrad,
                this, std::placeholders::_1, std::placeholders::_2);

  if (gradnorm / paramnorm < this->learn_.gradeps_) {
    LOG(INFO) << "param already been optimized";
    return;
  }

  if (this->learn_.l1_ > 0) {
    direction = -projgrad;
  } else {
    direction = -grad;
  }

  stepsize = 1 / direction.dot(direction);

  for (int i = 0; i < this->historycnt_; ++i) {
    this->gradhistory_[i].resize(param.size());
    this->paramhistory_[i].resize(param.size());
  }

  while (itercnt_ <= this->learn_.maxiter_) {
    LOG(INFO) << "*******Start iteration " << itercnt_ << "*******";
    pastparam = param;
    pastgrad = grad;

    LOG(INFO) << "objective value " << funcval;
    if (this->learn_.l1_ == 0) {
      lsgood = lsearch_->LineSearch(param, direction, grad, funcval, stepsize,
                                    evaluator);
    } else {
      lsgood = BackTrackForOWLQN(param, funcval, direction, grad, projgrad,
                                 stepsize);
      OwlqnPGradient(param, grad, projgrad);
    }

    LOG(INFO) << "using step size " << stepsize;

    if (!lsgood) {
      LOG(FATAL) << "line search failed revert";
      param = pastparam;
      break;
    }

    BOOST_ASSERT(!param.hasNaN());
    BOOST_ASSERT(!grad.hasNaN());

    funcval = OptMethodBaseType::EvaluateOnSet(this->trainiter_->GetAllData(),
                                               this->trainiter_->GetAllLabel());
    if (this->testiter_->IsValid()) {
      OptMethodBaseType::EvaluateOnSet(this->testiter_->GetAllData(),
                                       this->testiter_->GetAllLabel());
    }

#ifdef _DEBUG
    BOOST_ASSERT(!param.hasNaN());
    BOOST_ASSERT(!grad.hasNaN());
    BOOST_ASSERT(!pastgrad.hasNaN());
    BOOST_ASSERT(!pastparam.hasNaN());
#endif // _DEBUG

    paramnorm = param.norm();
    LOG(INFO) << "Param norm " << paramnorm;
    paramnorm = std::max(paramnorm, 1.0);
    if (this->learn_.l1_ > 0) {
      gradnorm = projgrad.norm();
    } else {
      gradnorm = grad.norm();
    }

    LOG(INFO) << "Gradient norm " << gradnorm;
    if (gradnorm / paramnorm < this->learn_.gradeps_) {
      LOG(INFO) << "optimization finished";
      break;
    }

    paramdiff = param - pastparam;
    gradiff = grad - pastgrad;
    this->gradhistory_[index].swap(gradiff);
    this->paramhistory_[index].swap(paramdiff);
    this->rhos_[index] =
        this->gradhistory_[index].dot(this->paramhistory_[index]);
    double scalar = this->rhos_[index] /
                    this->gradhistory_[index].dot(this->gradhistory_[index]);
#ifdef _DEBUG
    BOOST_ASSERT(!std::isnan(scalar));
#endif //

    int bound = itercnt_ >= this->historycnt_ ? this->historycnt_ : itercnt_;
    index = (index + 1) % this->historycnt_;
    ++itercnt_;

    if (this->learn_.l1_ > 0)
      direction = -projgrad;
    else
      direction = -grad;

    int i = 0, j = 0;
    for (i = 0, j = index; i < bound; ++i) {
      j = (j + this->historycnt_ - 1) % this->historycnt_;
      this->alphas_[j] = this->paramhistory_[j].dot(direction);
      this->alphas_[j] /= this->rhos_[j];
      direction -= this->alphas_[j] * this->gradhistory_[j];
#ifdef _DEBUG
      BOOST_ASSERT(!direction.hasNaN());
#endif //
    }

    direction *= scalar;

    for (int i = 0; i < bound; ++i) {
      this->betas_[j] = this->gradhistory_[j].dot(direction);
      this->betas_[j] /= this->rhos_[j];
      direction +=
          (this->alphas_[j] - this->betas_[j]) * this->paramhistory_[j];
#ifdef _DEBUG
      BOOST_ASSERT(!direction.hasNaN());
#endif //
      j = (j + 1) % this->historycnt_;
    }

    if (this->learn_.l2_ > 0) {
      funcval += 0.5 * this->learn_.l2_ * param.dot(param);
    }

    if (this->learn_.l1_ > 0) {
      funcval += this->learn_.l1_ * param.template lpNorm<1>();
      for (int i = 0; i < param.size(); ++i) {
        if (direction.coeff(i) * projgrad.coeff(i) >= 0) {
          direction.coeffRef(i) = 0.0;
        }
      }
    }

    LOG(INFO) << "new direction norm " << direction.norm();
    stepsize = 1.0;
  }
  OptMethodBaseType::ResultStats(param);
}

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
boost::program_options::options_description
LBFGS<ParameterType, SampleType, LabelType, SparseGradientType,
      DenseGradientType>::Options() {
  boost::program_options::options_description combined;
  combined.add(this->basedesc_);
  combined.add(this->lbfgsdesc_);
  return combined;
}

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
bool LBFGS<ParameterType, SampleType, LabelType, SparseGradientType,
           DenseGradientType>::BackTrackForOWLQN(ParameterType &oriparam,
                                                 double finit,
                                                 const ParameterType &direc,
                                                 DenseGradientType &origrad,
                                                 DenseGradientType &oripgrad,
                                                 double &stepsize) {
  double funcval;
  if (stepsize < 0) {
    LOG(FATAL) << "error, step size smaller than 0";
    return false;
  }
  for (int i = 0; i < oriparam.size(); ++i) {
    if (oriparam.coeff(i) == 0) {
      workorthant_.coeffRef(i) = -oripgrad.coeff(i);
    } else {
      workorthant_.coeffRef(i) = oriparam.coeff(i);
    }
  }
  int iter = 0;
  while (iter < this->learn_.maxlinetries_) {
    paramnew_ = oriparam + stepsize * direc;
    OWLQNProject(paramnew_, workorthant_);
    funcval = EvaluateValueAndGrad(paramnew_, origrad);
    double dgtest = oripgrad.dot(paramnew_ - oriparam);
    if (funcval <= finit + 1e-4 * dgtest) {
      break;
    }

    if (stepsize < 1e-15) {
      LOG(ERROR) << "stepsize too small";
      return false;
    }

    stepsize *= 0.5;
    ++iter;
  }

  if (iter >= this->learn_.maxlinetries_) {
    LOG(ERROR) << "exceed maximum number of tries";
    return false;
  }

  oriparam.swap(paramnew_);
  return true;
}

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
void LBFGS<ParameterType, SampleType, LabelType, SparseGradientType,
           DenseGradientType>::InitCmdDescription() {
  this->lbfgsdesc_.add_options()(
      this->kLineSearchOption,
      boost::program_options::value<std::string>(&this->lsfuncstr_)
          ->default_value("bt"),
      "line search function: bt(back tracking), mt(more theute)")(
      this->kHistoryOption,
      boost::program_options::value<int>(&this->historycnt_)->default_value(10),
      "count of available history")(
      this->kLineSearchStopOption,
      boost::program_options::value<std::string>(&this->lsconfstr_)
          ->default_value("armijo"),
      "stop criteria of line search: armijo, wolfe, swolfe");
}

template <class ParameterType, class SampleType, class LabelType,
          class SparseGradientType, class DenseGradientType>
void LBFGS<ParameterType, SampleType, LabelType, SparseGradientType,
           DenseGradientType>::ResetState() {
  this->lsfuncstr_.clear();
  this->lsconfstr_.clear();
  this->gradhistory_.clear();
  this->paramhistory_.clear();
  this->alphas_.clear();
  this->betas_.clear();
  this->rhos_.clear();
  this->historycnt_ = 0;
}

template class LBFGS<DenseVector, DataSamples, LabelVector, SparseVector,
                     DenseVector>;
template class LBFGS<DenseVector, LccrfSamples, LccrfLabels, SparseVector,
                     DenseVector>;
