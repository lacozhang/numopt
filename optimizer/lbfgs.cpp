#include <functional>
#include <algorithm>
#include <boost/make_shared.hpp>
#include "lbfgs.h"
#include "../util.h"


template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void LBFGS<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::InitFromCmd(int argc, const char * argv[])
{
	boost::program_options::options_description alldesc;
	alldesc.add(this->basedesc_);
	alldesc.add(this->lbfgsdesc_);

	auto vm = ParseArgs(argc, argv, alldesc, true);
	if (this->historycnt_ < 1) {
		BOOST_LOG_TRIVIAL(error) << "History count less than 1";
		return;
	}
	this->gradhistory_.resize(this->historycnt_);
	this->paramhistory_.resize(this->historycnt_);
	this->alphas_.resize(this->historycnt_, 0);
	this->betas_.resize(this->historycnt_, 0);
	this->rhos_.resize(this->historycnt_, 0);
	itercnt_ = 0;
	lsearch_.reset(new LineSearcher(this->lsfuncstr_, this->lsconfstr_, this->learn_.maxlinetries_));
	if (lsearch_.get() == nullptr) {
		BOOST_LOG_TRIVIAL(fatal) << "Can't allocate object";
		return;
	}
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void LBFGS<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::Train()
{
	itercnt_ = 1;
	ParameterType& param = this->model_.GetParameters();
	ParameterType pastparam;
	ParameterType paramdiff;

	ParameterType direction;

	DenseGradientType grad;
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

	funcval = EvaluateValueAndGrad(param, grad);
	paramnorm = param.norm();
	BOOST_LOG_TRIVIAL(info) << "Param norm " << paramnorm;
	paramnorm = std::max(paramnorm, 1.0);
	gradnorm = grad.norm();
	BOOST_LOG_TRIVIAL(info) << "Gradient norm " << gradnorm;

	std::function<double(ParameterType&, DenseGradientType&)> evaluator = std::bind(&LBFGS<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::EvaluateValueAndGrad,
		this, std::placeholders::_1, std::placeholders::_2);

	if (gradnorm / paramnorm < this->learn_.gradeps_) {
		BOOST_LOG_TRIVIAL(info) << "param already been optimized";
		return;
	}

	direction = -grad;
	stepsize = 1;

	for (int i = 0; i < this->historycnt_; ++i) {
		this->gradhistory_[i].resize(param.size());
		this->paramhistory_[i].resize(param.size());
	}

	while (itercnt_ <= this->learn_.maxiter_) {
		BOOST_LOG_TRIVIAL(info) << "*******Start iteration " << itercnt_ << "*******";
		pastparam = param;
		pastgrad = grad;

		if (this->learn_.l1_ == 0) {
			lsgood = lsearch_->BackTrackLineSearch(param, direction, grad, funcval, stepsize, evaluator);
		}
		else {
			BOOST_ASSERT(false);
		}

		BOOST_LOG_TRIVIAL(info) << "using step size " << stepsize;

		if (!lsgood) {
			BOOST_LOG_TRIVIAL(fatal) << "line search failed revert";
			param = pastparam;
			break;
		}

		BOOST_ASSERT(!param.hasNaN());
		BOOST_ASSERT(!grad.hasNaN());

		funcval = EvaluateOnSet(this->trainiter_->GetAllData(), this->trainiter_->GetAllLabel());
		BOOST_LOG_TRIVIAL(info) << "Loss value " << funcval;
		if (this->testiter_->IsValid()) {
			EvaluateOnSet(this->testiter_->GetAllData(), this->testiter_->GetAllLabel());
		}

#ifdef _DEBUG
		BOOST_ASSERT(!param.hasNaN());
		BOOST_ASSERT(!grad.hasNaN());
		BOOST_ASSERT(!pastgrad.hasNaN());
		BOOST_ASSERT(!pastparam.hasNaN());
#endif // _DEBUG


		paramnorm = param.norm();
		BOOST_LOG_TRIVIAL(info) << "Param norm " << paramnorm;
		paramnorm = std::max(paramnorm, 1.0);
		gradnorm = grad.norm();
		BOOST_LOG_TRIVIAL(info) << "Gradient norm " << gradnorm;
		if (gradnorm / paramnorm < this->learn_.gradeps_) {
			BOOST_LOG_TRIVIAL(info) << "optimization finished";
			break;
		}

		paramdiff = param - pastparam;
		gradiff = grad - pastgrad;
		this->gradhistory_[index].swap(gradiff);
		this->paramhistory_[index].swap(paramdiff);
		this->rhos_[index] = paramdiff.dot(gradiff);
		double scalar = this->rhos_[index] / gradiff.dot(gradiff);
		BOOST_LOG_TRIVIAL(info) << "scaling factor " << scalar;
#ifdef _DEBUG
		BOOST_ASSERT(!std::isnan(scalar));
#endif // 


		int bound = itercnt_ >= this->historycnt_ ? this->historycnt_ : itercnt_;
		index = (index + 1) % this->historycnt_;
		++itercnt_;

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
			direction += (this->alphas_[j] - this->betas_[j])* this->paramhistory_[j];
#ifdef _DEBUG
			BOOST_ASSERT(!direction.hasNaN());
#endif // 
			j = (j + 1) % this->historycnt_;
		}

		if (this->learn_.l2_ > 0) {
			funcval += 0.5 * this->learn_.l2_ * param.dot(param);
		}

		if (this->learn_.l1_ > 0) {
			BOOST_ASSERT(false);
			funcval += this->learn_.l1_ * param.lpNorm<1>();
		}

		BOOST_ASSERT(!direction.hasNaN());
		BOOST_LOG_TRIVIAL(info) << "new direction norm " << direction.norm();
		stepsize = 1.0;
	}
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
boost::program_options::options_description LBFGS<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::Options()
{
	boost::program_options::options_description combined;
	combined.add(this->basedesc_);
	combined.add(this->lbfgsdesc_);
	return combined;
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void LBFGS<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::InitCmdDescription()
{
	this->lbfgsdesc_.add_options()
		(this->kLineSearchOption,
			boost::program_options::value<std::string>(&this->lsfuncstr_)->default_value("bt"), "line search function: bt(back tracking), mt(more theute)")
			(this->kHistoryOption,
				boost::program_options::value<int>(&this->historycnt_)->default_value(10), "count of available history")
				(this->kLineSearchStopOption,
					boost::program_options::value<std::string>(&this->lsconfstr_)->default_value("armijo"),
					"stop criteria of line search: armijo, wolfe, swolfe");
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void LBFGS<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::ResetState()
{
	this->lsfuncstr_.clear();
	this->lsconfstr_.clear();
	this->gradhistory_.clear();
	this->paramhistory_.clear();
	this->alphas_.clear();
	this->betas_.clear();
	this->rhos_.clear();
	this->historycnt_ = 0;
}


template class LBFGS<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>;
template class LBFGS<DenseVector, LccrfSamples, LccrfLabels, SparseVector, DenseVector>;