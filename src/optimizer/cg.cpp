#include "cg.h"
#include "util/util.h"

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void ConjugateGradient<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::InitFromCmd(int argc, const char * argv[])
{
	boost::program_options::options_description alldesc;
	alldesc.add(this->basedesc_);
	alldesc.add(this->cgdesc_);

	auto vm = ParseArgs(argc, argv, alldesc, true);
	if (this->restartcounter_ < 1) {
		BOOST_LOG_TRIVIAL(fatal) << "small value will lead to poor performance";
		return;
	}

	BOOST_LOG_TRIVIAL(info) << "Restart iteration every " << this->restartcounter_ << " iterations";
	if ("fr" == this->methodstr_) {
		method_ = ConjugateGenerationMethod::FR;
	}
	else if("pr" == this->methodstr_) {
		method_ = ConjugateGenerationMethod::PR;
	}
	else {
		BOOST_LOG_TRIVIAL(info) << "Method " << this->methodstr_ << " can not be recognized, use default pr";
		method_ = ConjugateGenerationMethod::PR;
	}

	if (this->learn_.l1_ != 0) {
		BOOST_LOG_TRIVIAL(fatal) << "Error, Conjugate gradient can't handle l1 regularization, it will be ignored";
	}

	lsearch_.reset(new LineSearcher(this->lsfuncstr_, "swolfe", this->learn_.maxlinetries_));
	if (lsearch_.get() == nullptr) {
		BOOST_LOG_TRIVIAL(fatal) << "Failed to initialize line search";
	}
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void ConjugateGradient<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::Train()
{
	int itercnt=0;
	ParameterType& param = this->model_.GetParameters();
	ParameterType direction;
	ParameterType pastdirec;
	DenseGradientType grad;

	grad.resize(param.size());
	grad.setZero();
	pastgrad_.resize(param.size());
	pastgrad_.setZero();
	direction.resize(param.size());
	direction.setZero();
	pastdirec.resize(param.size());
	pastdirec.setZero();

	double funcval = 0, stepsize=1, beta;
	double paramnorm, gradnorm;
	std::function<double(ParameterType&, DenseGradientType&)> evaluator = std::bind(
		&ConjugateGradient<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::EvaluateValueAndGrad,
		this, std::placeholders::_1, std::placeholders::_2);
	funcval = EvaluateValueAndGrad(param, grad);
	direction = -grad;
	while (itercnt < this->learn_.maxiter_) {

		pastdirec = direction;
		BOOST_LOG_TRIVIAL(info) << "*******Start iteration " << itercnt << "*******";
		BOOST_LOG_TRIVIAL(info) << "Object value " << funcval;

		++itercnt;
		pastgrad_ = grad;
		bool lsgood = lsearch_->LineSearch(param, direction, grad, funcval, stepsize, evaluator);
		if (!lsgood) {
			BOOST_LOG_TRIVIAL(info) << "line search failed";
			break;
		}

		paramnorm = param.norm();
		paramnorm = std::max(1.0, paramnorm);
		gradnorm = grad.norm();

		BOOST_LOG_TRIVIAL(info) << "Step size      " << stepsize;
		BOOST_LOG_TRIVIAL(info) << "Gradient  norm " << gradnorm;
		BOOST_LOG_TRIVIAL(info) << "Parameter norm " << paramnorm;
		if (gradnorm / paramnorm < this->learn_.gradeps_) {
			BOOST_LOG_TRIVIAL(info) << "satisfy gradient epsilon, exit";
			break;
		}

		switch (method_)
		{
		case ConjugateGenerationMethod::FR:
		{
			beta = grad.dot(grad) / pastgrad_.dot(pastgrad_);
		}
		break;
		case ConjugateGenerationMethod::PR:
		{
			beta = grad.dot(grad - pastgrad_) / pastgrad_.dot(pastgrad_);
			beta = std::max(beta, 0.0);
		}
		break;
		}

		if (itercnt % this->restartcounter_ == 0) {
			beta = 0.0;
		}

		funcval = OptMethodBaseType::EvaluateOnSet(this->trainiter_->GetAllData(), this->trainiter_->GetAllLabel());
		if (this->testiter_->IsValid()) {
            OptMethodBaseType::EvaluateOnSet(this->testiter_->GetAllData(), this->testiter_->GetAllLabel());
		}

		if (this->learn_.l2_ > 0) {
			funcval += 0.5 * this->learn_.l2_ * param.dot(param);
		}

		if (beta != 0) {
			direction *= beta;
		}
		else {
			direction.setZero();
		}
		direction -= grad;
		BOOST_LOG_TRIVIAL(info) << "Orthognal rate " << pastdirec.dot(direction);


		stepsize = 4;
	}

    OptMethodBaseType::ResultStats(param);
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
boost::program_options::options_description ConjugateGradient<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::Options()
{
	boost::program_options::options_description alldesc("Overall options for Conjugate Gradient");
	alldesc.add(this->basedesc_);
	alldesc.add(this->cgdesc_);
	return std::move(alldesc);
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void ConjugateGradient<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::InitCmdDescription()
{
	this->cgdesc_.add_options()
		("cg.n", boost::program_options::value<int>(&this->restartcounter_)->default_value(20), "reset cg status after n iteration")
		("cg.m", boost::program_options::value<std::string>(&this->methodstr_)->default_value("pr"), "method for generation: fr/pr")
		("cg.ls", boost::program_options::value<std::string>(&this->lsfuncstr_)->default_value("bt"), "line search method: bt/mt");
}

template<class ParameterType, class SampleType, class LabelType, class SparseGradientType, class DenseGradientType>
void ConjugateGradient<ParameterType, SampleType, LabelType, SparseGradientType, DenseGradientType>::ResetState()
{
	this->restartcounter_ = 0;
}




template class ConjugateGradient<DenseVector, DataSamples, LabelVector, SparseVector, DenseVector>;
template class ConjugateGradient<DenseVector, LccrfSamples, LccrfLabels, SparseVector, DenseVector>;
