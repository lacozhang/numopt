#include "linesearch.h"
#include <boost/log/trivial.hpp>

LineSearcher::LineSearcher(const std::string & lsfuncstr, const std::string & lscondstr, int maxtries,
	float alpha, float beta,
	double minstep, double maxstep)
	: valid_(false), maxtries_(maxtries), alpha_(alpha), beta_(beta), minstep_(minstep), maxstep_(maxstep)
{
	if (maxtries_ < 1) {
		BOOST_LOG_TRIVIAL(fatal) << "For line search, maximum number of tries should larger than 0";
		return;
	}

	lsfunctype_ = ParseLineSearchString(lsfuncstr);
	lscondtype_ = ParseLineSearchConditionString(lscondstr);
	if (lsfunctype_ == LineSearchFunctionType::None ||
		lscondtype_ == LineSearchConditionType::None) {
		valid_ = false;
		BOOST_LOG_TRIVIAL(fatal) << "Line search function or condition not recognized : " << lsfuncstr << "/" << lscondstr;
		return;
	}
	else {
		valid_ = true;
	}
}

bool LineSearcher::BackTrackLineSearch(DenseVector& param, DenseVector& direc, DenseVector& grad, double finit, double& stepsize,
	std::function<double(DenseVector&, DenseVector&)> funcgrad)
{
	itercnt_ = 0;
	float stepupdate;
	double dginit = direc.dot(grad), dgtest, fval, dgval;
	const float stepshrink = 0.5, stepexpand = 2.1;
	if (dginit > 0) {
		BOOST_LOG_TRIVIAL(fatal) << "initial direction is not a decent direction";
		return false;
	}

	if (tparam_.size() != param.size()) {
		tparam_.resize(param.size());
	}

	dgtest = dginit * alpha_;
	while (itercnt_ < maxtries_)
	{
		tparam_ = param + stepsize * direc;
		fval = funcgrad(tparam_, grad);

		if (fval > finit + stepsize*dgtest) {
			stepupdate = stepshrink;
		}
		else {
			if (lscondtype_ == LineSearchConditionType::Armijo) {
				break;
			}

			dgval = tparam_.dot(grad);
			if (dgval < beta_ * dginit) {
				stepupdate = stepexpand;
			}
			else {
				if (lscondtype_ == LineSearchConditionType::Wolfe) {
					break;
				}

				if (dgval > -beta_*dginit) {
					stepupdate = stepshrink;
				}
				else {
					break;
				}
			}
		}

		if (stepsize < minstep_) {
			BOOST_LOG_TRIVIAL(error) << "Small than smallest step size";
			return false;
		}

		if (stepsize > maxstep_) {
			BOOST_LOG_TRIVIAL(error) << "Large than largest step size";
			return false;
		}

		stepsize *= stepupdate;
		++itercnt_;
	}

	if (itercnt_ >= maxtries_) {
		BOOST_LOG_TRIVIAL(error) << "Exceed Maximum number of iteration count";
		return false;
	}

	param.swap(tparam_);
	return true;
}

LineSearchFunctionType LineSearcher::ParseLineSearchString(const std::string & str)
{
	if ("bt" == str) {
		return LineSearchFunctionType::BackTrack;
	}
	else if ("mt" == str) {
		return LineSearchFunctionType::MoreThuente;
	}
	else {
		return LineSearchFunctionType::None;
	}
}

LineSearchConditionType LineSearcher::ParseLineSearchConditionString(const std::string& str)
{
	if ("armijo" == str) {
		return LineSearchConditionType::Armijo;
	}
	else if ("wolfe" == str) {
		return LineSearchConditionType::Wolfe;
	}
	else if ("swolfe" == str) {
		return LineSearchConditionType::StrongWolfe;
	}
	else {
		return LineSearchConditionType::None;
	}
}
