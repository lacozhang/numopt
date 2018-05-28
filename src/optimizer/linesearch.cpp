#include "linesearch.h"
#include <boost/log/trivial.hpp>

namespace {

#define fsigndiff(x, y) (*(x) * (*(y) / fabs(*(y))) < 0.)

	/**
	* Define the local variables for computing minimizers.
	*/
#define USES_MINIMIZER \
    double a, d, gamma, theta, p, q, r, s;

	/**
	* Find a minimizer of an interpolated cubic function.
	*  @param  cm      The minimizer of the interpolated cubic.
	*  @param  u       The value of one point, u.
	*  @param  fu      The value of f(u).
	*  @param  du      The value of f'(u).
	*  @param  v       The value of another point, v.
	*  @param  fv      The value of f(v).
	*  @param  du      The value of f'(v).
	*/
#define CUBIC_MINIMIZER(cm, u, fu, du, v, fv, dv) \
    d = (v) - (u); \
    theta = ((fu) - (fv)) * 3 / d + (du) + (dv); \
    p = fabs(theta); \
    q = fabs(du); \
    r = fabs(dv); \
    s = std::max(std::max(p, q), r); \
    /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */ \
    a = theta / s; \
    gamma = s * sqrt(a * a - ((du) / s) * ((dv) / s)); \
    if ((v) < (u)) gamma = -gamma; \
    p = gamma - (du) + theta; \
    q = gamma - (du) + gamma + (dv); \
    r = p / q; \
    (cm) = (u) + r * d;

	/**
	* Find a minimizer of an interpolated cubic function.
	*  @param  cm      The minimizer of the interpolated cubic.
	*  @param  u       The value of one point, u.
	*  @param  fu      The value of f(u).
	*  @param  du      The value of f'(u).
	*  @param  v       The value of another point, v.
	*  @param  fv      The value of f(v).
	*  @param  du      The value of f'(v).
	*  @param  xmin    The maximum value.
	*  @param  xmin    The minimum value.
	*/
#define CUBIC_MINIMIZER2(cm, u, fu, du, v, fv, dv, xmin, xmax) \
    d = (v) - (u); \
    theta = ((fu) - (fv)) * 3 / d + (du) + (dv); \
    p = fabs(theta); \
    q = fabs(du); \
    r = fabs(dv); \
    s = std::max(std::max(p, q), r); \
    /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */ \
    a = theta / s; \
    gamma = s * sqrt(std::max(0.0, a * a - ((du) / s) * ((dv) / s))); \
    if ((u) < (v)) gamma = -gamma; \
    p = gamma - (dv) + theta; \
    q = gamma - (dv) + gamma + (du); \
    r = p / q; \
    if (r < 0. && gamma != 0.) { \
        (cm) = (v) - r * d; \
    } else if (a < 0) { \
        (cm) = (xmax); \
    } else { \
        (cm) = (xmin); \
    }

	/**
	* Find a minimizer of an interpolated quadratic function.
	*  @param  qm      The minimizer of the interpolated quadratic.
	*  @param  u       The value of one point, u.
	*  @param  fu      The value of f(u).
	*  @param  du      The value of f'(u).
	*  @param  v       The value of another point, v.
	*  @param  fv      The value of f(v).
	*/
#define QUARD_MINIMIZER(qm, u, fu, du, v, fv) \
    a = (v) - (u); \
    (qm) = (u) + (du) / (((fu) - (fv)) / a + (du)) / 2 * a;

	/**
	* Find a minimizer of an interpolated quadratic function.
	*  @param  qm      The minimizer of the interpolated quadratic.
	*  @param  u       The value of one point, u.
	*  @param  du      The value of f'(u).
	*  @param  v       The value of another point, v.
	*  @param  dv      The value of f'(v).
	*/
#define QUARD_MINIMIZER2(qm, u, du, v, dv) \
    a = (u) - (v); \
    (qm) = (v) + (dv) / ((dv) - (du)) * a;

	/**
	* Update a safeguarded trial value and interval for line search.
	*
	*  The parameter x represents the step with the least function value.
	*  The parameter t represents the current step. This function assumes
	*  that the derivative at the point of x in the direction of the step.
	*  If the bracket is set to true, the minimizer has been bracketed in
	*  an interval of uncertainty with endpoints between x and y.
	*
	*  @param  x       The pointer to the value of one endpoint.
	*  @param  fx      The pointer to the value of f(x).
	*  @param  dx      The pointer to the value of f'(x).
	*  @param  y       The pointer to the value of another endpoint.
	*  @param  fy      The pointer to the value of f(y).
	*  @param  dy      The pointer to the value of f'(y).
	*  @param  t       The pointer to the value of the trial value, t.
	*  @param  ft      The pointer to the value of f(t).
	*  @param  dt      The pointer to the value of f'(t).
	*  @param  tmin    The minimum value for the trial value, t.
	*  @param  tmax    The maximum value for the trial value, t.
	*  @param  brackt  The pointer to the predicate if the trial value is
	*                  bracketed.
	*  @retval int     Status value. Zero indicates a normal termination.
	*
	*  @see
	*      Jorge J. More and David J. Thuente. Line search algorithm with
	*      guaranteed sufficient decrease. ACM Transactions on Mathematical
	*      Software (TOMS), Vol 20, No 3, pp. 286-307, 1994.
	*/
	static int update_trial_interval(
		double *x,
		double *fx,
		double *dx,
		double *y,
		double *fy,
		double *dy,
		double *t,
		double *ft,
		double *dt,
		const double tmin,
		const double tmax,
		int *brackt
	)
	{
		int bound;
		int dsign = fsigndiff(dt, dx);
		double mc; /* minimizer of an interpolated cubic. */
		double mq; /* minimizer of an interpolated quadratic. */
		double newt;   /* new trial value. */
		USES_MINIMIZER;     /* for CUBIC_MINIMIZER and QUARD_MINIMIZER. */

							/* Check the input parameters for errors. */
		if (*brackt) {
			if (*t <= std::min(*x, *y) || std::max(*x, *y) <= *t) {
				/* The trival value t is out of the interval. */
				return -1;
			}
			if (0. <= *dx * (*t - *x)) {
				/* The function must decrease from x. */
				return -1;
			}
			if (tmax < tmin) {
				/* Incorrect tmin and tmax specified. */
				return -1;
			}
		}

		/*
		Trial value selection.
		*/
		if (*fx < *ft) {
			/*
			Case 1: a higher function value.
			The minimum is brackt. If the cubic minimizer is closer
			to x than the quadratic one, the cubic one is taken, else
			the average of the minimizers is taken.
			*/
			*brackt = 1;
			bound = 1;
			CUBIC_MINIMIZER(mc, *x, *fx, *dx, *t, *ft, *dt);
			QUARD_MINIMIZER(mq, *x, *fx, *dx, *t, *ft);
			if (fabs(mc - *x) < fabs(mq - *x)) {
				newt = mc;
			}
			else {
				newt = mc + 0.5 * (mq - mc);
			}
		}
		else if (dsign) {
			/*
			Case 2: a lower function value and derivatives of
			opposite sign. The minimum is brackt. If the cubic
			minimizer is closer to x than the quadratic (secant) one,
			the cubic one is taken, else the quadratic one is taken.
			*/
			*brackt = 1;
			bound = 0;
			CUBIC_MINIMIZER(mc, *x, *fx, *dx, *t, *ft, *dt);
			QUARD_MINIMIZER2(mq, *x, *dx, *t, *dt);
			if (fabs(mc - *t) > fabs(mq - *t)) {
				newt = mc;
			}
			else {
				newt = mq;
			}
		}
		else if (fabs(*dt) < fabs(*dx)) {
			/*
			Case 3: a lower function value, derivatives of the
			same sign, and the magnitude of the derivative decreases.
			The cubic minimizer is only used if the cubic tends to
			infinity in the direction of the minimizer or if the minimum
			of the cubic is beyond t. Otherwise the cubic minimizer is
			defined to be either tmin or tmax. The quadratic (secant)
			minimizer is also computed and if the minimum is brackt
			then the the minimizer closest to x is taken, else the one
			farthest away is taken.
			*/
			bound = 1;
			CUBIC_MINIMIZER2(mc, *x, *fx, *dx, *t, *ft, *dt, tmin, tmax);
			QUARD_MINIMIZER2(mq, *x, *dx, *t, *dt);
			if (*brackt) {
				if (fabs(*t - mc) < fabs(*t - mq)) {
					newt = mc;
				}
				else {
					newt = mq;
				}
			}
			else {
				if (fabs(*t - mc) > fabs(*t - mq)) {
					newt = mc;
				}
				else {
					newt = mq;
				}
			}
		}
		else {
			/*
			Case 4: a lower function value, derivatives of the
			same sign, and the magnitude of the derivative does
			not decrease. If the minimum is not brackt, the step
			is either tmin or tmax, else the cubic minimizer is taken.
			*/
			bound = 0;
			if (*brackt) {
				CUBIC_MINIMIZER(newt, *t, *ft, *dt, *y, *fy, *dy);
			}
			else if (*x < *t) {
				newt = tmax;
			}
			else {
				newt = tmin;
			}
		}

		/*
		Update the interval of uncertainty. This update does not
		depend on the new step or the case analysis above.

		- Case a: if f(x) < f(t),
		x <- x, y <- t.
		- Case b: if f(t) <= f(x) && f'(t)*f'(x) > 0,
		x <- t, y <- y.
		- Case c: if f(t) <= f(x) && f'(t)*f'(x) < 0,
		x <- t, y <- x.
		*/
		if (*fx < *ft) {
			/* Case a */
			*y = *t;
			*fy = *ft;
			*dy = *dt;
		}
		else {
			/* Case c */
			if (dsign) {
				*y = *x;
				*fy = *fx;
				*dy = *dx;
			}
			/* Cases b and c */
			*x = *t;
			*fx = *ft;
			*dx = *dt;
		}

		/* Clip the new trial value in [tmin, tmax]. */
		if (tmax < newt) newt = tmax;
		if (newt < tmin) newt = tmin;

		/*
		Redefine the new trial value if it is close to the upper bound
		of the interval.
		*/
		if (*brackt && bound) {
			mq = *x + 0.66 * (*y - *x);
			if (*x < *y) {
				if (mq < newt) newt = mq;
			}
			else {
				if (newt < mq) newt = mq;
			}
		}

		/* Return the new trial value. */
		*t = newt;
		return 0;
	}
}


LineSearcher::LineSearcher(const std::string & lsfuncstr, const std::string & lscondstr, int maxtries,
	float alpha, float beta,
	double minstep, double maxstep,
	double parameps)
	: valid_(false), maxtries_(maxtries), alpha_(alpha), beta_(beta), minstep_(minstep), maxstep_(maxstep), parameps_(parameps)
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

bool LineSearcher::LineSearch(DenseVector& param, DenseVector& direc, DenseVector& grad, double finit, double& stepsize, std::function<double(DenseVector&, DenseVector&)>& funcgrad)
{
	if (lsfunctype_ == LineSearchFunctionType::BackTrack) {
		return BackTrackLineSearch(param, direc, grad, finit, stepsize, funcgrad);
	}
	else {
		return MoreThuenteLineSearch(param, direc, grad, finit, stepsize, funcgrad);
	}
}

bool LineSearcher::BackTrackLineSearch(DenseVector& param, DenseVector& direc, DenseVector& grad, double finit, double& stepsize,
	std::function<double(DenseVector&, DenseVector&)>& funcgrad)
{
	itercnt_ = 0;
	double stepupdate;
	double dginit = direc.dot(grad), dgtest, fval, dgval;
	const double stepshrink = 0.5, stepexpand = 2.1;
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

			dgval = direc.dot(grad);
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

bool LineSearcher::MoreThuenteLineSearch(DenseVector& param, DenseVector& direc, DenseVector& grad, double finit, double & stepsize, 
	std::function<double(DenseVector&, DenseVector&)>& funcgrad)
{

	itercnt_ = 0;
	int brackt, stage1, uinfo = 0;
	double dg;
	double stx, fx, dgx;
	double sty, fy, dgy;
	double fxm, dgxm, fym, dgym, fm, dgm;
	double ftest1, dginit, dgtest;
	double width, prev_width;
	double stmin, stmax;
	double fval;

	if (stepsize < 0) {
		BOOST_LOG_TRIVIAL(fatal) << "Stepsize less than 0";
		return false;
	}

	dginit = direc.dot(grad);
	if (dginit > 0) {
		BOOST_LOG_TRIVIAL(fatal) << "Direction not decent";
		return false;
	}

	if (tparam_.size() != param.size()) {
		tparam_.resize(param.size());
	}

	/* Initialize local variables. */
	brackt = 0;
	stage1 = 1;
	dgtest = alpha_ * dginit;
	width = maxstep_ - minstep_;
	prev_width = 2.0 * width;

	stx = sty = 0.;
	fx = fy = finit;
	dgx = dgy = dginit;

	while (itercnt_ < maxtries_)
	{
		/*
		Set the minimum and maximum steps to correspond to the
		present interval of uncertainty.
		*/
		if (brackt) {
			stmin = std::min(stx, sty);
			stmax = std::min(stx, sty);
		}
		else {
			stmin = stx;
			stmax = stepsize + 4.0 * (stepsize - stx);
		}

		/* Clip the step in the range of [minstep_, maxstep_]. */
		if (stepsize < minstep_) stepsize = minstep_;
		if (stepsize > maxstep_) stepsize = maxstep_;

		/*
		If an unusual termination is to occur then let
		stepsize be the lowest point obtained so far.
		*/
		if ((brackt && ((stepsize <= stmin || stepsize >= stmax) || (itercnt_ + 1 >= maxtries_) || uinfo != 0)) ||
			(brackt && (stmax - stmin <= parameps_*stmax))) {
			stepsize = stx;
		}

		tparam_ = param + stepsize*direc;
		fval = funcgrad(tparam_, grad);
		dg = grad.dot(direc);

		ftest1 = finit + stepsize*dgtest;
		++itercnt_;

		/* Test for errors and convergence. */
		if (brackt && ((stepsize <= stmin || stmax <= stepsize) || uinfo != 0)) {
			/* Rounding errors prevent further progress. */
			return false;
		}
		if (stepsize == maxstep_ && fval <= ftest1 && dg <= dgtest) {
			/* The step is the maximum value. */
			return false;
		}
		if (stepsize == minstep_ && (ftest1 < fval || dgtest <= dg)) {
			/* The step is the minimum value. */
			return false;
		}
		if (brackt && (stmax - stmin) <= parameps_ * stmax) {
			/* Relative width of the interval of uncertainty is at most xtol. */
			return false;
		}
		if (maxtries_ <= itercnt_) {
			/* Maximum number of iteration. */
			return false;
		}

		if (fval <= ftest1 && std::fabs(dg) <= beta_ * (-dginit)) {
			/* The sufficient decrease condition and the directional derivative condition hold. */
			param.swap(tparam_);
			return true;
		}

		/*
		In the first stage we seek a step for which the modified
		function has a nonpositive value and nonnegative derivative.
		*/
		if (stage1 && fval <= ftest1 && std::min(alpha_, beta_) * dginit <= dg) {
			stage1 = 0;
		}

		/*
		A modified function is used to predict the step only if
		we have not obtained a step for which the modified
		function has a nonpositive function value and nonnegative
		derivative, and if a lower function value has been
		obtained but the decrease is not sufficient.
		*/
		if (stage1 && ftest1 < fval && fval <= fx) {
			/* Define the modified function and derivative values. */
			fm = fval - stepsize * dgtest;
			fxm = fx - stx * dgtest;
			fym = fy - sty * dgtest;
			dgm = dg - dgtest;
			dgxm = dgx - dgtest;
			dgym = dgy - dgtest;

			/*
			Call update_trial_interval() to update the interval of
			uncertainty and to compute the new step.
			*/
			uinfo = update_trial_interval(
				&stx, &fxm, &dgxm,
				&sty, &fym, &dgym,
				&stepsize, &fm, &dgm,
				stmin, stmax, &brackt
			);

			/* Reset the function and gradient values for f. */
			fx = fxm + stx * dgtest;
			fy = fym + sty * dgtest;
			dgx = dgxm + dgtest;
			dgy = dgym + dgtest;
		}
		else {
			/*
			Call update_trial_interval() to update the interval of
			uncertainty and to compute the new step.
			*/
			uinfo = update_trial_interval(
				&stx, &fx, &dgx,
				&sty, &fy, &dgy,
				&stepsize, &fval, &dg,
				stmin, stmax, &brackt
			);
		}

		/*
		Force a sufficient decrease in the interval of uncertainty.
		*/
		if (brackt) {
			if (0.66 * prev_width <= fabs(sty - stx)) {
				stepsize = stx + 0.5 * (sty - stx);
			}
			prev_width = width;
			width = std::fabs(sty - stx);
		}
	}

	return false;
}

LineSearchFunctionType LineSearcher::ParseLineSearchString(const std::string & str)
{
	if ("bt" == str) {
		BOOST_LOG_TRIVIAL(info) << "Using backtrack line search";
		return LineSearchFunctionType::BackTrack;
	}
	else if ("mt" == str) {
		BOOST_LOG_TRIVIAL(info) << "Using more thuente line search";
		return LineSearchFunctionType::MoreThuente;
	}
	else {
		return LineSearchFunctionType::None;
	}
}

LineSearchConditionType LineSearcher::ParseLineSearchConditionString(const std::string& str)
{
	if ("armijo" == str) {
		BOOST_LOG_TRIVIAL(info) << "Using Armijo condition";
		return LineSearchConditionType::Armijo;
	}
	else if ("wolfe" == str) {
		BOOST_LOG_TRIVIAL(info) << "Using Wolfe Condition";
		return LineSearchConditionType::Wolfe;
	}
	else if ("swolfe" == str) {
		BOOST_LOG_TRIVIAL(info) << "Using Strong Wolfe Condition";
		return LineSearchConditionType::StrongWolfe;
	}
	else {
		return LineSearchConditionType::None;
	}
}
