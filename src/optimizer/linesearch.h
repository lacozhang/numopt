#pragma once

#ifndef __LINE_SEARCH_H__
#define __LINE_SEARCH_H__
#include <string>
#include <functional>
#include "../typedef.h"

enum class LineSearchFunctionType {
	None,
	BackTrack,
	MoreThuente
};

enum class LineSearchConditionType
{
	None,
	Armijo,
	Wolfe,
	StrongWolfe
};

class LineSearcher {
public:
	LineSearcher(const std::string& lsfuncstr, const std::string& lscondstr, int maxtries,
		float alpha=1e-4, float beta=0.9,
		double minstep=1e-15, double maxstep=1e15,
		double parameps=1e-15);
	~LineSearcher() {}

	bool IsValid() {
		return valid_;
	}

	bool LineSearch(DenseVector& param, DenseVector& direc, DenseVector& grad, double finit, double& stepsize,
		std::function<double(DenseVector&, DenseVector&)>& funcgrad);

	bool BackTrackLineSearch(DenseVector& param, DenseVector& direc, DenseVector& grad, double finit, double& stepsize,
		std::function<double(DenseVector&, DenseVector&)>& funcgrad);

	bool MoreThuenteLineSearch(DenseVector& param, DenseVector& direc, DenseVector& grad, double finit, double& stepsize,
		std::function<double(DenseVector&, DenseVector&)>& funcgrad);

private:

	LineSearchFunctionType ParseLineSearchString(const std::string& str);
	LineSearchConditionType ParseLineSearchConditionString(const std::string& str);

	int maxtries_;
	int itercnt_;
	bool valid_;
	LineSearchFunctionType lsfunctype_;
	LineSearchConditionType lscondtype_;

	// alpha_ is used for armijo condition, should be relative small like (1e-4)
	// beta_ is used for curvature condition, should be large like (0.9)
	float alpha_, beta_;
	double minstep_, maxstep_;
	double parameps_;
	DenseVector tparam_;
};

#endif // !__LINE_SEARCH_H__
