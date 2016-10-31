#include <iostream>
#include <boost/make_shared.hpp>
#include "linearmodel.h"
#include "util.h"

LinearModel::LinearModel(LossFunc loss, size_t featdim, size_t numclasses) {

	switch (loss)
	{
	case Squared:
		loss_ = boost::make_shared<SquaredLoss>();
		break;
	case Hinge:
		loss_ = boost::make_shared<HingeLoss>();
		break;
	case Logistic:
		loss_ = boost::make_shared<LogLoss>();
		break;
	case SquaredHinge:
		loss_ = boost::make_shared<SquaredHingeLoss>();
		break;
	default:
		break;
	}

	
}