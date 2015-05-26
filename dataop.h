#include <string>
#include <boost/shared_ptr.hpp>
#include "typedef.h"

#ifndef __DATA_OP_H__
#define __DATA_OP_H__

void matrix_size_estimation(std::string featfile, Eigen::VectorXi& datsize, int& row, int& col);

void load_libsvm_data(std::string featfile,
	       boost::shared_ptr< DataSamples >& Samples,
	       boost::shared_ptr< ClsLabelVector >& labels);
#endif
