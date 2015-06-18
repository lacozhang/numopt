#ifndef __LINEAR_MODEL_OPT_H__
#define __LINEAR_MODEL_OPT_H__

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include "parameter.h"
#include "lossfunc.h"


/*
  from optimization perspective, the only different between linear svm/ logistic regression is just about the loss function, we comine them here.
 */

class linearmodelopt {
	
public:
	linearmodelopt(Parameter& param,
		       boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& insts,
		       boost::shared_ptr<Eigen::VectorXi>& labels);
	
	double gettrainaccu(boost::shared_ptr<column_vector>& w);
	
	// used to construct function value object
	class funcval {

		friend class linearmodelopt;
	private:
		funcval(Parameter::LossFunc loss,
			double l1c,
			double l2c,
			boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& insts,
			boost::shared_ptr<Eigen::VectorXi>& labels);
		
		boost::shared_ptr<lossbase> loss_;
		double l1c_, l2c_;
		boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& insts_;
		boost::shared_ptr<Eigen::VectorXi>& labels_;
	public:
		double operator()(const column_vector& w) const;
	};

	class derivaval {
		
		friend class linearmodelopt;
	private:
		derivaval(Parameter::LossFunc loss,
			  double L1c,
			  double l2c,
			  boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& insts,
			  boost::shared_ptr<Eigen::VectorXi>& labels);
		
		boost::shared_ptr<lossbase> loss_;
		double l1c_, l2c_;
		boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& insts_;
		boost::shared_ptr<Eigen::VectorXi>& labels_;
	public:
		const column_vector operator()(const column_vector& w) const;
	};

	linearmodelopt::funcval* getFuncValObj();
	linearmodelopt::derivaval* getDerivaValObj();

private:
	Parameter& param_;
	boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& insts_;
	boost::shared_ptr<Eigen::VectorXi>& labels_;
};

#endif // __LINEAR_MODEL_OPT_H__
