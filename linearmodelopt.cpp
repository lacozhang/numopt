#include <iostream>
#include "linearmodelopt.h"

namespace {

  lossbase* getLoss(Parameter::LossFunc l){
	  
	  switch(l){
	  case Parameter::Squared:
		  std::cerr << "squaredloss not implemented yet" << std::endl;
		  return NULL;
		  break;
	  case Parameter::Hinge:
		  return new HingeLoss();
		  break;
	  case Parameter::Logistic:
		  return new LogLoss();
		  break;
	  default:
		  std::cerr << "unsupported loss type" << std::endl;
		  return NULL;
	  }
  }

	// convert a column matrix from eigen3 format from dlib format
	void eigen2dlib(column_vector& to, Eigen::VectorXd& f){
		
		int rows = f.rows();
		int cols = f.cols();
		
		if( 1 != cols ){
			std::cerr << "error, column must be 1" << std::endl;
			std::abort();
		}

		to.set_size(rows);

		for(int i=0; i<rows; i++){
			to(i) = f.coeff(i);
		}
	}

	
	void dlib2eigen(column_vector& from, Eigen::VectorXd& to){
		int rows = from.nr();
		int cols = from.nc();

		if( 1 != cols ){
			std::cerr << "Error, dlib vector must be column vector" << std::endl;
			std::abort();
		}

		to.resize(rows);

		for(int i=0; i< rows; ++i){
			to.coeffRef(i) = from(i);
		}
	}

}

linearmodelopt::linearmodelopt(Parameter& param,
			       boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& insts,
			       boost::shared_ptr<Eigen::VectorXi>& labels) : param_(param), insts_(insts), labels_(labels)
{}

linearmodelopt::funcval::funcval(Parameter::LossFunc loss,
				 double l1c,
				 double l2c,
				 boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& insts,
				 boost::shared_ptr<Eigen::VectorXi>& labels): l1c_(l1c), l2c_(l2c), insts_(insts), labels_(label){

	loss_.reset( getLoss( loss ) );
	
	if(! loss_.get() ){
		std::cerr << "alloc loss function failed" << std::endl;
		std::abort();
	}
}

double linearmodelopt::funcval::operator()(const column_vector& w){
	
	int rows=0, cols=0;
	Eigen::VectorXd param;
	dlib2eigen(w, param);
	Eigen::VectorXd xw = (*insts_)*param;

	double fx = 0;
	int rows = insts_->rows();
	for(int i=0; i<rows; ++i){
		fx += loss_->loss( xw.coeff(i), labels_->coeff(i) );
	}

	return fx;
}


linearmodelopt::derivaval::derivaval(Parameter::LossFunc loss,
				     double L1c,
				     double l2c,
				     boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& insts
				     boost::shared_ptr<Eigen::VectorXi>& labels):l1c_(l1c), l2c_(l2c), insts_(insts), labels_(labels){
	loss_.reset( getloss( loss ) );

	if(! loss_.get() ){
		std::cerr << "alloc loss func failed" << std::endl;
		std::abort();
	}
}

const column_vector linearmodelopt::derivaval::operator()(const column_vector& w){
	
	Eigen::VectorXd param;

	dlib2eigen(w, param);
	Eigen::VectorXd xw = (*insts_)*param;

	Eigen::VectorXd grad( w.nr() );
	column_vector ret_grad;
	grad.setZero();

	for (int k = 0; k < optInst->samples_->outerSize(); ++k){

		double factor = loss_->dloss( xw.coeff(k), labels_->coeff(k) );

		if( std::abs(factor) < 1e-10 ){
			continue;
		}

		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it( *(insts_) ,k); 
		     it; ++it){
			int colIdx = it.col();
			grad.coeffRef(colIdx) += factor*it.value();
		}
	}

	eigen2dlib(ret_grad, grad);
	return ret_grad;
}


linearmodelopt::funcval* getFuncValObj(){
	return new funcval(param_.loss_, param.l1c_, param.l2c_, insts_, labels);
}

linearmodelopt::derivaval* getDerivaValObj(){

	return new derivaval(param_.loss_, param.l1c_, param.l2c_, insts_, labels_);
}
