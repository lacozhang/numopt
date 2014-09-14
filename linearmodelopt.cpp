#include <iostream>
#include <dlib/logger.h>
#include "linearmodelopt.h"
#include "util.h"

namespace {

	static timeutil timer;

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

	
	void dlib2eigen(const column_vector& from, Eigen::VectorXd& to){
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
                                 boost::shared_ptr<Eigen::VectorXi>& labels): l1c_(l1c), l2c_(l2c), insts_(insts), labels_(labels){

	loss_.reset( getLoss( loss ) );
	
	if(! loss_.get() ){
		std::cerr << "alloc loss function failed" << std::endl;
		std::abort();
	}
}

double linearmodelopt::funcval::operator() (const column_vector& w) const {
	
	dlib::logger log("train");
	log << dlib::LINFO << "compute the function value";
	int rows=0, cols=0;
	Eigen::VectorXd param;

	timer.tic();
	dlib2eigen(w, param);
	log << dlib::LINFO << "conversion costs " << timer.toc() << " seconds";
	timer.tic();
	Eigen::VectorXd xw = (*insts_)*param;
	log << dlib::LINFO << "sparse dense matrix vector product costs " << timer.toc() << " seconds";

	double fx = 0;
	rows = insts_->rows();
	for(int i=0; i<rows; ++i){
		fx += loss_->loss( xw.coeff(i), labels_->coeff(i) );
	}
	
	fx += 0.5*l2c_* param.squaredNorm();
	return fx;
}


linearmodelopt::derivaval::derivaval(Parameter::LossFunc loss,
                                     double l1c,
                                     double l2c,
                                     boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& insts,
                                     boost::shared_ptr<Eigen::VectorXi>& labels):l1c_(l1c), l2c_(l2c), insts_(insts), labels_(labels){
	loss_.reset( getLoss( loss ) );

	if(! loss_.get() ){
		std::cerr << "alloc loss func failed" << std::endl;
		std::abort();
	}
}

const column_vector linearmodelopt::derivaval::operator() (const column_vector& w) const {
	
	dlib::logger log("train");
	log << dlib::LINFO << "calculate the derivative";

	Eigen::VectorXd param;
	timer.tic();
	dlib2eigen(w, param);
	log << dlib::LINFO << "convertion costs " << timer.toc() << " seconds";

	timer.tic();
	Eigen::VectorXd xw = (*insts_)*param;
	log << dlib::LINFO << "m by v costs " << timer.toc() << " seconds";

	Eigen::VectorXd grad( w.nr() );
	column_vector ret_grad;

	timer.tic();
	grad.setZero();

	for (int k = 0; k < insts_->outerSize(); ++k){

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

	grad += l2c_ * param;

	log << dlib::LINFO << "calculate the gradient costs " << timer.toc() << " seconds";

	eigen2dlib(ret_grad, grad);
	return ret_grad;
}


linearmodelopt::funcval* linearmodelopt::getFuncValObj(){
	return new linearmodelopt::funcval(param_.loss_, param_.l1_, param_.l2_, insts_, labels_);
}

linearmodelopt::derivaval* linearmodelopt::getDerivaValObj(){

	return new linearmodelopt::derivaval(param_.loss_, param_.l1_, param_.l2_, insts_, labels_);
}

double linearmodelopt::gettrainaccu(boost::shared_ptr<column_vector>& w){

	Eigen::VectorXd param;
	dlib2eigen(*w, param);
	
	Eigen::VectorXd xw = (*insts_)*(param);
	double rc = 0, ac = 0;
	
	int nsample = insts_->rows();
	for(int i=0; i<nsample; ++i){
		if( xw.coeff(i) * labels_->coeff(i)  > 0 ){
			rc += 1;
		}

		ac += 1;
	}

	return rc / ac;
}
