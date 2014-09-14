#include <iostream>
#include <string>
#include <cstdlib>
#include <dlib/logger.h>
#include <dlib/misc_api.h>
#include <dlib/optimization.h>
#include <dlib/cmd_line_parser.h>
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "util.h"
#include "parameter.h"
#include "cmdline.h"
#include "dataop.h"
#include "linearmodelopt.h"

boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> > trainSamples;
boost::shared_ptr<Eigen::VectorXi> trainLabels;
boost::shared_ptr<linearmodelopt> modelopts;
boost::shared_ptr<column_vector> parameters;

void randominit(column_vector& u){
	int rows = u.nr();
	int cols = u.nc();

	std::srand(0);

	if( 1 != cols ){
		std::abort();
	}

	for(int i=0; i<rows; ++i){
		u(i) = std::rand() / RAND_MAX;
	}
	
}


int main(int argc, char* argv[]){
  
	
	timeutil t;
	dlib::logger trainLog("train");
	trainLog.set_level(dlib::LALL);

	Parameter param;
	cmd_line_parse(argc, argv, param);

	trainLog << dlib::LINFO << "Traing with following parameters";
	std::cout << param;

	trainLog << dlib::LINFO << "Loading data into memory";
	t.tic();
	load_libsvm_data(param.train_,
			 trainSamples,
			 trainLabels);
	trainLog << dlib::LINFO << "Loading data costs " << t.toc() << " seconds ";
	
	modelopts.reset( new linearmodelopt(param, trainSamples, trainLabels) );

	if( ! modelopts.get() ){
		std::cerr << "linear model optimization object alloc failed"
			  << std::endl;
		std::abort();
	}

	linearmodelopt::funcval* func = modelopts->getFuncValObj();
	linearmodelopt::derivaval* der = modelopts->getDerivaValObj();

	if((!func) || (!der)){
		std::cerr << "Alloc func/deriva failed" << std::endl;
		std::abort();
	}
	
	int featsize = trainSamples->cols();

	parameters.reset( new column_vector(featsize) );
	if(!parameters.get()){
		std::cerr << "parameter alloc failed" << std::endl;
		std::abort();
	}

	trainLog << dlib::LINFO << "init parameters with uniform random";
	t.tic();
	randominit(*parameters);
	trainLog << dlib::LINFO << "init paramters costs " << t.toc() << " seconds";
	trainLog << dlib::LINFO << "train with LBFGS";


	dlib::find_min(dlib::lbfgs_search_strategy(10),
		       dlib::objective_delta_stop_strategy(1e-7).be_verbose(),
		       *func,
		       *der,
		       *parameters,
		       -1);

	/* too slow
	trainLog << dlib::LINFO << "train with CG";
	dlib::find_min(dlib::cg_search_strategy(),
		       dlib::objective_delta_stop_strategy(1e-7).be_verbose(),
		       *func,
		       *der,
		       *parameters,
		       -1);
	*/

	double accu = modelopts->gettrainaccu(parameters);
	trainLog << dlib::LINFO << "train accuracy is " << accu ;
	return 0;
}
