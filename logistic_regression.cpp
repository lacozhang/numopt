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
		u(i) = std::rand() / std::RAND_MAX;
	}
	
}


int main(int argc, char* argv[]){
  
	dlib::logger trainLog("train");
	trainLog.set_level(dlib::LALL);

	Parameter param;
	cmd_line_parse(argc, argv, param);

	trainLog << dlib::LINFO << "Traing with following parameters";
	std::cout << param;

	load_libsvm_data(param.train_,
			 trainSamples,
			 trainLabels);
	
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

	
	randominit(*parameters);

	dlib::find_min( dlib::lbfgs_search_stragety(10),
			dlib::objective_delta_stop_strategy(1e-7).be_verse(),
			
			*func,
			*der,
			*parameter,
			-1000);

    return 0;
}
