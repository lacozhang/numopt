#include <iostream>
#include <string>
#include <cstdlib>
#include <dlib/logger.h>
#include <dlib/misc_api.h>
#include <dlib/optimization.h>
#include <dlib/cmd_line_parser.h>

#include "parameter.h"


int main(int argc, char* argv[]){
  
  dlib::logger trainLog("train");
  trainLog.set_level(dlib::LALL);

  Parameter param;

  try {

    dlib::command_line_parser parser;

    parser.set_group_name("Traing data options");
    parser.add_option("in",  "input file we use", 1);
    parser.add_option("out", "output file we use", 1);

    parser.set_group_name("Optimization Related Options");
    parser.add_option("algo", "algorithms for optimization(LBFGS|CG|SGD|GD)", 1);
    parser.add_option("learn", "learning rate for SGD or GD", 1);

    parser.set_group_name("Loss function related options");
    parser.add_option("loss", "loss function(hinge|logistic|squared)", 1);

    parser.set_group_name("Regularization parameters options");
    parser.add_option("l1", "L1 regularization", 1);
    parser.add_option("l2", "L2 regularization", 1);

    parser.set_group_name("Miscellaneous Options");
    parser.add_option("h", "Display this message");

    parser.parse(argc, argv);

    const char* one_time_options[] = {"in", "out", "h", "algo", "learn", "loss"};
    parser.check_one_time_options(one_time_options);

    parser.check_incompatible_options("l1", "l2");

    const char* optim_algos[] = {"sgd", "gd", "lbfgs", "cg"};
    parser.check_option_arg_range("algo", optim_algos);

    const char* loss_funcs[] = {"hinge", "logistic", "squared"};
    parser.check_option_arg_range("loss", loss_funcs);

    parser.check_option_arg_type<double>("l1");
    parser.check_option_arg_type<double>("l2");
    parser.check_option_arg_type<double>("learn");
    
    /*
    const char* c_sub_opts[] = {"l"};
    parser.check_sub_options("c", c_sub_opts);
    */


    if( parser.option("h") ){
      parser.print_options();
      std::exit(-1);
    }


    if( parser.option("in") ){
      param.train = parser.option("in").argument();
    } else {
      std::cerr << "You must specify the input file" << std::endl;
    }

    if(parser.option("out")){
      param.model = parser.option("out").argument();
    } else {
      trainLog << dlib::LWARN << "you must specify the output file\n";
    }

    if(parser.option("l1")){
      param.l1 = std::atof(parser.option("l1").argument().c_str());
    }

    if(parser.option("l2")){
      param.l2 = std::atof(parser.option("l2").argument().c_str());
    }

    std::string opt;
    opt = parser.option("algo").argument();
    if( "gd" == opt){
      param.algo = GD;
    } else if("sgd" == opt){
      param.algo = SGD;
    } else if("cg" == opt){
      param.algo = CG;
    } else {
      param.algo = LBFGS;
    }

    std::string loss;
    loss = parser.option("loss").argument();
    if( "hinge" == loss ){
      param.loss = Hinge;
    } else if("logistic" == loss){
      param.loss = Logistic;
    } else {
      param.loss = Squared;
    }

    param.learningRate = std::atof(parser.option("learn").argument().c_str());
      
  } catch(std::exception& e){
    std::cerr << e.what() << std::endl;
    std::abort();
  }

  trainLog << dlib::LINFO << "Traing with following parameters";

  return 0;
}
