#include "cmdline.h"
#include <dlib/cmd_line_parser.h>
#include <dlib/misc_api.h>

void cmd_line_parse(int argc, char* argv[], Parameter& param)
{
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
            param.train_ = parser.option("in").argument();
        } else {
            std::cerr << "You must specify the input file" << std::endl;
            std::exit(-1);
        }

        if(parser.option("out")){
            param.model_ = parser.option("out").argument();
        } else {
            std::cerr << "you must specify the output file\n";
            std::exit(-1);
        }

        if(parser.option("l1")){
            param.l1_ = std::atof(parser.option("l1").argument().c_str());
        }

        if(parser.option("l2")){
            param.l2_ = std::atof(parser.option("l2").argument().c_str());
        }

        
        if(!parser.option("algo")){
            std::cerr << "Please select optimization algorithm(gd|sgd|cg|lbfgs)"
                      << std::endl;
            std::exit(-1);
        }
        std::string opt;    
        opt = parser.option("algo").argument();
        if( "gd" == opt){
            param.algo_ = Parameter::GD;
        } else if("sgd" == opt){
            param.algo_ = Parameter::SGD;
        } else if("cg" == opt){
            param.algo_ = Parameter::CG;
        } else {
            param.algo_ = Parameter::LBFGS;
        }


        if(!parser.option("loss")){
            std::cerr << "Must specify a Loss function(hinge|logistic|squared)"
                      << std::endl;
            std::exit(-1);
        }
        
        std::string loss;
        loss = parser.option("loss").argument();
        if( "hinge" == loss ){
		param.loss_ = Parameter::Hinge;
        } else if("logistic" == loss){
		param.loss_ = Parameter::Logistic;
        } else {
		param.loss_ = Parameter::Squared;
        }

	if( parser.option("learn") ){
		param.learningRate_ = std::atof(parser.option("learn").argument().c_str());
	}
      
    } catch(std::exception& e){
        std::cerr << e.what() << std::endl;
        std::abort();
    }

}

