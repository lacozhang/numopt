#include <iostream>
#include <string>
#include <cstdlib>
#include <dlib/logger.h>
#include <dlib/misc_api.h>
#include <dlib/optimization.h>
#include <dlib/cmd_line_parser.h>
#include "parameter.h"
#include "cmdline.h"


int main(int argc, char* argv[]){
  
    dlib::logger trainLog("train");
    trainLog.set_level(dlib::LALL);

    Parameter param;
    cmd_line_parse(argc, argv, param);

    trainLog << dlib::LINFO << "Traing with following parameters";
    std::cout << param;
  
    return 0;
}
