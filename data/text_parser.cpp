/*
 * =====================================================================================
 *
 *       Filename:  text_parser.cpp
 *
 *    Description:  implementation of parser
 *
 *        Version:  1.0
 *        Created:  05/07/2018 19:57:40
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#include "data/text_parser.h"

namespace mltools {

void ExampleParser::init(TextFormat format, bool ignore_feag_grp){
    ignore_feat_grp_ = ignore_feag_grp;

    switch(format){
        case DataConfig::LIBSVM:
            parser_ = [this](char* line, Example* ex) -> bool {
                return parseLibsvm(line, ex);
            };
            break;
        case DataConfig::VW:
            parser_ = [this](char* line, Example* ex) -> bool {
                return parseVw(line, ex);
            };
            break;
        case DataConfig::CRITEO:
        case DataConfig::ADFEA:
        case DataConfig::TERAFEA:
        case DataConfig::DENSE:
        case DataConfig::SPARSE:
        case DataConfig::SPARSE_BINARY:
            {
                std::cerr << "Error, other formats not supported yet";
                std::abort();
            }
            break;
    }
}


}
