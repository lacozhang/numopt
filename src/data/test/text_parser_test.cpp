/*
 * =====================================================================================
 *
 *       Filename:  text_data.cpp
 *
 *    Description:  unit test for test data reading
 *
 *        Version:  1.0
 *        Created:  05/28/2018 19:54:10
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#define BOOST_TEST_MODULE "TextParser"
#include "data/text_parser.h"
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <string>

BOOST_AUTO_TEST_CASE(ReadLibSVM) {
  std::string line = "0 20:3 39:2.3";
  std::cout << "Test";
}
