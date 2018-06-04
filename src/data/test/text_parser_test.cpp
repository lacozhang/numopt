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

#define BOOST_TEST_MODULE "TextParserTest"
#include "data/text_parser.h"
#include <boost/test/included/unit_test.hpp>
#include <boost/test/unit_test.hpp>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <string>

BOOST_AUTO_TEST_CASE(ReadLibSVM) {
  google::InitGoogleLogging("ReadLibSVM");
  mltools::ExampleParser parser;

  parser.init(mltools::DataConfig::LIBSVM, true);
  char line[] = "+1 20:3 39:2.3";
  mltools::Example ex;
  parser.toProto(line, &ex);

  BOOST_TEST(ex.slot_size() == 2);
  BOOST_TEST(ex.slot(1).key_size() == 2);
}
