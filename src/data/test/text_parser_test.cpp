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

#include "data/info_parser.h"
#include "data/text_parser.h"
#include "util/filelinereader.h"
#include "gtest/gtest.h"
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <string>

using namespace std;
using namespace mltools;

TEST(DataParser, SimpleLibSVM) {
  mltools::ExampleParser parser;
  parser.init(mltools::DataConfig::LIBSVM, true);
  char line[] = "+1 20:3 39:2.3";
  std::vector<uint64_t> keys = {20, 39};
  std::vector<double> vals = {3, 2.3};
  mltools::Example ex;
  ASSERT_TRUE(parser.toProto(line, &ex));
  ASSERT_EQ(ex.slot_size(), 2);
  ASSERT_EQ(ex.slot(0).val_size(), 1);
  ASSERT_EQ(ex.slot(0).key_size(), 0);
  ASSERT_EQ(ex.slot(0).val(0), 1);

  ASSERT_EQ(ex.slot(1).key_size(), keys.size());
  for (int i = 0; i < keys.size(); ++i) {
    ASSERT_EQ(ex.slot(1).key(i), keys[i]) << " at index " << i;
    EXPECT_TRUE(std::abs(ex.slot(1).val(i) - vals[i]) < 1e-5)
        << " at index " << i;
  }
}

TEST(DataParser, DefaultValueLibSVM) {
  mltools::ExampleParser parser;
  mltools::Example ex;
  parser.init(mltools::DataConfig::LIBSVM, false);
  char line[] = "-1 2 15 29:2.353 1920 2939:1.02 5000:-1.2";
  std::vector<int> keys = {2, 15, 29, 1920, 2939, 5000};
  std::vector<double> vals = {1.0, 1.0, 2.353, 1.0, 1.02, -1.2};
  ASSERT_TRUE(parser.toProto(line, &ex));
  ASSERT_EQ(ex.slot_size(), 2);
  ASSERT_EQ(ex.slot(0).key_size(), 0);
  ASSERT_EQ(ex.slot(0).val_size(), 1);
  ASSERT_EQ(ex.slot(0).val(0), -1);
  ASSERT_EQ(ex.slot(1).key_size(), keys.size());
  ASSERT_EQ(ex.slot(1).val_size(), vals.size());

  for (int i = 0; i < keys.size(); ++i) {
    ASSERT_EQ(ex.slot(1).key(i), keys[i]) << " at index " << i;
    EXPECT_TRUE(std::abs(ex.slot(1).val(i) - vals[i]) < 1e-5)
        << " at index " << i;
  }
}

TEST(DataParser, MetaDataParserInit) {
  mltools::InfoParser parser;
  parser.clear();
}

TEST(DataParser, ReadLineParser) {
  DataConfig cfg;
  cfg.add_file("/Users/edwinzhang/src/parameter_server/example/linear/data/"
               "rcv1/train/part-004");
  FileLineReader reader(cfg);
  int cnt = 0;
  ExampleParser parser;
  Example ex;

  parser.init(DataConfig::LIBSVM, true);
  auto handle = [&](char *line) {
    try {
      if (parser.toProto(line, &ex)) {
        cnt += 1;
      } else {
        CHECK(false) << "failed to parse example " << line;
      }
    } catch (std::exception &ex) {
      LOG(INFO) << "Exception " << ex.what();
    }
  };
  reader.setLineCallback(handle);
  reader.reload();
  CHECK(reader.loadedSuccessfully());
  LOG(INFO) << "read examples " << cnt << " successfully";
}
